"""Discovery of the font universe: walk a directory, extract face metadata.

The registry is the ground truth for the label space. It is snapshotted into the
stream manifest so a generated stream stays interpretable even if
``assets/fonts`` later changes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterator

from fontTools.ttLib import TTCollection, TTFont, TTLibError

from fontaine.contracts import FontFace, LabelGranularity
from fontaine.fonts.coverage import resolve_charset

FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".ttc", ".otc"})
COLLECTION_EXTENSIONS = frozenset({".ttc", ".otc"})

# name table IDs, typographic variants first — they are the ones that split
# "Roboto / Condensed Bold" correctly instead of "Roboto Condensed / Bold".
_NAME_FAMILY = (16, 1)
_NAME_SUBFAMILY = (17, 2)
_NAME_POSTSCRIPT = (6,)

_FS_SELECTION_ITALIC = 1 << 0
_MAC_STYLE_ITALIC = 1 << 1

#: Fallback weights when OS/2 usWeightClass is absent or out of spec.
_WEIGHT_KEYWORDS: tuple[tuple[str, int], ...] = (
    ("extrablack", 950),
    ("ultrablack", 950),
    ("black", 900),
    ("heavy", 900),
    ("extrabold", 800),
    ("ultrabold", 800),
    ("semibold", 600),
    ("demibold", 600),
    ("bold", 700),
    ("medium", 500),
    ("book", 400),
    ("regular", 400),
    ("normal", 400),
    ("semilight", 350),
    ("light", 300),
    ("extralight", 200),
    ("ultralight", 200),
    ("thin", 100),
    ("hairline", 100),
)


@dataclass(frozen=True, slots=True)
class RejectedFace:
    """A face that was parsed but excluded from the label space."""

    face: FontFace
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class UnreadableFile:
    """A file under the font dir that could not be parsed at all."""

    path: Path
    error: str


@dataclass(slots=True)
class FontRegistry:
    """The resolved label space, plus everything that was left out and why."""

    faces: list[FontFace]
    label_granularity: LabelGranularity = "face"
    charset: str = ""
    rejected: list[RejectedFace] = field(default_factory=list)
    unreadable: list[UnreadableFile] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self) -> Iterator[FontFace]:
        return iter(self.faces)

    @property
    def labels(self) -> list[str]:
        """The distinct labels at the configured granularity, in registry order."""
        return list(dict.fromkeys(face.label(self.label_granularity) for face in self.faces))

    @property
    def families(self) -> list[str]:
        return list(dict.fromkeys(face.family_id for face in self.faces))

    def by_face_id(self) -> dict[str, FontFace]:
        return {face.face_id: face for face in self.faces}

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_granularity": self.label_granularity,
            "charset": self.charset,
            "faces": [face.to_dict() for face in self.faces],
            "rejected": [
                {"face_id": item.face.face_id, "reason": item.reason, "detail": item.detail}
                for item in self.rejected
            ],
            "unreadable": [
                {"path": str(item.path), "error": item.error} for item in self.unreadable
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FontRegistry:
        """Rebuild a registry from a manifest snapshot.

        Codepoint sets are not restored — a replayed stream needs the labels, not
        the ability to re-render.
        """
        return cls(
            faces=[FontFace.from_dict(item) for item in data["faces"]],
            label_granularity=data.get("label_granularity", "face"),
            charset=data.get("charset", ""),
        )


@contextmanager
def quiet_fonttools(quiet: bool = True) -> Iterator[None]:
    """Suppress fontTools' per-table warnings.

    Real-world fonts trip a steady stream of cosmetic complaints ("extra bytes in
    post.stringData", odd ``head.created`` timestamps) that say nothing about
    whether the face is usable. Pass ``quiet=False`` to hear them.
    """
    logger = logging.getLogger("fontTools")
    previous = logger.level
    if quiet:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "unknown"


def _name(font: TTFont, name_ids: tuple[int, ...]) -> str | None:
    table = font.get("name")
    if table is None:
        return None
    for name_id in name_ids:
        # getDebugName already walks platforms in a sensible preference order.
        value = table.getDebugName(name_id)
        if value and value.strip():
            return value.strip()
    return None


def _is_italic(font: TTFont) -> bool:
    os2 = font.get("OS/2")
    if os2 is not None and getattr(os2, "fsSelection", 0) & _FS_SELECTION_ITALIC:
        return True
    head = font.get("head")
    if head is not None and getattr(head, "macStyle", 0) & _MAC_STYLE_ITALIC:
        return True
    post = font.get("post")
    if post is not None and abs(getattr(post, "italicAngle", 0.0) or 0.0) > 0.5:
        return True
    return False


def _weight(font: TTFont, subfamily: str) -> int:
    os2 = font.get("OS/2")
    declared = getattr(os2, "usWeightClass", None) if os2 is not None else None
    if isinstance(declared, int) and 1 <= declared <= 1000:
        return declared
    # Some hand-built fonts declare 0 or garbage; fall back to the style name.
    normalized = re.sub(r"[^a-z]", "", subfamily.lower())
    for keyword, weight in _WEIGHT_KEYWORDS:
        if keyword in normalized:
            return weight
    return 400


def _codepoints(font: TTFont) -> frozenset[int]:
    try:
        return frozenset(font.getBestCmap().keys())
    except (AttributeError, KeyError, TTLibError, AssertionError):
        return frozenset()


def _read_faces(path: Path) -> list[FontFace]:
    """Extract every face in a font file.

    Metadata is pulled eagerly and the file closed before returning: the fonts
    are opened lazily, so a ``TTFont`` outliving its file handle would fail on
    the next table access.
    """
    if path.suffix.lower() in COLLECTION_EXTENSIONS:
        collection = TTCollection(path, lazy=True)
        try:
            return [
                _face_from_font(font, path, font_number)
                for font_number, font in enumerate(collection.fonts)
            ]
        finally:
            collection.close()
    font = TTFont(path, fontNumber=0, lazy=True)
    try:
        return [_face_from_font(font, path, 0)]
    finally:
        font.close()


def _face_from_font(font: TTFont, path: Path, font_number: int) -> FontFace:
    family = _name(font, _NAME_FAMILY) or path.stem
    subfamily = _name(font, _NAME_SUBFAMILY) or "Regular"
    head = font.get("head")
    post = font.get("post")
    os2 = font.get("OS/2")
    return FontFace(
        # face_id is finalized by the caller, which owns collision resolution.
        face_id=f"{_slug(family)}:{_slug(subfamily)}",
        family_id=_slug(family),
        family=family,
        subfamily=subfamily,
        postscript_name=_name(font, _NAME_POSTSCRIPT),
        path=path,
        font_number=font_number,
        weight=_weight(font, subfamily),
        width_class=getattr(os2, "usWidthClass", 5) if os2 is not None else 5,
        italic=_is_italic(font),
        monospace=bool(getattr(post, "isFixedPitch", 0)) if post is not None else False,
        variable="fvar" in font,
        units_per_em=getattr(head, "unitsPerEm", 1000) if head is not None else 1000,
        n_glyphs=len(font.getGlyphOrder()),
        codepoints=_codepoints(font),
    )


def iter_font_files(font_dir: Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Font files under ``font_dir``, sorted so scans are deterministic."""
    paths = [
        path
        for path in sorted(font_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
    ]
    if not exclude:
        return paths
    return [
        path
        for path in paths
        if not any(fnmatch(path.name, pattern) or fnmatch(str(path), pattern) for pattern in exclude)
    ]


def scan(
    font_dir: Path,
    *,
    charset: str = "ascii_printable",
    label_granularity: LabelGranularity = "face",
    exclude: tuple[str, ...] = (),
    include_variable: bool = False,
    verbose: bool = False,
) -> FontRegistry:
    """Build the label space from the font files under ``font_dir``.

    A face is rejected — kept in ``registry.rejected`` rather than dropped
    silently — when it lacks a glyph for any character of ``charset``, or when it
    is a variable font and ``include_variable`` is false (v1 renders static
    instances only, so a variable file would be one arbitrary default face).
    """
    if not font_dir.is_dir():
        raise NotADirectoryError(f"font dir does not exist: {font_dir}")

    required = resolve_charset(charset)
    faces: list[FontFace] = []
    rejected: list[RejectedFace] = []
    unreadable: list[UnreadableFile] = []
    used_ids: Counter[str] = Counter()

    for path in iter_font_files(font_dir, exclude):
        try:
            with quiet_fonttools(not verbose):
                parsed = _read_faces(path)
        except (TTLibError, OSError, ValueError, KeyError, IndexError, AssertionError) as error:
            unreadable.append(UnreadableFile(path, f"{type(error).__name__}: {error}"))
            continue

        for face in parsed:
            used_ids[face.face_id] += 1
            occurrence = used_ids[face.face_id]
            if occurrence > 1:
                # Two files claiming the same family+style. Keep both, disambiguated,
                # rather than letting one shadow the other.
                face = replace(face, face_id=f"{face.face_id}#{occurrence}")

            if face.variable and not include_variable:
                rejected.append(RejectedFace(face, "variable-font", "static instances only in v1"))
                continue
            missing = face.missing_from(required)
            if missing:
                rejected.append(
                    RejectedFace(face, "missing-glyphs", f"{len(missing)} missing: {missing[:24]}")
                )
                continue
            faces.append(face)

    return FontRegistry(
        faces=faces,
        label_granularity=label_granularity,
        charset=charset,
        rejected=rejected,
        unreadable=unreadable,
    )
