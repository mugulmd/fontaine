"""Data contracts shared by the generator and the online recognizer.

This is the only module both programs are expected to import. Everything else
is private to one side or the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

LabelGranularity = Literal["face", "family"]


@dataclass(frozen=True, slots=True)
class FontFace:
    """A single renderable font face: one file, or one index inside a ``.ttc``.

    ``face_id`` is the classification label at face granularity; ``family_id``
    is the label at family granularity. Both are stable slugs derived from the
    font's ``name`` table, so they survive the file being moved or renamed.
    """

    face_id: str
    family_id: str
    family: str
    subfamily: str
    postscript_name: str | None
    path: Path
    font_number: int
    weight: int
    width_class: int
    italic: bool
    monospace: bool
    variable: bool
    units_per_em: int
    n_glyphs: int
    codepoints: frozenset[int] = field(repr=False, compare=False, default=frozenset())

    def label(self, granularity: LabelGranularity) -> str:
        """This face's class name at the requested granularity."""
        return self.face_id if granularity == "face" else self.family_id

    def covers(self, text: str) -> bool:
        """Whether every character of ``text`` has a glyph in this face."""
        return all(ord(char) in self.codepoints for char in text)

    def missing_from(self, text: str) -> str:
        """The characters of ``text`` this face has no glyph for, deduplicated."""
        seen: dict[str, None] = {}
        for char in text:
            if ord(char) not in self.codepoints:
                seen[char] = None
        return "".join(seen)

    def to_dict(self) -> dict[str, Any]:
        """Serializable view. ``codepoints`` is dropped — it is derivable from the file."""
        return {
            "face_id": self.face_id,
            "family_id": self.family_id,
            "family": self.family,
            "subfamily": self.subfamily,
            "postscript_name": self.postscript_name,
            "path": str(self.path),
            "font_number": self.font_number,
            "weight": self.weight,
            "width_class": self.width_class,
            "italic": self.italic,
            "monospace": self.monospace,
            "variable": self.variable,
            "units_per_em": self.units_per_em,
            "n_glyphs": self.n_glyphs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FontFace:
        """Rebuild a face from :meth:`to_dict` output."""
        return cls(
            face_id=data["face_id"],
            family_id=data["family_id"],
            family=data["family"],
            subfamily=data["subfamily"],
            postscript_name=data["postscript_name"],
            path=Path(data["path"]),
            font_number=data["font_number"],
            weight=data["weight"],
            width_class=data["width_class"],
            italic=data["italic"],
            monospace=data["monospace"],
            variable=data["variable"],
            units_per_em=data["units_per_em"],
            n_glyphs=data["n_glyphs"],
        )


@dataclass(slots=True)
class Sample:
    """One item of the stream: a crop around a text box, and the font that drew it.

    ``metadata`` carries the full generation parameters (px size, contrast,
    background source, crop jitter, ...) so failures can be sliced after the
    fact. The recognizer must never read it — only ``image`` at prediction time
    and ``label`` when it is subsequently allowed to learn.
    """

    index: int
    image: Image.Image
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
