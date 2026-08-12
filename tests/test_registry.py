from __future__ import annotations

import string
from pathlib import Path

import pytest
from synthetic_fonts import build_font

from fontaine.fonts import registry as font_registry
from fontaine.fonts.coverage import CHARSET_PRESETS, resolve_charset

ALNUM = string.ascii_letters + string.digits


def test_scan_extracts_face_metadata(font_dir: Path) -> None:
    build_font(font_dir / "regular.ttf", chars=ALNUM, family="Acme Grotesk", subfamily="Regular")
    build_font(
        font_dir / "bold-italic.ttf",
        chars=ALNUM,
        family="Acme Grotesk",
        subfamily="Bold Italic",
        weight=700,
        italic=True,
    )

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    assert len(registry) == 2
    assert registry.labels == ["acme-grotesk:bold-italic", "acme-grotesk:regular"]

    bold = registry.by_face_id()["acme-grotesk:bold-italic"]
    assert bold.weight == 700
    assert bold.italic is True
    assert bold.family == "Acme Grotesk"
    assert bold.units_per_em == 1000
    assert registry.by_face_id()["acme-grotesk:regular"].italic is False


def test_every_face_is_its_own_label(font_dir: Path) -> None:
    build_font(font_dir / "a.ttf", chars=ALNUM, family="Acme", subfamily="Regular")
    build_font(font_dir / "b.ttf", chars=ALNUM, family="Acme", subfamily="Bold", weight=700)

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    # Registry order follows the file scan, so a.ttf comes before b.ttf.
    assert registry.labels == ["acme:regular", "acme:bold"]


def test_faces_missing_glyphs_are_rejected_not_dropped(font_dir: Path) -> None:
    build_font(font_dir / "full.ttf", chars=ALNUM, family="Full", subfamily="Regular")
    build_font(font_dir / "partial.ttf", chars="AB", family="Partial", subfamily="Regular")

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    assert registry.labels == ["full:regular"]
    assert len(registry.rejected) == 1
    rejected = registry.rejected[0]
    assert rejected.face.face_id == "partial:regular"
    assert rejected.reason == "missing-glyphs"
    assert "missing" in rejected.detail


def test_duplicate_family_and_style_get_disambiguated(font_dir: Path) -> None:
    build_font(font_dir / "one.ttf", chars=ALNUM, family="Twin", subfamily="Regular")
    build_font(font_dir / "two.ttf", chars=ALNUM, family="Twin", subfamily="Regular")

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    assert registry.labels == ["twin:regular", "twin:regular#2"]


def test_font_collections_are_not_read_at_all(font_dir: Path) -> None:
    build_font(font_dir / "solo.ttf", chars=ALNUM, family="Solo", subfamily="Regular")
    (font_dir / "bundle.ttc").write_bytes(b"ttcf")

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    # Not merely rejected: a collection is outside the scanned extensions, so it
    # never reaches the parser and is not reported as an unreadable file either.
    assert registry.labels == ["solo:regular"]
    assert registry.rejected == []
    assert registry.unreadable == []


def test_unreadable_files_are_reported(font_dir: Path) -> None:
    font_dir.mkdir(parents=True)
    (font_dir / "broken.ttf").write_bytes(b"not a font at all")

    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    assert len(registry) == 0
    assert [item.path.name for item in registry.unreadable] == ["broken.ttf"]


def test_exclude_patterns_skip_files(font_dir: Path) -> None:
    build_font(font_dir / "keep.ttf", chars=ALNUM, family="Keep", subfamily="Regular")
    build_font(font_dir / "skip-me.ttf", chars=ALNUM, family="Skip", subfamily="Regular")

    registry = font_registry.scan(font_dir, charset="ascii_alnum", exclude=("skip-*",))

    assert registry.labels == ["keep:regular"]
    assert registry.rejected == []


def test_scan_order_is_deterministic(font_dir: Path) -> None:
    for index in range(5):
        build_font(
            font_dir / f"face-{index}.ttf", chars=ALNUM, family=f"Fam{index}", subfamily="Regular"
        )

    first = font_registry.scan(font_dir, charset="ascii_alnum").labels
    second = font_registry.scan(font_dir, charset="ascii_alnum").labels

    assert first == second == [f"fam{index}:regular" for index in range(5)]


def test_missing_font_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        font_registry.scan(tmp_path / "nope")


def test_registry_snapshot_roundtrip(font_dir: Path) -> None:
    build_font(font_dir / "a.ttf", chars=ALNUM, family="Acme", subfamily="Regular")

    registry = font_registry.scan(font_dir, charset="ascii_alnum")
    snapshot = registry.model_dump(mode="json")
    restored = font_registry.FontRegistry.model_validate(snapshot)

    assert restored.labels == registry.labels
    assert restored.charset == registry.charset
    # Codepoints are deliberately not in the snapshot: a replayed stream needs the
    # labels, not the ability to re-render.
    assert "codepoints" not in snapshot["faces"][0]
    assert restored.faces[0].codepoints == frozenset()


def test_resolve_charset_expands_presets_and_drops_whitespace() -> None:
    assert resolve_charset("ascii_alnum") == ALNUM
    assert resolve_charset("ab c\nd") == "abcd"
    assert " " not in resolve_charset("ascii_printable")
    assert all(preset for preset in CHARSET_PRESETS.values())
