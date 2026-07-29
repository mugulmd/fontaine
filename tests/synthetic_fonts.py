"""Builders for minimal but valid fonts.

The registry tests build their own fonts rather than reading ``assets/fonts``, so
they assert on known glyph coverage and metadata and pass on any machine.
"""

from __future__ import annotations

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTCollection, TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPEM = 1000


def build_font(
    path: Path,
    *,
    chars: str,
    family: str = "Test Family",
    subfamily: str = "Regular",
    weight: int = 400,
    italic: bool = False,
    glyph_height: int = 700,
) -> Path:
    """Write a minimal but valid TTF whose cmap covers exactly ``chars``.

    ``glyph_height`` is in font units out of a 1000 em, so two fonts can differ in
    apparent size at the same em size — which is what cap-height normalization is
    there to cancel out.
    """
    font = _make_font(
        chars=chars,
        family=family,
        subfamily=subfamily,
        weight=weight,
        italic=italic,
        glyph_height=glyph_height,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    font.save(path)
    return path


def build_collection(path: Path, faces: list[dict]) -> Path:
    """Write a ``.ttc`` bundling one face per entry in ``faces``."""
    collection = TTCollection()
    collection.fonts = [_make_font(**face) for face in faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    collection.save(str(path))
    return path


def _make_font(
    *,
    chars: str,
    family: str = "Test Family",
    subfamily: str = "Regular",
    weight: int = 400,
    italic: bool = False,
    glyph_height: int = 700,
) -> TTFont:
    builder = FontBuilder(UPEM, isTTF=True)
    glyph_names = {char: f"uni{ord(char):04X}" for char in dict.fromkeys(chars)}
    order = [".notdef", *glyph_names.values()]
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({ord(char): name for char, name in glyph_names.items()})

    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, glyph_height))
    pen.lineTo((450, glyph_height))
    pen.lineTo((450, 0))
    pen.closePath()
    box = pen.glyph()
    builder.setupGlyf({name: box for name in order})
    builder.setupHorizontalMetrics({name: (500, 50) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(
        {
            "familyName": family,
            "styleName": subfamily,
            "uniqueFontIdentifier": f"{family}-{subfamily}",
            "fullName": f"{family} {subfamily}",
            "psName": f"{family}-{subfamily}".replace(" ", ""),
            "version": "1.0",
        }
    )
    builder.setupOS2(usWeightClass=weight, fsSelection=1 if italic else 0)
    builder.setupPost()
    if italic:
        builder.font["head"].macStyle |= 1 << 1
    return builder.font
