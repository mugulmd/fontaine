"""Font loading and size normalization.

Sizes are expressed as a target **cap height in pixels**, not as an em size. Two
faces set at the same em size can differ by 30% in apparent size, which would
make absolute scale a cue for the label — a leak the recognizer would happily
exploit instead of learning letterforms. Normalizing by cap height removes it.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import ImageFont

from fontaine.contracts import FontFace

#: Em size used to measure a face's proportions. Large enough that rounding in
#: the rasterizer does not skew the ratio.
PROBE_EM = 256

#: Characters tried, in order, to measure cap height.
_CAP_PROBES = ("H", "E", "X", "h", "x", "0")


@lru_cache(maxsize=512)
def _load(path: str, font_number: int, em_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=em_px, index=font_number)


def load_font(face: FontFace, em_px: int) -> ImageFont.FreeTypeFont:
    """Load a face at an em size in pixels. Cached — loading dominates otherwise."""
    return _load(str(face.path), face.font_number, max(1, int(em_px)))


@lru_cache(maxsize=512)
def _cap_ratio(path: str, font_number: int) -> float:
    font = _load(path, font_number, PROBE_EM)
    for probe in _CAP_PROBES:
        box = font.getbbox(probe)
        height = box[3] - box[1]
        if height > 0:
            return height / PROBE_EM
    raise ValueError(f"cannot measure cap height for {path}[{font_number}]")


def cap_height_ratio(face: FontFace) -> float:
    """Cap height as a fraction of em size, measured from the rasterized glyph.

    Measured rather than read from ``OS/2.sCapHeight``, which many fonts either
    omit or fill in wrongly.
    """
    return _cap_ratio(str(face.path), face.font_number)


def em_size_for_cap_height(face: FontFace, cap_height_px: float) -> int:
    """The em size in pixels that renders ``face`` at the requested cap height."""
    return max(4, round(cap_height_px / cap_height_ratio(face)))
