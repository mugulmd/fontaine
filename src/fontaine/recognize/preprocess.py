"""Turning a crop into an ink mask.

Two things have to be handled before any measurement makes sense:

* **polarity.** Crops are light-on-dark as often as dark-on-light, and every
  feature would flip sign between the two. The mask is always ink-is-true.
* **scale.** Crops arrive at any size, and absolute size is deliberately not a
  clue to the font, so the mask is normalized to a fixed ink height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Height the ink is normalized to. Big enough for stroke widths to survive, small
#: enough that a whole stream can be processed at a useful rate.
INK_HEIGHT = 48


@dataclass(frozen=True, slots=True)
class InkMask:
    """A boolean mask where true is ink, normalized to :data:`INK_HEIGHT`."""

    mask: np.ndarray
    #: Whether the original crop was dark text on a light background.
    dark_on_light: bool
    #: Size of the crop this came from, before normalization.
    source_size: tuple[int, int]

    @property
    def height(self) -> int:
        """Rows in the mask, always :data:`INK_HEIGHT` for a non-empty crop."""
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        """Columns in the mask, which vary with how much text the crop holds."""
        return int(self.mask.shape[1])

    @property
    def empty(self) -> bool:
        """Whether no ink was found at all, which no feature can describe."""
        return not bool(self.mask.any())


def _otsu_threshold(gray: np.ndarray) -> float:
    """The grey level that best separates the image into two groups.

    Otsu's method: pick the threshold minimising within-group variance. Written out
    rather than pulled from a dependency — it is six lines on a histogram.
    """
    counts = np.bincount(gray.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    weights = np.cumsum(counts)
    total = weights[-1]
    if total == 0:
        return 128.0
    levels = np.arange(256, dtype=np.float64)
    sums = np.cumsum(counts * levels)
    # Between-group variance for every candidate threshold, in one vectorised pass.
    foreground = weights
    background = total - weights
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_fg = np.where(foreground > 0, sums / foreground, 0.0)
        mean_bg = np.where(background > 0, (sums[-1] - sums) / background, 0.0)
        variance = foreground * background * (mean_fg - mean_bg) ** 2
    return float(np.nanargmax(variance))


def ink_mask(image: Image.Image, ink_height: int = INK_HEIGHT) -> InkMask:
    """Extract a normalized ink mask from a crop.

    Polarity is decided from the border, on the assumption that the outer ring of a
    text-box crop is mostly background. Where that ring is itself mostly ink — a
    crop padded so tightly it clips the glyphs — it falls back to treating ink as
    the minority, which is true of text at any reasonable size.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    below = gray <= threshold

    border = np.concatenate(
        [below[0, :].ravel(), below[-1, :].ravel(), below[:, 0].ravel(), below[:, -1].ravel()]
    )
    border_is_mixed = 0.2 < border.mean() < 0.8
    dark_on_light = (
        # A mixed border says little, so fall back to ink being the minority.
        bool(below.mean() <= 0.5)
        if border_is_mixed
        # Otherwise the border is background, and ink is whatever it is not.
        else bool(border.mean() < 0.5)
    )

    mask = below if dark_on_light else ~below
    mask = _crop_to_ink(mask)
    return InkMask(
        mask=_normalize_height(mask, ink_height),
        dark_on_light=dark_on_light,
        source_size=(image.width, image.height),
    )


def _crop_to_ink(mask: np.ndarray) -> np.ndarray:
    """Trim to the ink's bounding box, so padding jitter cannot shift the features."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return mask
    return mask[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def _normalize_height(mask: np.ndarray, ink_height: int) -> np.ndarray:
    """Resample to a fixed ink height, preserving the aspect ratio."""
    height, width = mask.shape
    if height == 0 or width == 0:
        return np.zeros((ink_height, ink_height), dtype=bool)
    if height == ink_height:
        return mask
    scaled_width = max(1, round(width * ink_height / height))
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (scaled_width, ink_height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8) > 127
