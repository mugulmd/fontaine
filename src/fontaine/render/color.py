"""Colour picking driven by contrast, not by taste.

Text colour is chosen *relative to the background under it*: sample a target
contrast ratio, then solve for a colour that achieves it. Picking both colours
independently would put a large share of items somewhere between invisible and
trivially crisp, with no control over where.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass

import numpy as np

RGB = tuple[int, int, int]

# sRGB relative luminance weights (Rec. 709).
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def srgb_to_linear(channel: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(channel, dtype=np.float64)
    return np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1 / 2.4) - 0.055


def relative_luminance(rgb: RGB) -> float:
    """WCAG relative luminance of an 8-bit sRGB colour, in [0, 1]."""
    linear = srgb_to_linear(np.array(rgb, dtype=np.float64) / 255.0)
    return float(np.dot(np.asarray(linear), _LUMA_WEIGHTS))


def luminance_map(patch: np.ndarray) -> np.ndarray:
    """Per-pixel relative luminance of an ``(h, w, 3)`` uint8 array."""
    linear = srgb_to_linear(patch.astype(np.float64) / 255.0)
    return np.tensordot(np.asarray(linear), _LUMA_WEIGHTS, axes=([2], [0]))


def mean_luminance(patch: np.ndarray) -> float:
    return float(luminance_map(patch).mean())


def luminance_extremes(patch: np.ndarray, percentile: float = 12.0) -> tuple[float, float]:
    """The dark and light ends of a background patch, as ``(low, high)``.

    Targeting the mean is not enough: over a background with an edge running
    through the text box, dark text can be perfectly readable on the light half
    and invisible on the dark half. Dark text must therefore be matched against
    the dark end and light text against the light end, so the requested contrast
    holds across the whole box rather than on average.

    Percentiles rather than true extrema, so one stray pixel cannot dictate the
    colour of every crop.
    """
    luminance = luminance_map(patch)
    low, high = np.percentile(luminance, (percentile, 100.0 - percentile))
    return float(low), float(high)


def contrast_ratio(first: float, second: float) -> float:
    """WCAG contrast ratio between two relative luminances."""
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _solve_luminance(reference: float, ratio: float, darker: bool) -> float:
    """Luminance achieving ``ratio`` against ``reference``, clamped to the gamut.

    Deliberately does not switch sides when the ratio is unreachable — the caller
    compares both sides, which is the only way to be sure the chosen one is
    actually the better of the two.
    """
    if darker:
        return max(0.0, (reference + 0.05) / ratio - 0.05)
    return min(1.0, ratio * (reference + 0.05) - 0.05)


def guaranteed_contrast(extremes: tuple[float, float], luminance: float) -> tuple[float, float]:
    """The weakest contrast ``luminance`` has anywhere in the background range.

    Returns ``(ratio, reference)`` for whichever end of the range is hardest. This
    is the number that matters: contrast against the mean, or against the
    favourable end, says nothing about whether the text is readable throughout.
    """
    low, high = extremes
    against_low = contrast_ratio(low, luminance)
    against_high = contrast_ratio(high, luminance)
    if against_low <= against_high:
        return against_low, low
    return against_high, high


def _neutral_with_luminance(target: float) -> RGB:
    # For a grey, the luminance weights sum to 1, so the linear value *is* the
    # luminance — no search needed.
    channel = round(255 * linear_to_srgb(min(1.0, max(0.0, target))))
    return (channel, channel, channel)


def _hued_with_luminance(rng: random.Random, target: float) -> RGB:
    """A coloured hue moved in linear light to land exactly on ``target``.

    A hue at full brightness has a fixed luminance ceiling — pure blue tops out
    around 0.07 — so scaling it up towards a bright target clamps channels and
    lands far short of the request. Above the ceiling the hue is mixed towards
    white instead, which reaches any luminance. Saturation is what gives way,
    since contrast is the property that has to hold.
    """
    hue = rng.random()
    saturation = rng.uniform(0.35, 1.0)
    base = colorsys.hsv_to_rgb(hue, saturation, 1.0)
    base_linear = np.asarray(srgb_to_linear(np.array(base, dtype=np.float64)))
    ceiling = float(np.dot(base_linear, _LUMA_WEIGHTS))
    target = min(1.0, max(0.0, target))
    if ceiling <= 0.0:
        return _neutral_with_luminance(target)
    if target <= ceiling:
        # Darkening is exact: luminance is linear in a uniform scale factor.
        linear = base_linear * (target / ceiling)
    else:
        # Luminance is also linear along the path to white, so solve for the mix.
        mix = (target - ceiling) / (1.0 - ceiling) if ceiling < 1.0 else 0.0
        linear = base_linear + mix * (1.0 - base_linear)
    return tuple(round(255 * linear_to_srgb(float(value))) for value in np.clip(linear, 0.0, 1.0))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ContrastPlan:
    """A text luminance chosen against a background's luminance range.

    Kept separate from colour selection so the decision can be inspected — and a
    scrim inserted — before any randomness is spent on hue.
    """

    #: Target relative luminance for the text.
    luminance: float
    #: Whether the text ends up darker than the background range it was matched
    #: against. May differ from the request: a near-black background cannot host
    #: darker text at high contrast, so the side is abandoned, not the contrast.
    darker: bool
    #: The background luminance where this text is hardest to read.
    reference: float
    #: Contrast at that hardest point, and so the contrast guaranteed across the
    #: whole text box. Falls short of the request when the background range is too
    #: wide for any single colour, or when the colour would leave the sRGB gamut.
    achieved: float


def plan_contrast(extremes: tuple[float, float], *, ratio: float, darker: bool) -> ContrastPlan:
    """Solve for a text luminance at ``ratio`` against the harder end of ``extremes``.

    Both sides are solved and compared. The requested side is kept when it reaches
    the ratio; otherwise the side with the better guaranteed contrast wins, since a
    background spanning a wide range can make one side hopeless — that is the
    signal to put a scrim behind the text rather than to accept an illegible crop.
    """
    low, high = extremes
    candidates = []
    for side in (darker, not darker):
        target = _solve_luminance(low if side else high, ratio, side)
        achieved, reference = guaranteed_contrast(extremes, target)
        candidates.append(ContrastPlan(target, side, reference, achieved))

    preferred, alternative = candidates
    if preferred.achieved >= ratio * 0.995 or preferred.achieved >= alternative.achieved:
        return preferred
    return alternative


def color_with_luminance(rng: random.Random, target: float, *, saturated: bool) -> RGB:
    """A colour at (approximately) the requested relative luminance."""
    return _hued_with_luminance(rng, target) if saturated else _neutral_with_luminance(target)


def random_color(rng: random.Random) -> RGB:
    """An arbitrary colour, biased away from full saturation."""
    hue = rng.random()
    saturation = rng.betavariate(1.4, 3.0)
    value = rng.uniform(0.15, 1.0)
    return tuple(round(255 * channel) for channel in colorsys.hsv_to_rgb(hue, saturation, value))  # type: ignore[return-value]
