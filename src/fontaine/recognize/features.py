"""Describing a crop as numbers a river model can learn from.

Every feature is scale-free and none of them reads the text: the letterforms have
to carry the signal, or a model would be learning the corpus instead of the font.

What the measurements are reaching for, on a real twelve-font pool:

* **weight** — ink density and stroke thickness separate Anton from Roboto
* **set width** — column rhythm separates Oswald (condensed) from Montserrat (wide)
* **vertical proportions** — the row profile carries x-height against cap height,
  which is what makes Bebas Neue, with no true lowercase, trivially separable
* **stroke contrast** — thickness *variance* separates Playfair's hairlines from
  Ubuntu's even strokes, where set width alone cannot
* **regularity** — column-profile autocorrelation is what betrays a monospace
"""

from __future__ import annotations

import itertools

import numpy as np

from fontaine.recognize.preprocess import InkMask, ink_mask

try:  # pragma: no cover - exercised by whichever branch the install provides
    from scipy.ndimage import distance_transform_edt

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

#: Number of bins the row and column ink profiles are resampled to.
ROW_BINS = 12
COLUMN_BINS = 8
#: Shear angles tried when estimating slant, in degrees.
SLANT_ANGLES = (-24, -18, -12, -6, 0, 6, 12, 18, 24)


def describe(image) -> dict[str, float]:
    """Feature vector for one crop, as the dict river expects.

    Takes an image and nothing else — deliberately not a ``Sample``, whose metadata
    records the exact cap height and contrast used, and would let a model read the
    answer off the generator instead of the pixels.
    """
    return describe_mask(ink_mask(image))


def describe_mask(ink: InkMask) -> dict[str, float]:
    """Feature vector for an already-extracted mask."""
    if ink.empty:
        return _empty_features()

    mask = ink.mask
    features: dict[str, float] = {}
    features["aspect"] = float(np.log(ink.width / ink.height))
    features["ink_density"] = float(mask.mean())
    features["dark_on_light"] = 1.0 if ink.dark_on_light else 0.0

    features.update(_row_profile(mask))
    features.update(_column_profile(mask))
    features.update(_stroke_features(mask))
    features.update(_shape_features(mask))
    features["slant_deg"] = _slant(mask)
    return features


def _empty_features() -> dict[str, float]:
    """Zeros for a crop with no ink, so a blank never breaks the stream."""
    return dict.fromkeys(_feature_names(), 0.0)


def _feature_names() -> list[str]:
    names = ["aspect", "ink_density", "dark_on_light", "slant_deg"]
    names += [f"row_{index}" for index in range(ROW_BINS)]
    names += ["row_peak", "row_centroid", "row_spread"]
    names += [f"col_{index}" for index in range(COLUMN_BINS)]
    names += ["col_mean", "col_std", "col_gaps", "col_autocorr"]
    names += ["stroke_mean", "stroke_std", "stroke_contrast", "stroke_max"]
    names += ["edge_density", "hole_ratio", "grad_orientation"]
    return names


def _resample(values: np.ndarray, bins: int) -> np.ndarray:
    """Average ``values`` into ``bins`` equal buckets, whatever its length."""
    if values.size == 0:
        return np.zeros(bins)
    edges = np.linspace(0, values.size, bins + 1).astype(int)
    return np.array(
        [
            values[start:end].mean() if end > start else 0.0
            for start, end in itertools.pairwise(edges)
        ]
    )


def _row_profile(mask: np.ndarray) -> dict[str, float]:
    """Ink per row, top to bottom: where the baseline and x-height sit."""
    profile = mask.mean(axis=1)
    peak = profile.max()
    normalized = profile / peak if peak > 0 else profile
    features = {
        f"row_{index}": float(value) for index, value in enumerate(_resample(normalized, ROW_BINS))
    }

    positions = np.linspace(0.0, 1.0, profile.size)
    total = profile.sum()
    centroid = float((positions * profile).sum() / total) if total > 0 else 0.5
    spread = (
        float(np.sqrt(((positions - centroid) ** 2 * profile).sum() / total)) if total > 0 else 0.0
    )
    features["row_peak"] = float(peak)
    features["row_centroid"] = centroid
    features["row_spread"] = spread
    return features


def _column_profile(mask: np.ndarray) -> dict[str, float]:
    """Ink per column, left to right: stem rhythm, spacing and regularity."""
    profile = mask.mean(axis=0)
    features = {
        f"col_{index}": float(value) for index, value in enumerate(_resample(profile, COLUMN_BINS))
    }
    features["col_mean"] = float(profile.mean())
    features["col_std"] = float(profile.std())
    # Fraction of columns that are (nearly) empty: inter-letter and inter-word space.
    features["col_gaps"] = float((profile < 0.02).mean())

    # Autocorrelation at the dominant stem spacing. A monospaced font repeats on a
    # fixed pitch and scores high; a proportional one does not.
    centred = profile - profile.mean()
    if profile.size > 8 and centred.any():
        spectrum = np.correlate(centred, centred, mode="full")[profile.size - 1 :]
        spectrum /= spectrum[0]
        window = spectrum[2 : max(3, profile.size // 2)]
        features["col_autocorr"] = float(window.max()) if window.size else 0.0
    else:
        features["col_autocorr"] = 0.0
    return features


def _stroke_widths(mask: np.ndarray) -> np.ndarray:
    """Twice the distance to the nearest background pixel, at every ink pixel.

    That is the local stroke thickness. With scipy present this is an exact
    Euclidean distance transform; without it, a cheap approximation by repeated
    erosion, which is coarser but ordered the same way.
    """
    if not mask.any():
        return np.zeros(0)
    if _HAVE_SCIPY:
        return 2.0 * distance_transform_edt(mask)[mask]

    depth = np.zeros(mask.shape, dtype=np.float64)  # pragma: no cover - fallback path
    eroded = mask.copy()
    for _ in range(min(mask.shape) // 2):
        if not eroded.any():
            break
        depth[eroded] += 1.0
        interior = eroded.copy()
        interior[:-1, :] &= eroded[1:, :]
        interior[1:, :] &= eroded[:-1, :]
        interior[:, :-1] &= eroded[:, 1:]
        interior[:, 1:] &= eroded[:, :-1]
        eroded = interior
    return 2.0 * depth[mask]


def _stroke_features(mask: np.ndarray) -> dict[str, float]:
    widths = _stroke_widths(mask)
    if widths.size == 0:
        return dict.fromkeys(("stroke_mean", "stroke_std", "stroke_contrast", "stroke_max"), 0.0)
    height = mask.shape[0]
    mean = float(widths.mean()) / height
    std = float(widths.std()) / height
    # The thick-to-thin ratio a typographer would call contrast.
    thin, thick = np.percentile(widths, (20, 95))
    return {
        "stroke_mean": mean,
        "stroke_std": std,
        "stroke_contrast": float(thick / thin) if thin > 0 else 1.0,
        "stroke_max": float(widths.max()) / height,
    }


def _shape_features(mask: np.ndarray) -> dict[str, float]:
    """Edge density, enclosed space and stroke orientation."""
    ink = mask.sum()
    if ink == 0:
        return dict.fromkeys(("edge_density", "hole_ratio", "grad_orientation"), 0.0)

    horizontal = np.zeros_like(mask)
    horizontal[:, :-1] = mask[:, :-1] != mask[:, 1:]
    vertical = np.zeros_like(mask)
    vertical[:-1, :] = mask[:-1, :] != mask[1:, :]

    edges = float((horizontal | vertical).sum())
    # High for intricate or serifed shapes, low for fat slabs.
    edge_density = edges / float(ink)
    # Background enclosed by the ink's bounding rows and columns: counters and bowls.
    hole_ratio = float(1.0 - mask.mean())
    total = float(horizontal.sum() + vertical.sum())
    grad_orientation = float(vertical.sum() / total) if total > 0 else 0.5
    return {
        "edge_density": edge_density,
        "hole_ratio": hole_ratio,
        "grad_orientation": grad_orientation,
    }


def _slant(mask: np.ndarray) -> float:
    """The shear angle that makes the column profile spikiest.

    Upright stems line up into tall narrow peaks; slanted ones smear across
    columns. Un-shearing by the right angle restores the peaks, so the angle
    maximising profile variance is the font's slant.
    """
    height, width = mask.shape
    if height < 4 or width < 2 or not mask.any():
        return 0.0

    ink = mask.astype(np.float32)
    offsets = np.arange(height) - height / 2
    columns = np.arange(width)[None, :]
    best_angle, best_score = 0.0, -1.0
    for angle in SLANT_ANGLES:
        shift = np.round(np.tan(np.radians(angle)) * offsets).astype(int)
        # Gather the whole sheared image at once; a per-row roll costs 40x more.
        indices = (columns - shift[:, None]) % width
        score = float(np.take_along_axis(ink, indices, axis=1).mean(axis=0).var())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle
