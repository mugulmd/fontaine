"""What the text sits on.

Photo patches are the interesting source: real crops come from designed layouts
over photography, where the background under a single text box already varies in
luminance and texture. Solid and gradient canvases are kept as a fallback so the
pipeline works before any images have been gathered.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fontaine.config import BackgroundConfig
from fontaine.render.color import RGB, random_color

IMAGE_EXTENSIONS = frozenset({".png"})

Size = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Background:
    """A rendered canvas, and how it was made."""

    image: Image.Image
    source: str
    detail: dict[str, object]


@lru_cache(maxsize=8)
def _image_paths(photo_dir: str) -> tuple[Path, ...]:
    directory = Path(photo_dir)
    if not directory.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


@lru_cache(maxsize=32)
def _load_image(path: str) -> Image.Image:
    """Load and cache a base image. Kept in RGB so composites are predictable."""
    with Image.open(path) as handle:
        return handle.convert("RGB")


def available_sources(config: BackgroundConfig) -> dict[str, float]:
    """Configured sources with positive weight, minus ``photo`` if there are none."""
    usable = {name: weight for name, weight in config.sources.items() if weight > 0}
    unknown = set(usable) - {"photo", "gradient", "solid"}
    if unknown:
        raise ValueError(f"unknown background sources: {sorted(unknown)}")
    if "photo" in usable and not _image_paths(str(config.photo_dir)):
        usable.pop("photo")
    if not usable:
        raise ValueError("no usable background source: check sources and photo_dir")
    return usable


def count_images(config: BackgroundConfig) -> int:
    """How many usable images sit in the configured photo directory."""
    return len(_image_paths(str(config.photo_dir)))


def build(rng: random.Random, size: Size, config: BackgroundConfig) -> Background:
    """Make a canvas of ``size`` from one of the configured sources."""
    usable = available_sources(config)
    names = sorted(usable)
    source = rng.choices(names, weights=[usable[name] for name in names], k=1)[0]
    match source:
        case "photo":
            return _photo(rng, size, config)
        case "gradient":
            return _gradient(rng, size)
        case _:
            return _solid(rng, size)


def _solid(rng: random.Random, size: Size) -> Background:
    color = random_color(rng)
    return Background(Image.new("RGB", size, color), "solid", {"color": list(color)})


def _gradient(rng: random.Random, size: Size) -> Background:
    start, end = random_color(rng), random_color(rng)
    angle = rng.uniform(0, 2 * math.pi)
    width, height = size
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    # Project onto the gradient direction, then rescale to [0, 1].
    projection = xs * math.cos(angle) + ys * math.sin(angle)
    span = projection.max() - projection.min()
    ramp = (projection - projection.min()) / span if span else np.zeros_like(projection)
    ramp = np.broadcast_to(ramp, (height, width))[..., None]
    blended = (
        np.array(start, dtype=np.float32) * (1.0 - ramp) + np.array(end, dtype=np.float32) * ramp
    )
    image = Image.fromarray(blended.round().clip(0, 255).astype(np.uint8), mode="RGB")
    return Background(
        image,
        "gradient",
        {"start": list(start), "end": list(end), "angle_deg": round(math.degrees(angle), 1)},
    )


def _photo(rng: random.Random, size: Size, config: BackgroundConfig) -> Background:
    paths = _image_paths(str(config.photo_dir))
    path = paths[rng.randrange(len(paths))]
    source = _load_image(str(path))
    width, height = size
    scale = config.photo_scale.sample(rng)

    # The patch is taken at `scale` times the canvas size then resampled down to
    # it: above 1 shrinks the texture, below 1 magnifies it.
    patch_width = max(1, min(source.width, round(width * scale)))
    patch_height = max(1, min(source.height, round(height * scale)))
    left = rng.randint(0, source.width - patch_width)
    top = rng.randint(0, source.height - patch_height)
    patch = source.crop((left, top, left + patch_width, top + patch_height))
    if patch.size != size:
        patch = patch.resize(size, Image.Resampling.BICUBIC)
    return Background(
        patch,
        "photo",
        {
            "file": path.name,
            "scale": round(scale, 3),
            "patch": [left, top, patch_width, patch_height],
        },
    )


def add_scrim(
    image: Image.Image, box: tuple[int, int, int, int], color: RGB, opacity: float
) -> None:
    """Composite a translucent panel over ``box``, in place.

    Design layouts use these to keep text legible over busy photography, and they
    change the local statistics enough that the text colour must be chosen after.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(*color, round(255 * opacity)))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
