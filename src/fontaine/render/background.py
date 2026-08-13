"""What the text sits on.

Photo patches are the interesting source: real crops come from designed layouts
over photography, where the background under a single text box already varies in
luminance and texture.

The rest are drawn procedurally from the item's own RNG. They are **not** a
fallback for having no images — ``sources`` is a weighted mixture, so they keep
their share of the stream when ``photo_dir`` is full. Only ``photo`` is conditional,
being the one source that needs files on disk.

Each one exists for a distinct regime the photos do not reliably cover:

- ``noise`` — high-frequency grain, where a stroke and the texture around it live at
  the same spatial scale;
- ``blobs`` — the soft multi-colour wash of a mesh gradient, as design layouts use;
- ``geometric`` — hard colour edges crossing the text box, which is the case
  ``min_contrast`` and the forced scrim exist for;
- ``gradient``, ``vignette`` — a linear and a radial ramp, smooth but not flat;
- ``solid`` — the easy floor, useful as a control.

They were four PNGs in ``assets/backgrounds/`` until it became clear that a
checksum on a procedurally generated image pins the wrong thing: the bytes are
output, the function is the input. As code they cost nothing to distribute and never
repeat, where four files were cropped over and over. Parameters are sampled from the
item RNG with ranges fixed here rather than exposed in the config, following
``_gradient`` and ``_solid`` — a knob per pattern would triple the size of
:class:`BackgroundConfig` for choices no experiment has yet needed to vary.
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
    """Configured sources with positive weight, minus ``photo`` if there are none.

    Source names are validated by :class:`BackgroundConfig`; what cannot be checked
    there is whether ``photo_dir`` actually holds any images. Dropping ``photo``
    renormalizes the rest rather than substituting one particular source for it, so
    an empty photo directory shifts the mixture instead of silently making every
    canvas flat.
    """
    usable = {name: weight for name, weight in config.sources.items() if weight > 0}
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
        case "noise":
            return _noise(rng, size)
        case "blobs":
            return _blobs(rng, size)
        case "geometric":
            return _geometric(rng, size)
        case "gradient":
            return _gradient(rng, size)
        case "vignette":
            return _vignette(rng, size)
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


def _pixel_grid(size: Size) -> tuple[np.ndarray, np.ndarray]:
    """Normalized ``(ys, xs)`` coordinate planes over a canvas, both in ``[0, 1]``.

    Normalized rather than in pixels so a pattern looks the same at every canvas
    size: crops here span roughly an order of magnitude, and a radius in pixels
    would read as a broad wash on the small ones and a hard dot on the large.
    """
    width, height = size
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    return np.broadcast_to(ys, (height, width)), np.broadcast_to(xs, (height, width))


def _to_image(array: np.ndarray) -> Image.Image:
    """Clamp a float ``(h, w, 3)`` array into an 8-bit RGB image."""
    return Image.fromarray(array.round().clip(0, 255).astype(np.uint8), mode="RGB")


def _numpy_rng(rng: random.Random) -> np.random.Generator:
    """A numpy generator seeded from the item RNG.

    Per-pixel noise is impractical through :mod:`random`, but reproducibility is not
    negotiable, so the seed is drawn from the item RNG rather than the OS. Item ``i``
    still gives the same grain on every run.
    """
    return np.random.default_rng(rng.getrandbits(64))


def _noise(rng: random.Random, size: Size) -> Background:
    """Grain over a smooth ramp — texture at the same scale as the strokes.

    Layered over a gradient rather than a flat colour: real grainy surfaces vary at
    low frequency too, and flat-plus-noise leaves the contrast solver an easy job
    since every part of the box has the same mean.
    """
    base = _gradient(rng, size)
    sigma = rng.uniform(3.0, 22.0)
    array = np.asarray(base.image, dtype=np.float32)
    grain = _numpy_rng(rng).normal(0.0, sigma, array.shape).astype(np.float32)
    return Background(
        _to_image(array + grain),
        "noise",
        {"sigma": round(sigma, 2), "base": base.detail},
    )


def _blobs(rng: random.Random, size: Size) -> Background:
    """Soft radial washes of colour, the look of a mesh gradient.

    Centres are allowed outside the canvas, which is what puts a blob's *edge*
    through the text box instead of always its middle — the off-centre falloff is
    the part that makes one side of the box harder than the other.
    """
    ys, xs = _pixel_grid(size)
    array = np.zeros((*ys.shape, 3), dtype=np.float32) + np.array(
        random_color(rng), dtype=np.float32
    )
    count = rng.randint(2, 5)
    for _ in range(count):
        centre_x, centre_y = rng.uniform(-0.3, 1.3), rng.uniform(-0.3, 1.3)
        radius = rng.uniform(0.15, 0.7)
        distance = np.hypot(xs - centre_x, ys - centre_y)
        # Gaussian falloff, alpha-composited so the result stays in range whatever
        # the blobs overlap.
        alpha = np.exp(-0.5 * (distance / radius) ** 2, dtype=np.float32)[..., None]
        array = array * (1.0 - alpha) + np.array(random_color(rng), dtype=np.float32) * alpha
    return Background(_to_image(array), "blobs", {"count": count})


def _geometric(rng: random.Random, size: Size) -> Background:
    """Flat colour rectangles — hard edges running through the text box.

    The one synthetic source with discontinuities in it, which makes it the case
    ``min_contrast`` and the forced scrim were written for: matching the mean
    luminance of a box split by an edge leaves the text invisible on one side.
    """
    width, height = size
    image = Image.new("RGB", size, random_color(rng))
    draw = ImageDraw.Draw(image)
    count = rng.randint(2, 7)
    for _ in range(count):
        # Sized as a fraction of the canvas, and allowed to start off the edge, so
        # rectangles are cut off by the frame the way a real layout's panels are.
        left = rng.uniform(-0.2, 0.9) * width
        top = rng.uniform(-0.2, 0.9) * height
        right = left + rng.uniform(0.15, 0.8) * width
        bottom = top + rng.uniform(0.15, 0.8) * height
        draw.rectangle((left, top, right, bottom), fill=random_color(rng))
    return Background(image, "geometric", {"count": count})


def _vignette(rng: random.Random, size: Size) -> Background:
    """A radial ramp between two colours, darker or lighter towards the edges."""
    inner, outer = random_color(rng), random_color(rng)
    centre_x, centre_y = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    falloff = rng.uniform(0.8, 2.5)
    ys, xs = _pixel_grid(size)
    distance = np.hypot(xs - centre_x, ys - centre_y)
    span = distance.max()
    ramp = (distance / span if span else np.zeros_like(distance)) ** falloff
    ramp = ramp.clip(0.0, 1.0)[..., None]
    blended = (
        np.array(inner, dtype=np.float32) * (1.0 - ramp) + np.array(outer, dtype=np.float32) * ramp
    )
    return Background(
        _to_image(blended),
        "vignette",
        {"inner": list(inner), "outer": list(outer), "falloff": round(falloff, 2)},
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
