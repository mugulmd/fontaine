"""Capture artefacts applied to a finished crop.

Everything here is off by default: v1 generates clean renders so the plumbing can
be validated against an easy accuracy ceiling. Raising difficulty later is a
config edit, not a code change.

Note that crop jitter is *not* here — a crop coming from a text detector is
imprecise by nature, so that belongs to the crop itself rather than to an
optional degradation.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from fontaine.config import DegradeConfig


def apply(
    rng: random.Random, image: Image.Image, config: DegradeConfig
) -> tuple[Image.Image, dict[str, Any]]:
    """Degrade ``image``, returning it with a record of what was applied.

    Order matters and mirrors a capture pipeline: geometry, then resolution loss,
    then optics, then sensor noise, then compression.
    """
    applied: dict[str, Any] = {}

    if config.rotate_prob and rng.random() < config.rotate_prob:
        degrees = config.rotate_deg.sample(rng)
        image = image.rotate(
            degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=None
        )
        applied["rotate_deg"] = round(degrees, 2)

    if config.downscale_prob and rng.random() < config.downscale_prob:
        factor = config.downscale_factor.sample(rng)
        small = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        image = image.resize(small, Image.Resampling.BILINEAR)
        applied["downscale_factor"] = round(factor, 3)
        applied["downscaled_size"] = list(small)

    if config.blur_prob and rng.random() < config.blur_prob:
        radius = config.blur_radius.sample(rng)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        applied["blur_radius"] = round(radius, 3)

    if config.noise_prob and rng.random() < config.noise_prob:
        sigma = config.noise_sigma.sample(rng)
        # Seeded from the item's RNG so the noise field is reproducible too.
        generator = np.random.default_rng(rng.getrandbits(64))
        pixels = np.asarray(image, dtype=np.float32)
        noisy = pixels + generator.normal(0.0, sigma, pixels.shape).astype(np.float32)
        image = Image.fromarray(noisy.clip(0, 255).astype(np.uint8), mode=image.mode)
        applied["noise_sigma"] = round(sigma, 2)

    if config.jpeg_prob and rng.random() < config.jpeg_prob:
        quality = int(config.jpeg_quality.sample_int(rng))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            image = reopened.convert("RGB")
        applied["jpeg_quality"] = quality

    return image, applied
