from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Any

import pytest
from synthetic_fonts import build_font

from fontaine.config import (
    BackgroundConfig,
    CorpusConfig,
    CropConfig,
    DegradeConfig,
    Range,
    RenderConfig,
    TypographyConfig,
)
from fontaine.contracts import FontFace
from fontaine.fonts import registry as font_registry
from fontaine.render.metrics import em_size_for_cap_height, load_font
from fontaine.render.textbox import CropRenderer

ALNUM = string.ascii_letters + string.digits


def _config(**overrides: Any) -> RenderConfig:
    """A deterministic, solid-background config: one variable at a time."""
    defaults: dict[str, Any] = {
        "corpus": CorpusConfig(kinds={"word": 1.0}, casing={"as_is": 1.0}),
        "typography": TypographyConfig(cap_height_px=Range(24, 24)),
        "background": BackgroundConfig(sources={"solid": 1.0}, scrim_prob=0.0),
        "crop": CropConfig(pad=Range(0.0, 0.0)),
        "degrade": DegradeConfig(),
    }
    return RenderConfig(**(defaults | overrides))


@pytest.fixture
def face(font_dir: Path) -> FontFace:
    build_font(font_dir / "test.ttf", chars=ALNUM, family="Test", subfamily="Regular")
    return font_registry.scan(font_dir, charset="ascii_alnum").faces[0]


def test_render_returns_an_image_and_metadata(face: FontFace) -> None:
    crop = CropRenderer(_config()).render(face, random.Random(0))

    assert crop.image.mode == "RGB"
    assert crop.image.width > 0 and crop.image.height > 0
    assert crop.metadata["text"]
    assert crop.metadata["background"] == "solid"
    assert crop.metadata["crop_size"] == list(crop.image.size)


def test_same_rng_gives_the_same_crop(face: FontFace) -> None:
    renderer = CropRenderer(_config())
    first = renderer.render(face, random.Random(7))
    second = renderer.render(face, random.Random(7))

    assert first.image.tobytes() == second.image.tobytes()
    assert first.metadata == second.metadata


def test_zero_padding_crops_to_the_ink_box(face: FontFace) -> None:
    crop = CropRenderer(_config()).render(face, random.Random(1))

    left, top, right, bottom = crop.metadata["ink_box"]
    assert crop.image.size == (right - left, bottom - top)


def test_padding_is_independent_per_side(face: FontFace) -> None:
    config = _config(crop=CropConfig(pad=Range(0.0, 0.5)))
    pads = [
        tuple(CropRenderer(config).render(face, random.Random(seed)).metadata["crop_pad"])
        for seed in range(8)
    ]

    # Four independent draws per item, and they differ from each other.
    assert all(len(item) == 4 for item in pads)
    assert any(len(set(item)) == 4 for item in pads)


def test_negative_padding_clips_into_the_glyphs(face: FontFace) -> None:
    tight = CropRenderer(_config(crop=CropConfig(pad=Range(-0.1, -0.1)))).render(
        face, random.Random(2)
    )
    loose = CropRenderer(_config(crop=CropConfig(pad=Range(0.0, 0.0)))).render(
        face, random.Random(2)
    )

    assert tight.image.width < loose.image.width
    assert tight.image.height < loose.image.height


def test_cap_height_normalization_equalizes_apparent_size(font_dir: Path) -> None:
    """The point of sizing by cap height: absolute scale must not encode the label."""
    build_font(
        font_dir / "tall.ttf", chars=ALNUM, family="Tall", subfamily="Regular", glyph_height=900
    )
    build_font(
        font_dir / "short.ttf", chars=ALNUM, family="Short", subfamily="Regular", glyph_height=500
    )
    registry = font_registry.scan(font_dir, charset="ascii_alnum")

    heights = []
    for candidate in registry:
        crop = CropRenderer(_config()).render(candidate, random.Random(3))
        heights.append(crop.image.height)

    # Same target cap height in, near-identical ink height out, despite the faces
    # differing by 1.8x at equal em size.
    assert max(heights) - min(heights) <= 1
    ems = [em_size_for_cap_height(candidate, 24) for candidate in registry]
    assert max(ems) > min(ems) * 1.5


def test_letter_spacing_widens_the_crop(face: FontFace) -> None:
    config = _config(
        corpus=CorpusConfig(kinds={"phrase": 1.0}, words=Range(3, 3), casing={"as_is": 1.0})
    )
    tracked = _config(
        corpus=CorpusConfig(kinds={"phrase": 1.0}, words=Range(3, 3), casing={"as_is": 1.0}),
        typography=TypographyConfig(cap_height_px=Range(24, 24), letter_spacing=Range(0.3, 0.3)),
    )

    plain = CropRenderer(config).render(face, random.Random(6))
    spaced = CropRenderer(tracked).render(face, random.Random(6))

    assert spaced.metadata["text"] == plain.metadata["text"]
    assert spaced.image.width > plain.image.width


def test_degradations_are_off_by_default_and_recorded_when_on(face: FontFace) -> None:
    clean = CropRenderer(_config()).render(face, random.Random(4))
    assert clean.metadata["degradations"] == {}

    noisy_config = _config(
        degrade=DegradeConfig(blur_prob=1.0, jpeg_prob=1.0, jpeg_quality=Range(40, 40))
    )
    noisy = CropRenderer(noisy_config).render(face, random.Random(4))
    assert "blur_radius" in noisy.metadata["degradations"]
    assert noisy.metadata["degradations"]["jpeg_quality"] == 40


def test_contrast_target_is_met_on_a_flat_background(face: FontFace) -> None:
    config = _config(
        background=BackgroundConfig(
            sources={"solid": 1.0}, scrim_prob=0.0, contrast_ratio=Range(5.0, 5.0)
        )
    )
    met = 0
    for seed in range(20):
        metadata = CropRenderer(config).render(face, random.Random(seed)).metadata
        # A flat background may still be too dark or light to reach 5:1 either
        # way; when it is reachable it must be hit.
        if metadata["contrast_achieved"] >= 4.9:
            met += 1
    assert met >= 15


def test_scrim_is_forced_when_the_background_is_too_varied(face: FontFace) -> None:
    """A photo background spanning black to white leaves no legible single colour."""
    config = _config(
        background=BackgroundConfig(
            sources={"gradient": 1.0},
            scrim_prob=0.0,
            contrast_ratio=Range(12.0, 12.0),
            min_contrast=3.0,
        )
    )
    forced = 0
    for seed in range(30):
        metadata = CropRenderer(config).render(face, random.Random(seed)).metadata
        if metadata["scrim"] is not None:
            assert metadata["scrim"]["forced"] is True
            forced += 1
    assert forced > 0


def test_neighbor_bleed_does_not_move_the_ink_box(face: FontFace) -> None:
    """The crop must be positioned on the labelled text, never on its neighbour."""
    plain = CropRenderer(_config()).render(face, random.Random(8))
    bled = CropRenderer(
        _config(crop=CropConfig(pad=Range(0.0, 0.0), neighbor_bleed_prob=1.0))
    ).render(face, random.Random(8))

    assert bled.metadata["neighbor"] is not None
    assert bled.metadata["ink_box"] == plain.metadata["ink_box"]
    assert bled.image.size == plain.image.size


def test_font_loading_is_cached(face: FontFace) -> None:
    assert load_font(face, 32) is load_font(face, 32)
    assert load_font(face, 32) is not load_font(face, 33)
