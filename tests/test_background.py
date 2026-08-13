"""The background sources, especially the synthetic ones that replaced files."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fontaine.config import BackgroundConfig, BackgroundSource, Range
from fontaine.render import background as background_module
from fontaine.render.background import build

#: Every source that needs no files on disk. Named explicitly rather than derived
#: from the config Literal, so adding a source without a test is a failure here.
SYNTHETIC: tuple[BackgroundSource, ...] = (
    "noise",
    "blobs",
    "geometric",
    "gradient",
    "vignette",
    "solid",
)

#: Two extremes of the range crops actually span: a small crop of short text, and a
#: wide one of a sentence at a large cap height.
SIZES = ((32, 14), (900, 220))


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    """A photo directory holding one usable image."""
    directory = tmp_path / "backgrounds"
    directory.mkdir()
    Image.new("RGB", (400, 300), (90, 120, 160)).save(directory / "flat.png")
    # The loader and the path scan are both cached on the directory string, and
    # tmp_path differs per test, so no eviction is needed between tests.
    return directory


class TestSyntheticSources:
    @pytest.mark.parametrize("source", SYNTHETIC)
    @pytest.mark.parametrize("size", SIZES)
    def test_it_fills_the_requested_canvas(
        self, source: BackgroundSource, size: tuple[int, int]
    ) -> None:
        result = build(random.Random(0), size, BackgroundConfig(sources={source: 1.0}))

        assert result.source == source
        assert result.image.size == size
        assert result.image.mode == "RGB"

    @pytest.mark.parametrize("source", SYNTHETIC)
    def test_it_needs_no_files(self, source: BackgroundSource, tmp_path: Path) -> None:
        # The whole point of moving the patterns into code: an empty asset
        # directory is not a degraded mode for anything but `photo`.
        config = BackgroundConfig(sources={source: 1.0}, photo_dir=tmp_path / "nothing")
        assert build(random.Random(1), (64, 32), config).source == source

    @pytest.mark.parametrize("source", SYNTHETIC)
    def test_the_same_rng_gives_the_same_canvas(self, source: BackgroundSource) -> None:
        # Includes `noise`, whose grain comes from a numpy generator seeded off the
        # item RNG — reproducibility is what the whole stream rests on.
        config = BackgroundConfig(sources={source: 1.0})
        first = build(random.Random(7), (120, 60), config)
        second = build(random.Random(7), (120, 60), config)

        assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
        assert first.detail == second.detail

    @pytest.mark.parametrize("source", SYNTHETIC)
    def test_a_different_rng_gives_a_different_canvas(self, source: BackgroundSource) -> None:
        config = BackgroundConfig(sources={source: 1.0})
        canvases = {
            build(random.Random(seed), (120, 60), config).image.tobytes() for seed in range(6)
        }
        # A function rather than a fixed file: no two items share a canvas, which is
        # what four cropped PNGs could not offer.
        assert len(canvases) == 6

    @pytest.mark.parametrize("source", SYNTHETIC)
    def test_the_detail_is_json_serializable(self, source: BackgroundSource) -> None:
        # It lands in annotations.jsonl through Sample.metadata.
        result = build(random.Random(3), (80, 40), BackgroundConfig(sources={source: 1.0}))
        assert json.loads(json.dumps(result.detail)) == result.detail


class TestPatternCharacter:
    """Each pattern earns its place by covering a regime the others do not."""

    def test_noise_varies_at_the_pixel_scale(self) -> None:
        noisy = build(random.Random(0), (200, 200), BackgroundConfig(sources={"noise": 1.0}))
        smooth = build(random.Random(0), (200, 200), BackgroundConfig(sources={"gradient": 1.0}))

        # Mean absolute difference between neighbouring pixels: grain shows up here
        # and a ramp does not, which is the property `noise` exists to add.
        def roughness(image: Image.Image) -> float:
            array = np.asarray(image, dtype=np.float32)
            return float(np.abs(np.diff(array, axis=1)).mean())

        assert roughness(noisy.image) > 5 * roughness(smooth.image)

    def test_geometric_puts_hard_edges_in_the_canvas(self) -> None:
        # The case min_contrast and the forced scrim exist for: a box split by an
        # edge has no single legible text colour.
        result = build(random.Random(4), (200, 200), BackgroundConfig(sources={"geometric": 1.0}))
        array = np.asarray(result.image, dtype=np.float32)

        assert np.abs(np.diff(array, axis=1)).max() > 30

    def test_blobs_and_vignette_stay_smooth(self) -> None:
        for source in ("blobs", "vignette"):
            result = build(random.Random(2), (200, 200), BackgroundConfig(sources={source: 1.0}))
            array = np.asarray(result.image, dtype=np.float32)
            # Smooth means no pixel-to-pixel jumps, unlike geometric's edges.
            assert np.abs(np.diff(array, axis=1)).max() < 12, source

    def test_solid_is_actually_flat(self) -> None:
        result = build(random.Random(5), (60, 40), BackgroundConfig(sources={"solid": 1.0}))
        array = np.asarray(result.image)
        assert array.min() == array.max() or len(np.unique(array.reshape(-1, 3), axis=0)) == 1


class TestSourceMixture:
    def test_photos_and_patterns_both_appear_when_images_are_present(self, photo_dir: Path) -> None:
        # `sources` is a mixture, not a fallback chain — having photos does not
        # switch the synthetic sources off.
        config = BackgroundConfig(
            sources={"photo": 1.0, "blobs": 1.0}, photo_dir=photo_dir, photo_scale=Range(1.0, 1.0)
        )
        seen = {build(random.Random(seed), (64, 32), config).source for seed in range(40)}

        assert seen == {"photo", "blobs"}

    def test_photo_drops_out_and_the_rest_renormalize(self, tmp_path: Path) -> None:
        config = BackgroundConfig(
            sources={"photo": 6.0, "blobs": 1.0}, photo_dir=tmp_path / "empty"
        )
        assert background_module.available_sources(config) == {"blobs": 1.0}

        seen = {build(random.Random(seed), (64, 32), config).source for seed in range(10)}
        assert seen == {"blobs"}

    def test_a_zero_weight_source_never_appears(self) -> None:
        config = BackgroundConfig(sources={"solid": 1.0, "noise": 0.0})
        seen = {build(random.Random(seed), (64, 32), config).source for seed in range(20)}
        assert seen == {"solid"}

    def test_no_usable_source_is_an_error(self, tmp_path: Path) -> None:
        config = BackgroundConfig(sources={"photo": 1.0}, photo_dir=tmp_path / "empty")
        with pytest.raises(ValueError, match="no usable background source"):
            build(random.Random(0), (64, 32), config)

    def test_the_default_mixture_keeps_photos_at_sixty_percent(self) -> None:
        # Stated so a re-weighting is a deliberate edit rather than a drift: the
        # patterns replaced files inside the photo pool, and the split they had
        # before that change is the one preserved here.
        weights = BackgroundConfig().sources
        total = sum(weights.values())
        assert weights["photo"] / total == pytest.approx(0.6)
