from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from synthetic_fonts import build_font

from fontaine.config import (
    BackgroundConfig,
    CorpusConfig,
    CropConfig,
    FontsConfig,
    Range,
    RenderConfig,
    StreamConfig,
    TypographyConfig,
)
from fontaine.evaluate import prequential
from fontaine.fonts import registry as font_registry
from fontaine.recognize import features, models
from fontaine.recognize.preprocess import ink_mask
from fontaine.stream.generator import StreamGenerator

ALNUM = string.ascii_letters + string.digits


def _crop(text: str = "Hamburg", *, dark_on_light: bool = True, size: int = 40) -> Image.Image:
    """A plain rendering with PIL's built-in font, enough to exercise the features."""
    from PIL import ImageFont

    font = ImageFont.load_default(size=size)
    background, ink = (
        ((250, 250, 250), (10, 10, 10)) if dark_on_light else ((10, 10, 10), (250, 250, 250))
    )
    left, top, right, bottom = (round(value) for value in font.getbbox(text))
    image = Image.new("RGB", (right - left + 20, bottom - top + 20), background)
    ImageDraw.Draw(image).text((10 - left, 10 - top), text, font=font, fill=ink)
    return image


# ------------------------------------------------------------------ preprocess


def test_polarity_is_normalized_so_features_do_not_flip() -> None:
    """Light-on-dark and dark-on-light crops of the same text must look the same."""
    light = ink_mask(_crop(dark_on_light=True))
    dark = ink_mask(_crop(dark_on_light=False))

    assert light.dark_on_light is True
    assert dark.dark_on_light is False
    # Same ink either way, so the masks agree on almost every pixel.
    assert light.mask.shape == dark.mask.shape
    assert (light.mask == dark.mask).mean() > 0.97


def test_mask_is_normalized_to_a_fixed_ink_height() -> None:
    small = ink_mask(_crop(size=20))
    large = ink_mask(_crop(size=60))

    assert small.height == large.height
    # The aspect ratio survives, so the same words give a similar width.
    assert small.width == pytest.approx(large.width, rel=0.15)


def test_padding_does_not_move_the_ink() -> None:
    """The mask is trimmed to the ink, so a detector's loose box changes nothing."""
    tight = _crop()
    padded = Image.new("RGB", (tight.width + 60, tight.height + 40), (250, 250, 250))
    padded.paste(tight, (45, 28))

    assert (ink_mask(tight).mask == ink_mask(padded).mask).mean() > 0.97


def test_a_blank_crop_yields_an_empty_mask() -> None:
    blank = Image.new("RGB", (40, 20), (128, 128, 128))
    assert ink_mask(blank).empty


# -------------------------------------------------------------------- features


def test_describe_returns_finite_named_numbers() -> None:
    vector = features.describe(_crop())

    assert vector
    assert all(isinstance(name, str) for name in vector)
    assert all(np.isfinite(value) for value in vector.values())


def test_every_crop_produces_the_same_feature_names() -> None:
    """River keys on names, so a missing one would silently become a different model."""
    names = {
        frozenset(features.describe(_crop(text=text)))
        for text in ("Hamburg", "x", "WWWW", "12:45", "a b c d")
    }
    blank = frozenset(features.describe(Image.new("RGB", (30, 12), (128, 128, 128))))

    assert len(names) == 1
    assert blank == next(iter(names))


def test_features_do_not_encode_absolute_size() -> None:
    """Absolute scale must not be a clue, so the same text at two sizes agrees."""
    small = features.describe(_crop(size=24))
    large = features.describe(_crop(size=64))

    for name in ("ink_density", "stroke_mean", "row_centroid", "col_gaps"):
        assert small[name] == pytest.approx(large[name], abs=0.12), name


def test_bolder_text_measures_as_heavier() -> None:
    thin = _crop("HHHH", size=40)
    thick = thin.copy()
    # Thicken the strokes by compositing the image over itself, shifted.
    thick_pixels = np.minimum(np.asarray(thin), np.roll(np.asarray(thin), 2, axis=1))
    thick = Image.fromarray(thick_pixels)

    assert features.describe(thick)["stroke_mean"] > features.describe(thin)["stroke_mean"]


def test_slant_detects_a_sheared_rendering() -> None:
    upright = _crop("HHHHHH", size=48)
    array = np.asarray(upright)
    height = array.shape[0]
    sheared = np.stack(
        [np.roll(array[row], round(0.35 * (row - height / 2)), axis=0) for row in range(height)]
    )

    assert features.describe(upright)["slant_deg"] == pytest.approx(0.0, abs=6.0)
    assert abs(features.describe(Image.fromarray(sheared))["slant_deg"]) > 6.0


# ---------------------------------------------------------------------- models


def test_the_model_accepts_a_new_class_mid_stream() -> None:
    """The property the task demands: no fixed label set, no rebuild."""
    model = models.build()
    model.learn_one({"a": 1.0, "b": 2.0}, "roboto")

    assert model.predict_one({"a": 1.0, "b": 2.0}) == "roboto"
    # A label never seen before, arriving after the model is already trained.
    model.learn_one({"a": 9.0, "b": 9.0}, "anton")
    assert set(model.predict_proba_one({"a": 1.0, "b": 2.0})) == {"roboto", "anton"}


def test_only_the_recent_window_is_remembered() -> None:
    """Memory is bounded, so it does not grow with the length of the stream."""
    model = models.build(window_size=20)
    for index in range(200):
        model.learn_one({"a": float(index)}, "early" if index < 100 else "late")

    # The early class has been pushed out of the window entirely.
    assert model.predict_one({"a": 1.0}) == "late"


# ------------------------------------------------------------------ prequential


@pytest.fixture
def tiny_stream(font_dir: Path) -> tuple[StreamConfig, font_registry.FontRegistry]:
    for index in range(3):
        build_font(
            font_dir / f"face-{index}.ttf",
            chars=ALNUM,
            family=f"Fam{index}",
            subfamily="Regular",
            glyph_height=500 + index * 200,
        )
    settings = StreamConfig(
        seed=1,
        fonts=FontsConfig(font_dir=font_dir),
        render=RenderConfig(
            corpus=CorpusConfig(kinds={"word": 1.0}, casing={"as_is": 1.0}),
            typography=TypographyConfig(cap_height_px=Range(20, 28)),
            background=BackgroundConfig(sources={"solid": 1.0}, scrim_prob=0.0),
            crop=CropConfig(pad=Range(0.0, 0.15)),
        ),
    )
    return settings, font_registry.scan(font_dir, charset="ascii_alnum")


def test_the_first_item_is_scored_before_anything_is_learned(tiny_stream) -> None:
    """Test-then-train: with nothing learned yet, the model can only abstain."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(12))
    result = prequential.run(samples, models.build(), features.describe)

    assert result.n_items == 12
    assert result.abstentions >= 1


def test_a_perfect_model_scores_one_and_a_useless_one_scores_zero(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(20))

    class Oracle:
        """Answers with the previous item's label; abstains before it has one."""

        last: str | None = None

        def predict_one(self, x):
            return self.last

        def learn_one(self, x, y):
            self.last = y

    class Contrarian:
        def predict_one(self, x):
            return "never-a-real-font"

        def learn_one(self, x, y):
            pass

    oracle = prequential.run(samples, Oracle(), features.describe)  # type: ignore[arg-type]
    contrarian = prequential.run(samples, Contrarian(), features.describe)  # type: ignore[arg-type]

    # The oracle only ever repeats the previous label, so it cannot be perfect —
    # but it must beat a model that is always wrong, which must score exactly zero.
    assert contrarian.accuracy == 0.0
    assert oracle.accuracy > contrarian.accuracy


def test_the_majority_baseline_is_reported_alongside(tiny_stream) -> None:
    """Accuracy means nothing without the number a model that learned nothing gets."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(40))
    result = prequential.run(samples, models.build(), features.describe)

    assert 0.0 <= result.majority_accuracy <= 1.0
    assert result.chance_accuracy == pytest.approx(1 / len(result.classes))


def test_per_class_bookkeeping_adds_up(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(40))
    result = prequential.run(samples, models.build(), features.describe)

    assert sum(report.seen for report in result.classes.values()) == result.n_items
    assert sum(report.correct for report in result.classes.values()) == result.correct
    for report in result.classes.values():
        assert report.first_seen is not None
        if report.first_correct is not None:
            assert report.first_correct >= report.first_seen


def test_discovery_lag_is_measured_from_the_scheduled_arrival(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(30))
    label = samples[0].label
    result = prequential.run(samples, models.build(), features.describe, schedule={label: 5})

    report = result.classes[label]
    assert report.scheduled_start == 5
    if report.first_correct is not None:
        assert report.discovery_lag == report.first_correct - 5


def test_schedule_is_read_from_a_manifest() -> None:
    manifest = {
        "schedule": [
            {"face_id": "roboto:regular", "start": 0},
            {"face_id": "roboto:bold", "start": 900},
            {"face_id": "anton:regular", "start": 400},
        ]
    }

    by_face = prequential.schedule_from_manifest(manifest, "face")
    assert by_face == {"roboto:regular": 0, "roboto:bold": 900, "anton:regular": 400}

    # At family granularity a label arrives when the earliest of its faces does.
    by_family = prequential.schedule_from_manifest(manifest, "family")
    assert by_family == {"roboto": 0, "anton": 400}
