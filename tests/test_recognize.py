from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw
from river import metrics as river_metrics
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
from fontaine.contracts import Recognizer
from fontaine.evaluate import prequential
from fontaine.fonts import registry as font_registry
from fontaine.recognize import discovery, features
from fontaine.recognize.preprocess import ink_mask
from fontaine.stream.generator import StreamGenerator

ALNUM = string.ascii_letters + string.digits

#: The repo's own model directory, found from this file rather than the cwd.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _baseline(**kwargs) -> Any:
    """The shipped baseline, loaded the way the CLI loads any model.

    Typed loosely on purpose: the model directory is resolved at runtime, so the
    concrete class is not importable here — only the interface it satisfies is,
    and ``test_discovery`` is where that is asserted.
    """
    return discovery.discover(MODEL_DIR)["baseline"](**kwargs)


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


# --------------------------------------------------------------------- baseline


def test_the_model_accepts_a_new_class_mid_stream() -> None:
    """The property the task demands: no fixed label set, no rebuild."""
    # Driven through the river pipeline directly: the claim under test is about
    # the estimator's configuration, not about featurizing a crop.
    pipeline = _baseline().pipeline
    pipeline.learn_one({"a": 1.0, "b": 2.0}, "roboto")

    assert pipeline.predict_one({"a": 1.0, "b": 2.0}) == "roboto"
    # A label never seen before, arriving after the model is already trained.
    pipeline.learn_one({"a": 9.0, "b": 9.0}, "anton")
    assert set(pipeline.predict_proba_one({"a": 1.0, "b": 2.0})) == {"roboto", "anton"}


def test_only_the_recent_window_is_remembered() -> None:
    """Memory is bounded, so it does not grow with the length of the stream."""
    pipeline = _baseline(window_size=20).pipeline
    for index in range(200):
        pipeline.learn_one({"a": float(index)}, "early" if index < 100 else "late")

    # The early class has been pushed out of the window entirely.
    assert pipeline.predict_one({"a": 1.0}) == "late"


def test_the_crop_is_featurized_once_per_item_not_twice(monkeypatch) -> None:
    """Predict then learn on the same object must not pay for the features twice."""
    calls = 0
    original = features.describe

    def counted(image):
        nonlocal calls
        calls += 1
        return original(image)

    monkeypatch.setattr(features, "describe", counted)
    model = _baseline()
    crop = _crop()

    model.predict(crop)
    model.learn(crop, "roboto")
    assert calls == 1
    # A different crop is a different item, and must be measured afresh.
    model.predict(_crop("Other"))
    assert calls == 2


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


def _abstentions(scores: prequential.Scores) -> int:
    """Items the model declined to answer, read out of the sentinel column."""
    return sum(
        int(count)
        for (_, predicted), count in scores.worst_confusions(limit=10_000)
        if predicted == prequential.ABSTAINED
    )


def test_the_first_item_is_scored_before_anything_is_learned(tiny_stream) -> None:
    """Test-then-train: with nothing learned yet, the model can only abstain."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(12))
    result = prequential.run(samples, _baseline())

    assert result.n_items == 12
    # Abstaining is a miss, not a skip: it lands in the matrix and costs accuracy.
    assert _abstentions(result.overall) >= 1
    assert result.overall.accuracy < 1.0


def test_abstaining_never_becomes_a_font(tiny_stream) -> None:
    """The sentinel is a column of the matrix, so it must not join the label space."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(30))

    class Mute(Recognizer):
        """Never answers."""

        name = "mute"

        def predict(self, image):
            return None

        def learn(self, image, label):
            pass

    result = prequential.run(samples, Mute())

    assert prequential.ABSTAINED not in result.overall.labels
    assert result.overall.accuracy == 0.0
    # Three fonts in the stream, so chance is a third — the sentinel is not a fourth.
    assert result.chance_accuracy == pytest.approx(1 / 3)


def test_a_perfect_model_scores_one_and_a_useless_one_scores_zero(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(20))

    class Oracle(Recognizer):
        """Answers with the previous item's label; abstains before it has one."""

        name = "oracle"
        last: str | None = None

        def predict(self, image):
            return self.last

        def learn(self, image, label):
            self.last = label

    class Contrarian(Recognizer):
        """Always wrong, on purpose."""

        name = "contrarian"

        def predict(self, image):
            return "never-a-real-font"

        def learn(self, image, label):
            pass

    oracle = prequential.run(samples, Oracle())
    contrarian = prequential.run(samples, Contrarian())

    # The oracle only ever repeats the previous label, so it cannot be perfect —
    # but it must beat a model that is always wrong, which must score exactly zero.
    assert contrarian.overall.accuracy == 0.0
    assert oracle.overall.accuracy > contrarian.overall.accuracy
    # Answering a font that is not in the stream is a column, never a row.
    assert "never-a-real-font" not in contrarian.overall.labels


def test_the_majority_baseline_is_reported_alongside(tiny_stream) -> None:
    """Accuracy means nothing without the number a model that learned nothing gets."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(40))
    result = prequential.run(samples, _baseline())

    assert 0.0 <= result.majority_accuracy <= 1.0
    assert result.chance_accuracy == pytest.approx(1 / len(result.overall.labels))


def test_per_font_counts_add_up_to_the_totals(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(40))
    result = prequential.run(samples, _baseline())
    scores = result.overall

    assert sum(scores.support(label) for label in scores.labels) == result.n_items
    correct = sum(scores.correct(label) for label in scores.labels)
    assert scores.accuracy == pytest.approx(correct / result.n_items)


# The whole point of reading everything off one matrix is that the numbers stay the
# ones everybody means by those names. These pin that down against river's own.


def test_the_derived_metrics_agree_with_rivers_own() -> None:
    """Accuracy, balanced accuracy and macro F1 are read off the matrix, not tracked."""
    labels = ["a", "b", "c", "d"]
    random.seed(0)
    pairs = [(random.choice(labels), random.choice(labels)) for _ in range(300)]

    matrix = river_metrics.ConfusionMatrix()
    accuracy = river_metrics.Accuracy()
    balanced = river_metrics.BalancedAccuracy()
    macro_f1 = river_metrics.MacroF1()
    for true, predicted in pairs:
        for metric in (matrix, accuracy, balanced, macro_f1):
            metric.update(true, predicted)

    scores = prequential.Scores(matrix)

    assert scores.accuracy == pytest.approx(accuracy.get())
    assert scores.balanced_accuracy == pytest.approx(balanced.get())
    assert scores.macro_f1 == pytest.approx(macro_f1.get())


def test_the_recent_window_forgets_what_the_lifetime_matrix_keeps(tiny_stream) -> None:
    """The two matrices exist to disagree — a learner's whole story is in the gap."""
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(prequential.WINDOW + 200))

    class LateBloomer(Recognizer):
        """Wrong on purpose until the window has moved past the early items."""

        name = "late-bloomer"

        def __init__(self) -> None:
            self.seen = 0
            self.last: str | None = None

        def predict(self, image):
            return self.last if self.seen > 400 else "never-a-real-font"

        def learn(self, image, label):
            self.seen += 1
            self.last = label

    result = prequential.run(samples, LateBloomer())

    assert result.recent.n_items == prequential.WINDOW
    assert result.overall.n_items == prequential.WINDOW + 200
    # The early failures are still in the lifetime matrix and gone from the window.
    assert result.recent.accuracy > result.overall.accuracy


def test_the_curve_samples_the_rolling_accuracy(tiny_stream) -> None:
    settings, registry = tiny_stream
    samples = list(StreamGenerator(settings, registry).take(40))
    result = prequential.run(samples, _baseline(), curve_every=10)

    assert [n_items for n_items, _ in result.accuracy_curve] == [10, 20, 30, 40]
    assert all(0.0 <= value <= 1.0 for _, value in result.accuracy_curve)
