"""Prequential evaluation: predict, score, then learn.

The only honest way to score an online learner. Every item is a test case before it
is a training example, so nothing is ever scored on data it has already seen, and
there is no split to get wrong.

Two things are reported alongside accuracy, because accuracy on its own says very
little on this dataset:

* **what a trivial model would score.** Always answering with the commonest label
  so far is a real strategy, and on a deliberately imbalanced stream it can look
  respectable. A model that cannot beat it has learned nothing.
* **discovery lag.** How long after a font was *scheduled* to arrive the model
  first got it right. That is the question the whole stream design exists to ask,
  and average accuracy hides it completely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from river import metrics, utils

from fontaine.contracts import Recognizer, Sample

#: Window for the rolling accuracy, in items.
WINDOW = 500


def _score(metric: Any) -> float:
    """Read a river metric's value.

    River's ``Rolling`` proxies attribute access to the metric it wraps, so its
    ``get`` cannot be resolved statically; the boundary is typed loosely here rather
    than sprinkling suppressions at every call.
    """
    return float(metric.get())


@dataclass(slots=True)
class ClassReport:
    """How one font fared."""

    label: str
    seen: int = 0
    correct: int = 0
    #: Item the label first appeared at, and the item it was first predicted right.
    first_seen: int | None = None
    first_correct: int | None = None
    #: Item the schedule allowed it from, when the stream recorded one.
    scheduled_start: int | None = None

    @property
    def accuracy(self) -> float:
        """Share of this font's items that were predicted correctly."""
        return self.correct / self.seen if self.seen else 0.0

    @property
    def discovery_lag(self) -> int | None:
        """Items between the intended arrival and the first correct prediction."""
        if self.first_correct is None:
            return None
        baseline = self.scheduled_start if self.scheduled_start is not None else self.first_seen
        return None if baseline is None else self.first_correct - baseline


@dataclass(slots=True)
class Result:
    """Everything the run measured."""

    n_items: int = 0
    correct: int = 0
    #: Correct answers a "always say the commonest label so far" strategy would get.
    majority_correct: int = 0
    #: Items where the model had nothing to say, having seen no labels yet.
    abstentions: int = 0
    accuracy_curve: list[tuple[int, float]] = field(default_factory=list)
    classes: dict[str, ClassReport] = field(default_factory=dict)
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)
    rolling_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0

    @property
    def accuracy(self) -> float:
        """Share of all items predicted correctly, abstentions counting against."""
        return self.correct / self.n_items if self.n_items else 0.0

    @property
    def majority_accuracy(self) -> float:
        """What always answering with the commonest label so far would have scored."""
        return self.majority_correct / self.n_items if self.n_items else 0.0

    @property
    def chance_accuracy(self) -> float:
        """What guessing uniformly among the labels seen would score."""
        return 1.0 / len(self.classes) if self.classes else 0.0

    def worst_confusions(self, limit: int = 5) -> list[tuple[tuple[str, str], int]]:
        """The commonest (true label, predicted label) mistakes."""
        return self.confusions.most_common(limit)


def run(
    samples: Iterable[Sample],
    model: Recognizer,
    *,
    schedule: dict[str, int] | None = None,
    curve_every: int = 100,
    on_item: Callable[[Sample], None] | None = None,
) -> Result:
    """Score ``model`` over ``samples``, one item at a time.

    The model is handed ``sample.image`` and nothing else. The metadata records the
    exact cap height, contrast and background used to draw it, so a model given the
    whole sample could read the answer rather than look at the pixels — and
    featurization is the model's business, not the loop's, or every entry would be
    stuck with the same view of the crop.
    """
    result = Result()
    rolling = utils.Rolling(metrics.Accuracy(), window_size=WINDOW)
    balanced = metrics.BalancedAccuracy()
    macro_f1 = metrics.MacroF1()
    label_counts: Counter[str] = Counter()

    for sample in samples:
        prediction = model.predict(sample.image)

        report = result.classes.setdefault(sample.label, ClassReport(label=sample.label))
        if report.first_seen is None:
            report.first_seen = sample.index
            if schedule is not None:
                report.scheduled_start = schedule.get(sample.label)
        report.seen += 1

        # Score before learning: the item is a test case first, a lesson second.
        if prediction is None:
            result.abstentions += 1
        else:
            if prediction == sample.label:
                result.correct += 1
                report.correct += 1
                if report.first_correct is None:
                    report.first_correct = sample.index
            else:
                result.confusions[(sample.label, prediction)] += 1
            rolling.update(sample.label, prediction)
            balanced.update(sample.label, prediction)
            macro_f1.update(sample.label, prediction)

        if label_counts:
            result.majority_correct += label_counts.most_common(1)[0][0] == sample.label
        label_counts[sample.label] += 1
        result.n_items += 1

        if result.n_items % curve_every == 0:
            result.accuracy_curve.append((result.n_items, _score(rolling)))
        if on_item is not None:
            on_item(sample)

        model.learn(sample.image, sample.label)

    result.rolling_accuracy = _score(rolling)
    result.balanced_accuracy = _score(balanced)
    result.macro_f1 = _score(macro_f1)
    return result


def schedule_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    """Map label → the item its schedule allowed it from."""
    return {plan["face_id"]: int(plan["start"]) for plan in manifest.get("schedule", [])}
