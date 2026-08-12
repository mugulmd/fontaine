"""Prequential evaluation: predict, score, then learn.

The only honest way to score an online learner. Every item is a test case before it
is a training example, so nothing is ever scored on data it has already seen, and
there is no split to get wrong.

Every number here is read off a confusion matrix. Accuracy, balanced accuracy, macro
F1 and the per-font breakdown are all functions of the same counts, so there is one
place for a number to be wrong rather than four, and a metric added later is a
function over the matrix rather than another ``update`` call in the loop.

Two matrices, because they answer different questions and neither derives from the
other: one over the whole run, one over the last :data:`WINDOW` items. The lifetime
matrix never forgets, so it carries the cold start forever; the rolling one has
already forgotten it. A model that was lost for its first few thousand items and is
near-perfect now reads as mediocre on the first and excellent on the second, and it
takes both to tell it apart from one that is uniformly mediocre.

One number cannot come from a matrix and is tracked beside them: what always
answering with the commonest label so far would have scored. That depends on the
order the labels arrived in, not just the final counts. It is worth the two lines
because on a deliberately imbalanced stream it can look respectable, and a model
that cannot beat it has learned nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from river import metrics, utils

from fontaine.contracts import Recognizer, Sample

#: Window for the rolling matrix, in items.
WINDOW = 500

#: Stands in for a prediction the model declined to make. It is a column of the
#: matrix and never a row: abstaining is scored as a miss so it has to be counted
#: somewhere, but "no answer" is not a font and must not become a class — it would
#: sit in the macro average with no support and drag every per-font number down.
ABSTAINED = "∅"


@dataclass(frozen=True, slots=True)
class Scores:
    """Everything one confusion matrix has to say.

    Wraps a river ``ConfusionMatrix`` — or a ``Rolling`` around one, which proxies
    attribute access to the metric it wraps and so cannot be resolved statically.
    The loose type is the boundary; nothing past this class touches river.
    """

    matrix: Any

    @property
    def labels(self) -> list[str]:
        """The fonts actually seen, commonest first.

        Read off the rows, so :data:`ABSTAINED` — which only ever appears as a
        column — is not among them. Rows that have fallen out of a rolling window
        keep their key with a count of zero, and are dropped here too.
        """
        rows = {label: total for label, total in self.matrix.sum_row.items() if total > 0}
        return sorted(rows, key=lambda label: -rows[label])

    @property
    def n_items(self) -> int:
        """Items counted in this matrix."""
        return int(self.matrix.total_weight)

    @property
    def accuracy(self) -> float:
        """Share of items predicted correctly, abstentions counting against."""
        return self._correct / self.n_items if self.n_items else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Mean recall over fonts, so a rare font weighs as much as a common one."""
        labels = self.labels
        return sum(self.recall(label) for label in labels) / len(labels) if labels else 0.0

    @property
    def macro_f1(self) -> float:
        """Mean per-font F1: unlike recall, it also punishes over-predicting a font."""
        labels = self.labels
        return sum(self.f1(label) for label in labels) / len(labels) if labels else 0.0

    def support(self, label: str) -> int:
        """Items whose true font was ``label``."""
        return int(self.matrix.sum_row.get(label, 0.0))

    def correct(self, label: str) -> int:
        """Items of ``label`` predicted correctly."""
        return int(self._cell(label, label))

    def recall(self, label: str) -> float:
        """Share of this font's items that were caught."""
        support = self.support(label)
        return self.correct(label) / support if support else 0.0

    def precision(self, label: str) -> float:
        """Share of the guesses of this font that were right."""
        predicted = int(self.matrix.sum_col.get(label, 0.0))
        return self.correct(label) / predicted if predicted else 0.0

    def f1(self, label: str) -> float:
        """Harmonic mean of this font's precision and recall."""
        precision, recall = self.precision(label), self.recall(label)
        total = precision + recall
        return 2 * precision * recall / total if total else 0.0

    def worst_confusions(self, limit: int = 5) -> list[tuple[tuple[str, str], int]]:
        """The commonest (true font, answered) mistakes, abstentions included."""
        mistakes = {
            (true, predicted): int(count)
            for true, row in self.matrix.data.items()
            for predicted, count in row.items()
            if predicted != true and count > 0
        }
        return sorted(mistakes.items(), key=lambda item: -item[1])[:limit]

    @property
    def _correct(self) -> int:
        return sum(self.correct(label) for label in self.labels)

    def _cell(self, true: str, predicted: str) -> float:
        # Through .get rather than matrix[true][predicted]: the matrix is a
        # defaultdict, so indexing a cell that has never been filled creates it.
        return self.matrix.data.get(true, {}).get(predicted, 0.0)


@dataclass(slots=True)
class Result:
    """Everything the run measured."""

    #: Over the whole stream, and over the last :data:`WINDOW` items.
    overall: Scores
    recent: Scores
    #: Correct answers "always say the commonest label so far" would have got.
    majority_correct: int = 0
    #: Rolling accuracy, sampled as the run goes: the shape of the learning curve.
    accuracy_curve: list[tuple[int, float]] = field(default_factory=list)

    @property
    def n_items(self) -> int:
        """Items scored."""
        return self.overall.n_items

    @property
    def majority_accuracy(self) -> float:
        """What always answering with the commonest label so far would have scored."""
        return self.majority_correct / self.n_items if self.n_items else 0.0

    @property
    def chance_accuracy(self) -> float:
        """What guessing uniformly among the fonts seen would score."""
        labels = self.overall.labels
        return 1.0 / len(labels) if labels else 0.0


def run(
    samples: Iterable[Sample],
    model: Recognizer,
    *,
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
    overall = metrics.ConfusionMatrix()
    recent = utils.Rolling(metrics.ConfusionMatrix(), window_size=WINDOW)
    result = Result(overall=Scores(overall), recent=Scores(recent))
    label_counts: Counter[str] = Counter()

    for sample in samples:
        prediction = model.predict(sample.image)
        # Score before learning: the item is a test case first, a lesson second.
        answer = ABSTAINED if prediction is None else prediction
        overall.update(sample.label, answer)
        recent.update(sample.label, answer)

        if label_counts:
            result.majority_correct += label_counts.most_common(1)[0][0] == sample.label
        label_counts[sample.label] += 1

        if result.n_items % curve_every == 0:
            result.accuracy_curve.append((result.n_items, result.recent.accuracy))
        if on_item is not None:
            on_item(sample)

        model.learn(sample.image, sample.label)

    return result
