"""The reference recognizer: hand-crafted ink features into an online kNN.

This file is not special. It sits in ``models/`` and implements
:class:`~fontaine.contracts.Recognizer` exactly like yours will, and the
framework finds it the same way. Copy it as a starting point.

k-nearest-neighbours over a bounded window of recent samples. It is here for two
reasons. It has the property the task demands — a label never seen before can arrive
mid-stream and simply be learned, with no output layer to resize and no retraining.
And on this data it beat the alternatives comfortably: 56% against 8.3% chance on a
twelve-font pool, where Gaussian naive Bayes, a Hoeffding tree and an adaptive forest
all sat near 42%.

Nothing is fitted at all. The model keeps the last ``window_size`` feature vectors
and answers by majority vote among the nearest few, which makes it a fair floor for
anything more sophisticated to be measured against.

Two details that matter more than they look:

* **the window size.** It is what the accuracy rests on. Measured over the same
  2,000 items, a window of 1000 scores 53% where 300 scores 48% and 50 scores 31%:
  a handful of remembered examples cannot cover a dozen fonts.
* **the scaler.** River's online ``StandardScaler`` keeps a running mean and variance
  per feature. The distance metric needs it: stroke width lives in the hundredths
  and slant in the tens, and unscaled the vote would be decided by units alone.
"""

from __future__ import annotations

from PIL import Image
from river import compose, neighbors, preprocessing

from fontaine.contracts import Recognizer
from fontaine.recognize import features

#: Neighbours consulted per prediction.
N_NEIGHBORS = 5
#: Feature vectors remembered. Bounded, so memory does not grow with the stream.
WINDOW_SIZE = 1000


class KnnBaseline(Recognizer):
    """Ink statistics into an online k-nearest-neighbours vote."""

    name = "baseline"

    def __init__(self, n_neighbors: int = N_NEIGHBORS, window_size: int = WINDOW_SIZE) -> None:
        self.pipeline = self._build(n_neighbors, window_size)
        # The loop calls predict(image) then learn(image, label) with the same
        # object, so the feature vector is computed once per item rather than
        # twice. Keyed on identity, not content: two crops can look alike.
        self._cached_for: Image.Image | None = None
        self._cached: dict[str, float] = {}

    @staticmethod
    def _build(n_neighbors: int, window_size: int) -> compose.Pipeline:
        """An online scaler chained to a k-nearest-neighbours classifier.

        The engine is exact search over the window, rather than river's default
        approximate neighbour graph. The graph scores about a point and a half higher,
        which is inside the noise, and costs a great deal of explaining: it needs a seed
        to be reproducible at all, carries warm-up and pruning parameters, and fails
        outright on a window too small to build a graph from. For a demonstration
        baseline, "remembers the last N vectors and compares against all of them" is
        worth more than the point and a half.
        """
        return preprocessing.StandardScaler() | neighbors.KNNClassifier(
            n_neighbors=n_neighbors,
            engine=neighbors.LazySearch(window_size=window_size),
        )

    def _features(self, image: Image.Image) -> dict[str, float]:
        if image is not self._cached_for:
            self._cached_for, self._cached = image, features.describe(image)
        return self._cached

    def predict(self, image: Image.Image) -> str | None:
        """Majority vote among the nearest remembered vectors."""
        return self.pipeline.predict_one(self._features(image))

    def learn(self, image: Image.Image, label: str) -> None:
        """Push the vector into the window, evicting the oldest."""
        self.pipeline.learn_one(self._features(image), label)
