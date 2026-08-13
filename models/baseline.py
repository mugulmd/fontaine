"""The reference recognizer: hand-crafted ink features into an online kNN."""

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
