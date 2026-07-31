"""Online classifiers, built from river.

Every model here shares the property the task demands: it accepts a label it has
never seen before, mid-stream, without being rebuilt. That rules out anything with
a fixed output layer and is the reason river is a good fit — its classifiers grow
their label set as the stream supplies one.

Each is wrapped in river's online ``StandardScaler``, which keeps a running mean
and variance per feature. Without it, stroke width in the hundredths and slant in
the tens would be weighted by nothing but their units.
"""

from __future__ import annotations

from typing import Any

from river import compose, forest, naive_bayes, neighbors, preprocessing, tree

#: Model name → what it is and why it earns a place, shown by ``--help``.
MODELS: dict[str, str] = {
    "gaussian-nb": "Gaussian naive Bayes — one mean and variance per feature per font",
    "hoeffding-tree": "Hoeffding tree — the standard streaming decision tree",
    "adaptive-forest": "Adaptive random forest — a forest with built-in drift detection",
    "knn": "k-nearest neighbours over a sliding window of recent samples",
}

DEFAULT_MODEL = "gaussian-nb"


def build(name: str = DEFAULT_MODEL, **kwargs: Any) -> compose.Pipeline:
    """A scaler chained to the named classifier.

    ``gaussian-nb`` is the honest baseline: it is little more than a mean and a
    variance per feature per class, so whatever it scores is close to the floor
    that any real model has to beat.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")

    match name:
        case "gaussian-nb":
            classifier = naive_bayes.GaussianNB()
        case "hoeffding-tree":
            classifier = tree.HoeffdingTreeClassifier(grace_period=50, **kwargs)
        case "adaptive-forest":
            classifier = forest.ARFClassifier(n_models=10, seed=0, **kwargs)
        case _:
            classifier = neighbors.KNNClassifier(n_neighbors=5, **kwargs)

    return preprocessing.StandardScaler() | classifier
