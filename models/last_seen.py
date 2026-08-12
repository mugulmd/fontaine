"""Answers with the label of the previous item. A floor, not a model."""

from __future__ import annotations

from PIL import Image

from fontaine.contracts import Recognizer


class LastSeen(Recognizer):
    """Repeats the previous item's label."""

    name = "last-seen"

    def __init__(self) -> None:
        self.last: str | None = None

    def predict(self, image: Image.Image) -> str | None:
        return self.last

    def learn(self, image: Image.Image, label: str) -> None:
        self.last = label
