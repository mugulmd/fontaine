"""The stream itself: an endless iterator of labelled crops.

This is the contract the recognizer consumes. Nothing here touches the disk — the
same iterator can be materialized by ``io.writer`` or fed straight to an online
learner, which is what keeps benchmarking against a frozen stream and training
against a live one the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterator

from fontaine.config import StreamConfig
from fontaine.contracts import Sample
from fontaine.fonts.registry import FontRegistry
from fontaine.render.textbox import CropRenderer, RenderError
from fontaine.rng import item_rng
from fontaine.stream.arrival import ArrivalProcess, ArrivalStats


@dataclass(slots=True)
class SkippedItem:
    """An item that could not be rendered. Recorded rather than silently dropped."""

    index: int
    face_id: str
    error: str


class StreamGenerator:
    """Composes the arrival process with the renderer.

    Iteration is infinite. ``index`` is the arrival step, so a skipped item leaves
    a gap in the index sequence rather than shifting every later item — the index
    has to keep identifying the same draw from the arrival process.
    """

    def __init__(self, config: StreamConfig, registry: FontRegistry) -> None:
        if not registry.faces:
            raise ValueError("cannot generate a stream from an empty label space")
        self.config = config
        self.registry = registry
        self.renderer = CropRenderer(config.render)
        self.arrivals = ArrivalProcess(
            registry.faces,
            config.arrival,
            seed=config.seed,
            label_granularity=registry.label_granularity,
        )
        self.skipped: list[SkippedItem] = []

    @property
    def stats(self) -> ArrivalStats:
        """Discovery ground truth so far: first-seen steps and appearance counts."""
        return self.arrivals.stats

    def __iter__(self) -> Iterator[Sample]:
        for arrival in self.arrivals:
            face = arrival.face
            try:
                crop = self.renderer.render(face, item_rng(self.config.seed, arrival.index))
            except RenderError as error:
                self.skipped.append(SkippedItem(arrival.index, face.face_id, str(error)))
                continue
            yield Sample(
                index=arrival.index,
                image=crop.image,
                label=face.label(self.registry.label_granularity),
                metadata={
                    "face_id": face.face_id,
                    "family_id": face.family_id,
                    "weight": face.weight,
                    "italic": face.italic,
                    "monospace": face.monospace,
                    "first_seen": arrival.first_seen,
                    "label_first_seen": arrival.label_first_seen,
                    "n_seen": arrival.n_seen,
                    **crop.metadata,
                },
            )

    def take(self, count: int) -> Iterator[Sample]:
        """The first ``count`` items that rendered successfully."""
        return islice(iter(self), count)
