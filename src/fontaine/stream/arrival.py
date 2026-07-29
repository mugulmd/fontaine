"""Which font each arriving item uses.

The font set is not declared up front. At each step the process either introduces
a font never seen before, or reuses one it has already seen with probability
proportional to that font's recent popularity — a Chinese-restaurant process. Two
properties fall out of that for free:

* **progressive discovery.** New fonts keep appearing as the stream advances,
  which is what the recognizer has to cope with.
* **a long tail.** Popularity is rich-get-richer, so a few fonts dominate and many
  stay rare — much harder, and much more realistic, than a uniform draw.

``half_life`` adds the third property, **drift**: popularity is counted with
exponential forgetting, so a font that stops appearing fades out and can come
back later. It also keeps discovery going indefinitely — without forgetting, the
probability of a new font decays like 1/t and the stream ossifies.

Unlike the render, this process is inherently sequential: what appears next
depends on what came before. It is driven by its own generator seeded from the
base seed, so the label sequence can be replayed cheaply — without rendering
anything — while per-item render parameters stay independently reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from fontaine.config import ArrivalConfig
from fontaine.contracts import FontFace, LabelGranularity


@dataclass(frozen=True, slots=True)
class Arrival:
    """One step of the process."""

    index: int
    face: FontFace
    #: True when this face has never appeared before — the novelty ground truth.
    first_seen: bool
    #: True when this face's *label* has never appeared before. Differs from
    #: ``first_seen`` at family granularity, where a new face of a known family
    #: is not a new class.
    label_first_seen: bool
    #: How many distinct faces have appeared, including this one.
    n_seen: int


@dataclass(slots=True)
class ArrivalStats:
    """Ground truth about a run of the process, for scoring discovery later."""

    n_items: int = 0
    #: Step at which each face and each label first appeared.
    face_first_seen: dict[str, int] = field(default_factory=dict)
    label_first_seen: dict[str, int] = field(default_factory=dict)
    #: Total appearances per face and per label.
    face_counts: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "face_first_seen": self.face_first_seen,
            "label_first_seen": self.label_first_seen,
            "face_counts": self.face_counts,
            "label_counts": self.label_counts,
        }


class ArrivalProcess:
    """Draws faces from a pool, discovering them progressively.

    Iterating is infinite; the pool bounds how many *distinct* faces can appear,
    not how many items.
    """

    def __init__(
        self,
        faces: Sequence[FontFace],
        config: ArrivalConfig | None = None,
        *,
        seed: int = 0,
        label_granularity: LabelGranularity = "face",
    ) -> None:
        if not faces:
            raise ValueError("the arrival process needs at least one face")
        self.config = config or ArrivalConfig()
        if self.config.concentration <= 0:
            raise ValueError(f"concentration must be positive: {self.config.concentration}")
        if self.config.half_life < 0:
            raise ValueError(f"half_life cannot be negative: {self.config.half_life}")

        self.label_granularity = label_granularity
        self._rng = random.Random(f"fontaine:arrival:{seed}")
        self._pool = list(faces)
        if self.config.shuffle_pool:
            # Discovery order should not be alphabetical, but it must be reproducible.
            self._rng.shuffle(self._pool)

        self._decay = (
            0.5 ** (1.0 / self.config.half_life) if self.config.half_life > 0 else 1.0
        )
        self._active: list[FontFace] = []
        self._weights: list[float] = []
        self._step = 0
        self.stats = ArrivalStats()

    def __iter__(self) -> Iterator[Arrival]:
        while True:
            yield self.step()

    def take(self, count: int) -> list[Arrival]:
        return [self.step() for _ in range(count)]

    @property
    def exhausted(self) -> bool:
        """True once every face in the pool has appeared at least once."""
        return len(self._active) == len(self._pool)

    def new_font_probability(self) -> float:
        """Chance the next item introduces a face never seen before."""
        if self.exhausted:
            return 0.0
        total = math.fsum(self._weights)
        return self.config.concentration / (total + self.config.concentration)

    def step(self) -> Arrival:
        if self.exhausted or self._rng.random() >= self.new_font_probability():
            face = self._reuse()
            first_seen = False
        else:
            face = self._introduce()
            first_seen = True

        label = face.label(self.label_granularity)
        label_first_seen = label not in self.stats.label_first_seen
        if first_seen:
            self.stats.face_first_seen[face.face_id] = self._step
        if label_first_seen:
            self.stats.label_first_seen[label] = self._step
        self.stats.face_counts[face.face_id] = self.stats.face_counts.get(face.face_id, 0) + 1
        self.stats.label_counts[label] = self.stats.label_counts.get(label, 0) + 1
        self.stats.n_items = self._step + 1

        arrival = Arrival(
            index=self._step,
            face=face,
            first_seen=first_seen,
            label_first_seen=label_first_seen,
            n_seen=len(self._active),
        )
        self._step += 1
        return arrival

    def _introduce(self) -> FontFace:
        face = self._pool[len(self._active)]
        self._forget()
        self._active.append(face)
        self._weights.append(1.0)
        return face

    def _reuse(self) -> FontFace:
        position = self._rng.choices(range(len(self._active)), weights=self._weights, k=1)[0]
        self._forget()
        self._weights[position] += 1.0
        return self._active[position]

    def _forget(self) -> None:
        """Decay every popularity weight one step, before the new arrival counts.

        With ``half_life`` set, total weight saturates near ``half_life / ln 2``
        instead of growing with the stream, which is what keeps the rate of new
        fonts from decaying to nothing.
        """
        if self._decay == 1.0:
            return
        self._weights = [weight * self._decay for weight in self._weights]

    def popularity(self) -> dict[str, float]:
        """Current recency-weighted share per face — the drifting distribution."""
        total = math.fsum(self._weights)
        if total <= 0:
            return {}
        return {
            face.face_id: weight / total
            for face, weight in zip(self._active, self._weights, strict=True)
        }
