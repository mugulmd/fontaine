"""Which font each arriving item uses.

Uniform over the label space by default: every font carries the same weight and is
available from the first item. Two kinds of override, both stated in the config
rather than emergent, turn that into an experiment:

* **weights** make some fonts commoner than others, so a recognizer has to learn a
  class from a handful of examples while another class has thousands.
* **schedules** hold a font back until a chosen item, or retire it at one, so a
  new class arrives — or an old one leaves — at a point known in advance.

Both exist to be *dictated*. A process that produced imbalance and arrival times
by itself would leave you measuring whatever a particular seed happened to do,
where what you want is to fix the conditions and measure the algorithm.

The process is sequential — the active set changes as the stream advances — but it
depends only on the item index, so the whole label sequence can be replayed
cheaply without rendering anything.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

from pydantic import BaseModel, Field

from fontaine.config import ArrivalConfig, FontRule
from fontaine.contracts import FontFace


class ScheduleError(ValueError):
    """Raised when a configured schedule cannot produce a stream."""


class FontSchedule(BaseModel):
    """The resolved plan for one font: the experiment as the process will run it."""

    face_id: str
    weight: float
    start: int
    stop: int | None
    #: The config pattern this came from, or ``None`` where the default applied.
    pattern: str | None

    def active_at(self, index: int) -> bool:
        """Whether this font can be drawn for the item at ``index``."""
        return self.weight > 0 and self.start <= index and (self.stop is None or index < self.stop)


@dataclass(frozen=True, slots=True)
class Arrival:
    """One step of the process. In-memory only — never serialized."""

    index: int
    face: FontFace
    #: True when this face has never appeared before — the novelty ground truth.
    #: Note this is the first *actual* appearance, which for a rare font can fall
    #: well after the item its schedule allowed it from.
    first_seen: bool
    #: How many distinct faces have appeared, including this one.
    n_seen: int


class ArrivalStats(BaseModel):
    """Ground truth about a run of the process, for scoring discovery later."""

    n_items: int = 0
    #: Step at which each face first actually appeared.
    face_first_seen: dict[str, int] = Field(default_factory=dict)
    #: Total appearances per face.
    face_counts: dict[str, int] = Field(default_factory=dict)


def resolve_schedule(
    faces: Sequence[FontFace], config: ArrivalConfig | None = None
) -> tuple[FontSchedule, ...]:
    """Apply the config's rules to a label space, giving one plan per font.

    Raises :class:`ScheduleError` for a pattern that matches nothing, since the
    alternative is an experiment that quietly stopped being the one you designed.
    A rule that makes no sense on its own — a negative weight, a window that
    closes before it opens — is rejected earlier, by :class:`FontRule` itself.
    """
    settings = config or ArrivalConfig()
    matched: set[str] = set()
    schedules: list[FontSchedule] = []
    for face in faces:
        pattern = _best_pattern(face.face_id, settings.fonts)
        rule = settings.fonts[pattern] if pattern is not None else FontRule()
        if pattern is not None:
            matched.add(pattern)
        weight = settings.default_weight if rule.weight is None else rule.weight
        schedules.append(
            FontSchedule(
                face_id=face.face_id,
                weight=float(weight),
                start=rule.start,
                stop=rule.stop,
                pattern=pattern,
            )
        )

    unmatched = sorted(set(settings.fonts) - matched)
    if unmatched:
        raise ScheduleError(
            f"arrival rules match no font: {unmatched} — check for a renamed font "
            f"or a mistyped face id (run `fontaine fonts scan` for the real ones)"
        )
    return tuple(schedules)


def _best_pattern(face_id: str, rules: dict[str, FontRule]) -> str | None:
    """The most specific pattern matching ``face_id``: exact first, then longest."""
    candidates = [pattern for pattern in rules if fnmatchcase(face_id, pattern)]
    if not candidates:
        return None
    return max(candidates, key=lambda pattern: (pattern == face_id, len(pattern)))


class ArrivalProcess:
    """Draws faces according to the resolved schedule. Iterating is infinite."""

    def __init__(
        self,
        faces: Sequence[FontFace],
        config: ArrivalConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not faces:
            raise ScheduleError("the arrival process needs at least one face")
        self.config = config or ArrivalConfig()
        self.schedule = resolve_schedule(faces, self.config)
        self.stats = ArrivalStats()

        self._faces = {face.face_id: face for face in faces}
        self._rng = random.Random(f"fontaine:arrival:{seed}")
        self._step = 0
        self._active: list[FontFace] = []
        self._cumulative: list[float] = []
        self._next_change: int | None = None
        # Fail on an unusable schedule now rather than on the first draw.
        self._refresh()

    def __iter__(self) -> Iterator[Arrival]:
        while True:
            yield self.step()

    def take(self, count: int) -> list[Arrival]:
        """Advance the process ``count`` steps and return them."""
        return [self.step() for _ in range(count)]

    @property
    def active(self) -> tuple[FontSchedule, ...]:
        """The schedules in force for the next item."""
        return tuple(plan for plan in self.schedule if plan.active_at(self._step))

    def shares(self) -> dict[str, float]:
        """The probability of each font for the next item, by face id."""
        active = self.active
        total = sum(plan.weight for plan in active)
        return {plan.face_id: plan.weight / total for plan in active} if total else {}

    def step(self) -> Arrival:
        """Draw the next item's face."""
        if self._next_change is not None and self._step >= self._next_change:
            self._refresh()

        face = self._rng.choices(self._active, cum_weights=self._cumulative, k=1)[0]
        first_seen = face.face_id not in self.stats.face_first_seen
        if first_seen:
            self.stats.face_first_seen[face.face_id] = self._step
        self.stats.face_counts[face.face_id] = self.stats.face_counts.get(face.face_id, 0) + 1
        self.stats.n_items = self._step + 1

        arrival = Arrival(
            index=self._step,
            face=face,
            first_seen=first_seen,
            n_seen=len(self.stats.face_first_seen),
        )
        self._step += 1
        return arrival

    def _refresh(self) -> None:
        """Recompute the active set and the next step at which it changes.

        Cumulative weights are built once per change rather than once per item, so
        a draw costs a binary search however many fonts are in play.
        """
        active = [plan for plan in self.schedule if plan.active_at(self._step)]
        if not active:
            raise ScheduleError(
                f"no font is scheduled to appear at item {self._step}: every font is "
                f"either weighted zero or outside its start/stop window"
            )
        self._active = [self._faces[plan.face_id] for plan in active]
        total = 0.0
        self._cumulative = []
        for plan in active:
            total += plan.weight
            self._cumulative.append(total)

        boundaries = {plan.start for plan in self.schedule if plan.start > self._step}
        boundaries |= {
            plan.stop for plan in self.schedule if plan.stop is not None and plan.stop > self._step
        }
        self._next_change = min(boundaries) if boundaries else None


def describe(schedule: Sequence[FontSchedule]) -> str:
    """A one-line summary of how far a schedule departs from uniform."""
    weights = {plan.weight for plan in schedule}
    parts = [
        f"{len(schedule)} fonts",
        "uniform weights" if len(weights) == 1 else f"{len(weights)} distinct weights",
    ]
    timed = sum(1 for plan in schedule if plan.start > 0 or plan.stop is not None)
    if timed:
        parts.append(f"{timed} scheduled")
    excluded = sum(1 for plan in schedule if plan.weight == 0)
    if excluded:
        parts.append(f"{excluded} excluded")
    return ", ".join(parts)
