"""YAML-backed configuration.

One file describes a whole stream: the font universe it draws labels from, and
how each item is rendered. Relative paths are resolved against the current
working directory, so commands are meant to be run from the repo root.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("configs/stream.yaml")


class Range(BaseModel):
    """An inclusive sampling interval. Written ``[lo, hi]`` in YAML."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"range is inverted: [{self.lo}, {self.hi}]")

    @property
    def fixed(self) -> bool:
        """Whether the interval is a single value, making sampling a no-op."""
        return self.lo == self.hi

    def sample(self, rng: random.Random) -> float:
        """A value drawn uniformly from the interval."""
        return self.lo if self.fixed else rng.uniform(self.lo, self.hi)

    def sample_int(self, rng: random.Random) -> int:
        """A whole number drawn uniformly from the interval, both ends included."""
        return int(self.lo) if self.fixed else rng.randint(int(self.lo), int(self.hi))

    def sample_log(self, rng: random.Random) -> float:
        """Sample uniformly in log space — the right default for scale parameters.

        Uniform sampling of a size range spends most of its draws on large sizes;
        log-uniform gives small text, the hard regime, a fair share.
        """
        if self.fixed:
            return self.lo
        if self.lo <= 0:
            raise ValueError(f"log sampling needs a positive lower bound: {self}")
        return math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))


class FontsConfig(BaseModel):
    """Which fonts make up the label space."""

    #: Directory holding the font universe. Scanned recursively.
    font_dir: Path = Path("assets/fonts")
    #: Glyphs a face must have to enter the label space: a preset name from
    #: ``fonts.coverage.CHARSET_PRESETS`` or literal characters. Keep this to a
    #: core set — the corpus intersects its alphabet with each face's real
    #: coverage, so a face need not cover every character the corpus can emit.
    admission_charset: str = "ascii_alnum"


class FontRule(BaseModel):
    """How often one font appears, and over which stretch of the stream.

    Everything is stated rather than emergent, so a stream is a designed
    experiment: the imbalance and the arrival points are the ones you asked for,
    not the ones a particular seed happened to produce.
    """

    #: Relative weight. ``None`` inherits :attr:`ArrivalConfig.default_weight`.
    #: Probability is the weight divided by the total over the fonts active at
    #: that point in the stream. Zero excludes the font entirely.
    weight: float | None = None
    #: First item at which the font may appear. Everything before this is drawn
    #: as if the font did not exist, which is what makes a new class arrive
    #: mid-stream at a point you chose.
    start: int = 0
    #: Item at which the font stops appearing, exclusive. ``None`` means never.
    #: A class going away tests graceful degradation as much as one arriving
    #: tests discovery.
    stop: int | None = None


class ArrivalConfig(BaseModel):
    """Which font each arriving item uses.

    Uniform over the whole label space unless told otherwise: every font carries
    ``default_weight`` and is available from the first item. Overrides in
    :attr:`fonts` are what create imbalance and mid-stream change.
    """

    #: Weight for every font without a rule of its own.
    default_weight: float = 1.0
    #: Per-font overrides, keyed by face id or by a glob over face ids
    #: (``"roboto:*"`` for a whole family). The most specific matching pattern
    #: wins: an exact face id beats a glob, and a longer glob beats a shorter one.
    #: A pattern matching no font is an error — a renamed font would otherwise
    #: turn a designed experiment back into a uniform one without saying so.
    fonts: dict[str, FontRule] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    """What the text says. Deliberately uncorrelated with the font that draws it."""

    #: Relative weights over content kinds. See ``text.corpus.CONTENT_KINDS``.
    kinds: dict[str, float] = Field(
        default_factory=lambda: {
            "word": 3.0,
            "phrase": 4.0,
            "sentence": 2.0,
            "number": 1.0,
            "price": 1.0,
            "date": 0.5,
            "time": 0.5,
            "token": 1.0,
        }
    )
    #: Word count for multi-word kinds.
    words: Range = Range(lo=2, hi=5)
    #: Relative weights over casing transforms.
    casing: dict[str, float] = Field(
        default_factory=lambda: {"as_is": 4.0, "title": 3.0, "upper": 2.0, "lower": 1.0}
    )


class TypographyConfig(BaseModel):
    """How big and how tightly set the text is."""

    #: Height of the capital letters in pixels. Sizing by cap height rather than
    #: em size keeps apparent size comparable across faces, so absolute scale
    #: cannot leak the label.
    #:
    #: Log-sampled, so the median is ``sqrt(lo * hi)`` rather than the midpoint —
    #: a range starting at 8 puts half the crops under 19px, which reads as
    #: pixelated however sharply it was drawn.
    cap_height_px: Range = Range(lo=18.0, hi=64.0)
    #: Extra tracking between characters, as a fraction of cap height. Non-zero
    #: values switch rendering to a per-character path, so v1 leaves it off.
    letter_spacing: Range = Range(lo=0.0, hi=0.0)


class BackgroundConfig(BaseModel):
    """What the text sits on, and how well it stands out from it."""

    #: Relative weights over background sources. ``photo`` is dropped and the
    #: rest renormalized when ``photo_dir`` holds no images.
    sources: dict[str, float] = Field(
        default_factory=lambda: {"photo": 6.0, "gradient": 2.0, "solid": 2.0}
    )
    #: Directory of PNG images to crop patches from. Scanned recursively.
    photo_dir: Path = Path("assets/backgrounds")
    #: Patch size relative to the canvas before being resampled to fit: above 1
    #: zooms out to finer texture, below 1 zooms in to smoother gradients.
    photo_scale: Range = Range(lo=0.5, hi=3.0)
    #: Target WCAG contrast ratio between text and the background under it, held
    #: against the harder end of the background's range rather than its mean.
    #: 4.5 is the WCAG floor for body text, 21 is black on white; dropping the
    #: lower bound toward 2 raises difficulty sharply at small cap heights.
    contrast_ratio: Range = Range(lo=4.5, hi=16.0)
    #: Probability the text is darker than its background rather than lighter.
    dark_text_prob: float = 0.6
    #: Probability the text colour is a saturated hue rather than near-neutral.
    saturated_prob: float = 0.25
    #: Probability of a semi-transparent panel behind the text, as design layouts
    #: use over busy photos. A scrim is also forced, regardless of this
    #: probability, whenever the background varies too much across the text box
    #: for any single colour to stay legible over all of it.
    scrim_prob: float = 0.15
    scrim_opacity: Range = Range(lo=0.25, hi=0.8)
    #: Contrast floor anywhere in the text box, below which a scrim is forced.
    min_contrast: float = 1.9


class CropConfig(BaseModel):
    """How the crop is placed around the text — the text-detector's imprecision."""

    #: Per-side padding around the tight ink box, as a fraction of ink height,
    #: sampled independently per side. Negative values clip into the glyphs, as
    #: a too-tight detector box does.
    pad: Range = Range(lo=-0.04, hi=0.35)
    #: Probability of a neighbouring text line bleeding in at an edge.
    neighbor_bleed_prob: float = 0.0
    #: Gap between the text and a bleeding neighbour, as a fraction of ink height.
    neighbor_gap: Range = Range(lo=0.15, hi=0.7)


class RenderConfig(BaseModel):
    """Everything that goes into drawing one crop."""

    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    typography: TypographyConfig = Field(default_factory=TypographyConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    crop: CropConfig = Field(default_factory=CropConfig)


class StreamConfig(BaseModel):
    """Everything needed to reproduce a stream."""

    #: Base seed. Every item derives its own RNG from ``(seed, index)``, so item
    #: i is reproducible without replaying the items before it.
    seed: int = 0
    fonts: FontsConfig = Field(default_factory=FontsConfig)
    arrival: ArrivalConfig = Field(default_factory=ArrivalConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)


# FIXME: this will likely not work because:
# - nested dicts
# - current config file is wrongly formatted
# - we need to handle range properly
def load_stream_config(path: Path | None = DEFAULT_CONFIG_PATH) -> StreamConfig:
    """Load a :class:`StreamConfig`; ``None`` yields the built-in defaults."""
    if path is None:
        return StreamConfig()
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return StreamConfig.model_validate(data)
