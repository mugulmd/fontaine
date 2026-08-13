"""YAML-backed configuration.

One file describes a whole stream: the font universe it draws labels from, and
how each item is rendered. Relative paths are resolved against the current
working directory, so commands are meant to be run from the repo root.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("configs/stream.yaml")

#: A probability. Out-of-range values are almost always a units mistake — a
#: percentage written where a fraction was meant.
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

#: A relative sampling weight. Zero excludes; negative is meaningless.
Weight = Annotated[float, Field(ge=0.0)]

#: Kinds of text the corpus can emit. Declared here rather than in ``text.corpus``
#: so a mistyped kind is caught when the config loads, not when the first item
#: renders. ``text.corpus.CONTENT_KINDS`` is derived from this.
ContentKind = Literal["word", "phrase", "sentence", "number", "price", "date", "time", "token"]

#: Casing transforms applied to sampled text.
Casing = Literal["as_is", "title", "upper", "lower"]

#: Where a canvas comes from. Everything but ``photo`` is drawn from the item's own
#: RNG, so those need no files and never repeat — a synthetic canvas is a function,
#: not an asset, and pinning a PNG of one would pin the wrong thing.
BackgroundSource = Literal["photo", "noise", "blobs", "geometric", "gradient", "vignette", "solid"]


class ConfigBase(BaseModel):
    """Base for every config model: unknown keys are an error.

    A mistyped option is the commonest config mistake and the hardest to notice,
    since the default it silently leaves in place is usually reasonable.
    """

    model_config = ConfigDict(extra="forbid")


class Range(ConfigBase):
    """An inclusive sampling interval. Written ``[lo, hi]`` in YAML."""

    lo: float
    hi: float

    def __init__(self, lo: float | None = None, hi: float | None = None, **data: Any) -> None:
        """Accept ``Range(18, 64)`` as well as ``Range(lo=18, hi=64)``."""
        if lo is not None:
            data["lo"] = lo
        if hi is not None:
            data["hi"] = hi
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthands(cls, value: Any) -> Any:
        """Accept the YAML ``[lo, hi]`` and bare-number forms, as well as a mapping."""
        if isinstance(value, list | tuple):
            if len(value) != 2:
                raise ValueError(f"a range needs exactly two values, got {list(value)}")
            return {"lo": value[0], "hi": value[1]}
        # A single number is a fixed range: `words: 3` rather than `words: [3, 3]`.
        if isinstance(value, int | float) and not isinstance(value, bool):
            return {"lo": value, "hi": value}
        return value

    @model_validator(mode="after")
    def _check_ordered(self) -> Range:
        if self.hi < self.lo:
            raise ValueError(f"range is inverted: [{self.lo}, {self.hi}]")
        return self

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


def _positive(value: Range) -> Range:
    if value.lo <= 0:
        raise ValueError(f"both ends must be positive: [{value.lo}, {value.hi}]")
    return value


#: A range whose lower bound is above zero, for quantities where zero is nonsense
#: (a size, a scale factor) and for the log-sampled ones where it is undefined.
PositiveRange = Annotated[Range, AfterValidator(_positive)]


class FontsConfig(ConfigBase):
    """Which fonts make up the label space."""

    #: Directory holding the font universe. Scanned recursively.
    font_dir: Path = Path("assets/fonts")
    #: Glyphs a face must have to enter the label space: a preset name from
    #: ``fonts.coverage.CHARSET_PRESETS`` or literal characters. Keep this to a
    #: core set — the corpus intersects its alphabet with each face's real
    #: coverage, so a face need not cover every character the corpus can emit.
    admission_charset: str = "ascii_alnum"
    #: Glob patterns over file names and paths, skipped before parsing.
    exclude: tuple[str, ...] = ()
    #: Keep variable fonts. Off by default: v1 renders static instances only, so a
    #: variable file would enter the label space as one arbitrary default face.
    include_variable: bool = False


class FontRule(ConfigBase):
    """How often one font appears, and over which stretch of the stream.

    Everything is stated rather than emergent, so a stream is a designed
    experiment: the imbalance and the arrival points are the ones you asked for,
    not the ones a particular seed happened to produce.
    """

    #: Relative weight. ``None`` inherits :attr:`ArrivalConfig.default_weight`.
    #: Probability is the weight divided by the total over the fonts active at
    #: that point in the stream. Zero excludes the font entirely.
    weight: Weight | None = None
    #: First item at which the font may appear. Everything before this is drawn
    #: as if the font did not exist, which is what makes a new class arrive
    #: mid-stream at a point you chose.
    start: Annotated[int, Field(ge=0)] = 0
    #: Item at which the font stops appearing, exclusive. ``None`` means never.
    #: A class going away tests graceful degradation as much as one arriving
    #: tests discovery.
    stop: int | None = None

    @model_validator(mode="after")
    def _check_window_is_open(self) -> FontRule:
        if self.stop is not None and self.stop <= self.start:
            raise ValueError(
                f"stop ({self.stop}) must be after start ({self.start}), "
                f"otherwise the font never appears at all"
            )
        return self


class ArrivalConfig(ConfigBase):
    """Which font each arriving item uses.

    Uniform over the whole label space unless told otherwise: every font carries
    ``default_weight`` and is available from the first item. Overrides in
    :attr:`fonts` are what create imbalance and mid-stream change.
    """

    #: Weight for every font without a rule of its own.
    default_weight: Weight = 1.0
    #: Per-font overrides, keyed by face id or by a glob over face ids
    #: (``"roboto:*"`` for a whole family). The most specific matching pattern
    #: wins: an exact face id beats a glob, and a longer glob beats a shorter one.
    #: A pattern matching no font is an error — a renamed font would otherwise
    #: turn a designed experiment back into a uniform one without saying so.
    fonts: dict[str, FontRule] = Field(default_factory=dict)


class CorpusConfig(ConfigBase):
    """What the text says. Deliberately uncorrelated with the font that draws it."""

    #: Relative weights over content kinds.
    kinds: dict[ContentKind, Weight] = Field(
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
    casing: dict[Casing, Weight] = Field(
        default_factory=lambda: {"as_is": 4.0, "title": 3.0, "upper": 2.0, "lower": 1.0}
    )


class TypographyConfig(ConfigBase):
    """How big and how tightly set the text is."""

    #: Height of the capital letters in pixels. Sizing by cap height rather than
    #: em size keeps apparent size comparable across faces, so absolute scale
    #: cannot leak the label.
    #:
    #: Log-sampled, so the median is ``sqrt(lo * hi)`` rather than the midpoint —
    #: a range starting at 8 puts half the crops under 19px, which reads as
    #: pixelated however sharply it was drawn. Log sampling needs a positive
    #: lower bound, so that is enforced here rather than on the first draw.
    cap_height_px: PositiveRange = Range(lo=18.0, hi=64.0)
    #: Extra tracking between characters, as a fraction of cap height. Non-zero
    #: values switch rendering to a per-character path, so v1 leaves it off.
    letter_spacing: Range = Range(lo=0.0, hi=0.0)


class BackgroundConfig(ConfigBase):
    """What the text sits on, and how well it stands out from it."""

    #: Relative weights over background sources — a mixture, not a fallback chain:
    #: with images present the synthetic sources still get their share. ``photo`` is
    #: the exception, dropped and the rest renormalized when ``photo_dir`` holds no
    #: images, since it is the only source that needs files. The defaults keep
    #: photos at 60% of items and the six synthetic patterns at 40% between them.
    sources: dict[BackgroundSource, Weight] = Field(
        default_factory=lambda: {
            "photo": 6.0,
            "noise": 1.0,
            "blobs": 1.0,
            "geometric": 0.75,
            "gradient": 0.75,
            "vignette": 0.25,
            "solid": 0.25,
        }
    )
    #: Directory of PNG images to crop patches from. Scanned recursively.
    photo_dir: Path = Path("assets/backgrounds")
    #: Patch size relative to the canvas before being resampled to fit: above 1
    #: zooms out to finer texture, below 1 zooms in to smoother gradients.
    photo_scale: PositiveRange = Range(lo=0.5, hi=3.0)
    #: Target WCAG contrast ratio between text and the background under it, held
    #: against the harder end of the background's range rather than its mean.
    #: 4.5 is the WCAG floor for body text, 21 is black on white; dropping the
    #: lower bound toward 2 raises difficulty sharply at small cap heights.
    contrast_ratio: Range = Range(lo=4.5, hi=16.0)
    #: Probability the text is darker than its background rather than lighter.
    dark_text_prob: Probability = 0.6
    #: Probability the text colour is a saturated hue rather than near-neutral.
    saturated_prob: Probability = 0.25
    #: Probability of a semi-transparent panel behind the text, as design layouts
    #: use over busy photos. A scrim is also forced, regardless of this
    #: probability, whenever the background varies too much across the text box
    #: for any single colour to stay legible over all of it.
    scrim_prob: Probability = 0.15
    scrim_opacity: Range = Range(lo=0.25, hi=0.8)
    #: Contrast floor anywhere in the text box, below which a scrim is forced.
    #: 1.0 is the identical-colour floor, so anything at or below it is a no-op.
    min_contrast: Annotated[float, Field(ge=1.0, le=21.0)] = 1.9


class CropConfig(ConfigBase):
    """How the crop is placed around the text — the text-detector's imprecision."""

    #: Per-side padding around the tight ink box, as a fraction of ink height,
    #: sampled independently per side. Negative values clip into the glyphs, as
    #: a too-tight detector box does.
    pad: Range = Range(lo=-0.04, hi=0.35)
    #: Probability of a neighbouring text line bleeding in at an edge.
    neighbor_bleed_prob: Probability = 0.0
    #: Gap between the text and a bleeding neighbour, as a fraction of ink height.
    neighbor_gap: Range = Range(lo=0.15, hi=0.7)


class RenderConfig(ConfigBase):
    """Everything that goes into drawing one crop."""

    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    typography: TypographyConfig = Field(default_factory=TypographyConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    crop: CropConfig = Field(default_factory=CropConfig)


class StreamConfig(ConfigBase):
    """Everything needed to reproduce a stream."""

    #: Base seed. Every item derives its own RNG from ``(seed, index)``, so item
    #: i is reproducible without replaying the items before it.
    seed: int = 0
    fonts: FontsConfig = Field(default_factory=FontsConfig)
    arrival: ArrivalConfig = Field(default_factory=ArrivalConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)


def load_stream_config(path: Path | None = DEFAULT_CONFIG_PATH) -> StreamConfig:
    """Load a :class:`StreamConfig`; ``None`` yields the built-in defaults."""
    if path is None:
        return StreamConfig()
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return StreamConfig.model_validate(data)
