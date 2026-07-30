"""YAML-backed configuration.

One file describes a whole stream: the font universe it draws labels from, and
how each item is rendered. Relative paths are resolved against the current
working directory, so commands are meant to be run from the repo root.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from fontaine.contracts import LabelGranularity

DEFAULT_CONFIG_PATH = Path("configs/stream.yaml")


@dataclass(frozen=True, slots=True)
class Range:
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


@dataclass(slots=True)
class FontsConfig:
    """Which fonts make up the label space, and how they are labelled."""

    #: Directory holding the font universe. Scanned recursively.
    font_dir: Path = Path("assets/fonts")
    #: ``face`` treats Roboto-Bold and Roboto-Regular as distinct classes;
    #: ``family`` merges every weight and slant of a family into one.
    label_granularity: LabelGranularity = "face"
    #: Glyphs a face must have to enter the label space: a preset name from
    #: ``fonts.coverage.CHARSET_PRESETS`` or literal characters. Keep this to a
    #: core set — the corpus intersects its alphabet with each face's real
    #: coverage, so a face need not cover every character the corpus can emit.
    admission_charset: str = "ascii_alnum"
    #: Filename or path globs to skip entirely.
    exclude: tuple[str, ...] = ()
    #: Variable fonts are excluded by default: v1 renders static instances only.
    include_variable: bool = False


@dataclass(slots=True)
class ArrivalConfig:
    """Which font each arriving item uses, and when new fonts first appear.

    The set of fonts is not declared up front: it is discovered as the stream
    advances, which is the whole point of the exercise.
    """

    #: Concentration of the Chinese-restaurant process. Higher means new fonts
    #: keep appearing for longer and popularity is spread more evenly; lower means
    #: the stream settles onto a few fonts quickly.
    concentration: float = 8.0
    #: Half-life in items of a font's popularity, giving concept drift: a font
    #: that stops appearing fades and can later come back. 0 disables forgetting,
    #: leaving a plain rich-get-richer process whose discovery rate decays to zero.
    half_life: int = 2000
    #: Introduce fonts in a seed-shuffled order rather than registry order, so
    #: discovery order is not alphabetical.
    shuffle_pool: bool = True


@dataclass(slots=True)
class CorpusConfig:
    """What the text says. Deliberately uncorrelated with the font that draws it."""

    #: Relative weights over content kinds. See ``text.corpus.CONTENT_KINDS``.
    kinds: dict[str, float] = field(
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
    words: Range = Range(2, 5)
    #: Relative weights over casing transforms.
    casing: dict[str, float] = field(
        default_factory=lambda: {"as_is": 4.0, "title": 3.0, "upper": 2.0, "lower": 1.0}
    )


@dataclass(slots=True)
class TypographyConfig:
    """How big and how tightly set the text is."""

    #: Height of the capital letters in pixels. Sizing by cap height rather than
    #: em size keeps apparent size comparable across faces, so absolute scale
    #: cannot leak the label.
    #:
    #: Log-sampled, so the median is ``sqrt(lo * hi)`` rather than the midpoint —
    #: a range starting at 8 puts half the crops under 19px, which reads as
    #: pixelated however sharply it was drawn.
    cap_height_px: Range = Range(18.0, 64.0)
    #: Extra tracking between characters, as a fraction of cap height. Non-zero
    #: values switch rendering to a per-character path, so v1 leaves it off.
    letter_spacing: Range = Range(0.0, 0.0)


@dataclass(slots=True)
class BackgroundConfig:
    """What the text sits on, and how well it stands out from it."""

    #: Relative weights over background sources. ``photo`` is dropped and the
    #: rest renormalized when ``photo_dir`` holds no images.
    sources: dict[str, float] = field(
        default_factory=lambda: {"photo": 6.0, "gradient": 2.0, "solid": 2.0}
    )
    #: Directory of PNG images to crop patches from. Scanned recursively.
    photo_dir: Path = Path("assets/backgrounds")
    #: Patch size relative to the canvas before being resampled to fit: above 1
    #: zooms out to finer texture, below 1 zooms in to smoother gradients.
    photo_scale: Range = Range(0.5, 3.0)
    #: Target WCAG contrast ratio between text and the background under it, held
    #: against the harder end of the background's range rather than its mean.
    #: 4.5 is the WCAG floor for body text, 21 is black on white; dropping the
    #: lower bound toward 2 raises difficulty sharply at small cap heights.
    contrast_ratio: Range = Range(4.5, 16.0)
    #: Probability the text is darker than its background rather than lighter.
    dark_text_prob: float = 0.6
    #: Probability the text colour is a saturated hue rather than near-neutral.
    saturated_prob: float = 0.25
    #: Probability of a semi-transparent panel behind the text, as design layouts
    #: use over busy photos. A scrim is also forced, regardless of this
    #: probability, whenever the background varies too much across the text box
    #: for any single colour to stay legible over all of it.
    scrim_prob: float = 0.15
    scrim_opacity: Range = Range(0.25, 0.8)
    #: Contrast floor anywhere in the text box, below which a scrim is forced.
    min_contrast: float = 1.9


@dataclass(slots=True)
class CropConfig:
    """How the crop is placed around the text — the text-detector's imprecision."""

    #: Per-side padding around the tight ink box, as a fraction of ink height,
    #: sampled independently per side. Negative values clip into the glyphs, as
    #: a too-tight detector box does.
    pad: Range = Range(-0.04, 0.35)
    #: Probability of a neighbouring text line bleeding in at an edge.
    neighbor_bleed_prob: float = 0.0
    #: Gap between the text and a bleeding neighbour, as a fraction of ink height.
    neighbor_gap: Range = Range(0.15, 0.7)


@dataclass(slots=True)
class DegradeConfig:
    """Capture artefacts. All off in v1 — turn them on to raise difficulty."""

    blur_prob: float = 0.0
    blur_radius: Range = Range(0.3, 1.2)
    downscale_prob: float = 0.0
    downscale_factor: Range = Range(0.4, 0.9)
    jpeg_prob: float = 0.0
    jpeg_quality: Range = Range(30, 85)
    noise_prob: float = 0.0
    noise_sigma: Range = Range(2.0, 12.0)
    rotate_prob: float = 0.0
    rotate_deg: Range = Range(-2.5, 2.5)


@dataclass(slots=True)
class RenderConfig:
    """Everything that goes into drawing one crop."""

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    typography: TypographyConfig = field(default_factory=TypographyConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    degrade: DegradeConfig = field(default_factory=DegradeConfig)


@dataclass(slots=True)
class StreamConfig:
    """Everything needed to reproduce a stream."""

    #: Base seed. Every item derives its own RNG from ``(seed, index)``, so item
    #: i is reproducible without replaying the items before it.
    seed: int = 0
    fonts: FontsConfig = field(default_factory=FontsConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


def _coerce(value: Any, declared: Any, *, context: str) -> Any:
    if declared is Path:
        return Path(value)
    if declared is Range:
        if isinstance(value, Range):
            return value
        if isinstance(value, (int, float)):
            return Range(float(value), float(value))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return Range(float(value[0]), float(value[1]))
        raise ValueError(f"{context}: expected [lo, hi] or a single number, got {value!r}")
    if is_dataclass(declared):
        if not isinstance(value, dict):
            raise ValueError(f"{context}: expected a mapping, got {value!r}")
        return _build(declared, value, context=context)
    if get_origin(declared) is tuple:
        return tuple(value)
    if get_origin(declared) is dict:
        _, value_type = get_args(declared)
        return {
            key: _coerce(item, value_type, context=f"{context}.{key}")
            for key, item in value.items()
        }
    return value


def _build[T](cls: type[T], data: dict[str, Any], *, context: str) -> T:
    hints = get_type_hints(cls)
    known = {field.name for field in fields(cls)}  # ty: ignore[invalid-argument-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {context}: {sorted(unknown)}")
    return cls(
        **{
            key: _coerce(value, hints[key], context=f"{context}.{key}")
            for key, value in data.items()
        }
    )


def to_dict(value: Any) -> Any:
    """Recursively convert a config object to JSON-serializable data.

    Used to snapshot the exact configuration into a stream's manifest, so a
    generated stream stays interpretable even after the config file moves on.
    """
    if isinstance(value, Range):
        # Same shape it has in YAML, so a snapshot can be loaded back as a config.
        return [value.lo, value.hi]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value


def from_dict(data: dict[str, Any], *, context: str = "config") -> StreamConfig:
    """Rebuild a :class:`StreamConfig` from :func:`to_dict` output.

    Lets a stream's manifest snapshot be loaded back as a config, so the exact
    settings a stream was generated with can be recovered from the stream itself.
    """
    return _build(StreamConfig, data, context=context)


def load_stream_config(path: Path | None = DEFAULT_CONFIG_PATH) -> StreamConfig:
    """Load a :class:`StreamConfig`; ``None`` yields the built-in defaults."""
    if path is None:
        return StreamConfig()
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return from_dict(data, context=path.name)
