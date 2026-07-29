"""YAML-backed configuration.

Relative paths in a config file are resolved against the current working
directory, so commands are meant to be run from the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from fontaine.contracts import LabelGranularity

T = TypeVar("T")


@dataclass(slots=True)
class FontsConfig:
    """Which fonts make up the label space, and how they are labelled."""

    #: Directory holding the font universe. Scanned recursively.
    font_dir: Path = Path("assets/fonts")
    #: ``face`` treats Roboto-Bold and Roboto-Regular as distinct classes;
    #: ``family`` merges every weight and slant of a family into one.
    label_granularity: LabelGranularity = "face"
    #: Preset name from ``fonts.coverage.CHARSET_PRESETS``, or literal characters.
    #: A face missing any of these glyphs is reported and excluded.
    charset: str = "ascii_printable"
    #: Filename or path glob patterns to skip entirely.
    exclude: tuple[str, ...] = ()
    #: Variable fonts are excluded by default: v1 renders static instances only.
    include_variable: bool = False


def _coerce(value: Any, declared: str) -> Any:
    # ``from __future__ import annotations`` means field types reach us as strings.
    if declared.startswith("Path"):
        return Path(value)
    if declared.startswith("tuple"):
        return tuple(value)
    return value


def _build(cls: type[T], data: dict[str, Any], *, context: str) -> T:
    known = {field.name: str(field.type) for field in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in {context}: {sorted(unknown)}")
    return cls(**{key: _coerce(value, known[key]) for key, value in data.items()})


def load_fonts_config(path: Path | None = None) -> FontsConfig:
    """Load a :class:`FontsConfig`, falling back to defaults when ``path`` is None."""
    if path is None:
        return FontsConfig()
    data = yaml.safe_load(path.read_text()) or {}
    return _build(FontsConfig, data, context=str(path))
