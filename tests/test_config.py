"""The config is the experiment's input, so a bad one must fail at load.

Every check here exists because the failure it catches is otherwise silent or
reported far from its cause: a mistyped key keeps a reasonable default, and a
nonsensical range only blows up on the item that happens to sample it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fontaine.config import (
    DEFAULT_CONFIG_PATH,
    CorpusConfig,
    Range,
    StreamConfig,
    load_stream_config,
)


def test_the_shipped_config_loads() -> None:
    """configs/stream.yaml is the documented starting point; it must parse."""
    settings = load_stream_config(DEFAULT_CONFIG_PATH)

    assert settings.render.typography.cap_height_px == Range(18.0, 64.0)
    assert settings.fonts.admission_charset == "ascii_alnum"


def test_no_config_gives_the_built_in_defaults() -> None:
    assert load_stream_config(None) == StreamConfig()


def test_a_missing_config_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_stream_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------- ranges


def test_a_range_is_written_as_a_pair_in_yaml() -> None:
    settings = StreamConfig.model_validate({"render": {"crop": {"pad": [-0.1, 0.4]}}})

    assert settings.render.crop.pad == Range(-0.1, 0.4)


def test_a_bare_number_is_a_fixed_range() -> None:
    assert CorpusConfig.model_validate({"words": 3}).words == Range(3, 3)


@pytest.mark.parametrize("value", [[1.0], [1.0, 2.0, 3.0]])
def test_a_range_needs_exactly_two_values(value: list[float]) -> None:
    with pytest.raises(ValidationError, match="exactly two values"):
        Range.model_validate(value)


def test_an_inverted_range_is_rejected() -> None:
    """Previously silent: the dataclass __post_init__ check was dead under pydantic."""
    with pytest.raises(ValidationError, match="range is inverted"):
        Range.model_validate([64, 18])


def test_a_log_sampled_range_needs_a_positive_lower_bound() -> None:
    """cap_height_px is log-sampled, so a lower bound of 0 has no logarithm."""
    with pytest.raises(ValidationError, match="both ends must be positive"):
        StreamConfig.model_validate({"render": {"typography": {"cap_height_px": [0, 64]}}})


# ----------------------------------------------------------------- bad values


def test_a_mistyped_key_is_rejected() -> None:
    """The commonest config mistake, and invisible when the default is sensible."""
    with pytest.raises(ValidationError, match="font_dirr"):
        StreamConfig.model_validate({"fonts": {"font_dirr": "assets/fonts"}})


def test_a_probability_outside_zero_to_one_is_rejected() -> None:
    """Catches a percentage written where a fraction was meant."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        StreamConfig.model_validate({"render": {"background": {"scrim_prob": 15}}})


def test_a_negative_weight_is_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        CorpusConfig(kinds={"word": -1.0})


def test_an_unknown_background_source_is_rejected() -> None:
    with pytest.raises(ValidationError, match="photoo"):
        StreamConfig.model_validate({"render": {"background": {"sources": {"photoo": 1.0}}}})


# ---------------------------------------------------------------- round trip


def test_a_config_survives_a_round_trip_through_the_manifest_form() -> None:
    """The manifest snapshot has to load back as the config that produced it.

    Dumped in JSON mode, the same way ``store.writer`` writes it, so this also
    pins that every field stays representable in a manifest.
    """
    settings = load_stream_config(DEFAULT_CONFIG_PATH)

    restored = StreamConfig.model_validate(settings.model_dump(mode="json"))

    assert restored == settings
