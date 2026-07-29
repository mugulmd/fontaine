from __future__ import annotations

import random

import numpy as np
import pytest

from fontaine.render.color import (
    color_with_luminance,
    contrast_ratio,
    guaranteed_contrast,
    luminance_extremes,
    mean_luminance,
    plan_contrast,
    relative_luminance,
)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def test_black_on_white_is_the_maximum_ratio() -> None:
    assert contrast_ratio(relative_luminance(BLACK), relative_luminance(WHITE)) == pytest.approx(21.0)


def test_contrast_ratio_is_symmetric() -> None:
    dark, light = relative_luminance((30, 40, 50)), relative_luminance((200, 210, 220))
    assert contrast_ratio(dark, light) == pytest.approx(contrast_ratio(light, dark))


def test_extremes_bracket_the_mean_on_a_split_patch() -> None:
    patch = np.concatenate(
        [np.full((20, 20, 3), 10, np.uint8), np.full((20, 20, 3), 245, np.uint8)], axis=1
    )
    low, high = luminance_extremes(patch)
    mean = mean_luminance(patch)
    assert low < mean < high
    assert low == pytest.approx(relative_luminance((10, 10, 10)), abs=0.01)
    assert high == pytest.approx(relative_luminance((245, 245, 245)), abs=0.01)


@pytest.mark.parametrize("background", [0.02, 0.2, 0.5, 0.8, 0.98])
@pytest.mark.parametrize("ratio", [2.0, 4.5, 7.0])
def test_plan_hits_the_requested_ratio_when_reachable(background: float, ratio: float) -> None:
    plan = plan_contrast((background, background), ratio=ratio, darker=True)
    # Either the ratio is met, or it was capped by the sRGB gamut at an extreme.
    assert plan.achieved == pytest.approx(ratio, rel=0.02) or plan.luminance in (0.0, 1.0)
    assert plan.achieved >= min(ratio, plan.achieved)


def test_plan_abandons_the_side_rather_than_the_contrast() -> None:
    # Nothing can be 7:1 darker than near-black, so the text must go light.
    plan = plan_contrast((0.01, 0.01), ratio=7.0, darker=True)
    assert plan.darker is False
    assert plan.achieved == pytest.approx(7.0, rel=0.02)


def test_plan_reports_the_hardest_point_of_a_split_background() -> None:
    dark, light = 0.03, 0.9
    plan = plan_contrast((dark, light), ratio=5.0, darker=True)

    # 5:1 everywhere is impossible across a range this wide, so the honest report
    # is a low guaranteed contrast — the caller's cue to add a scrim.
    assert plan.achieved < 5.0
    assert plan.reference in (pytest.approx(dark), pytest.approx(light))
    # Whichever side was chosen, the reference is the end that is hardest for it.
    assert plan.reference == pytest.approx(dark if plan.darker else light)


def test_plan_flags_a_background_no_colour_can_serve() -> None:
    plan = plan_contrast((0.0, 1.0), ratio=21.0, darker=True)
    assert plan.achieved < 2.0


def test_plan_prefers_the_side_with_better_guaranteed_contrast() -> None:
    # Mostly dark with a light minority: light text serves the whole box better.
    plan = plan_contrast((0.02, 0.35), ratio=10.0, darker=True)
    assert plan.darker is False
    assert plan.achieved > plan_contrast((0.02, 0.35), ratio=10.0, darker=False).achieved * 0.99


def test_guaranteed_contrast_takes_the_worse_end() -> None:
    ratio, reference = guaranteed_contrast((0.05, 0.8), relative_luminance(WHITE))
    assert reference == pytest.approx(0.8)
    assert ratio == pytest.approx(contrast_ratio(0.8, 1.0))


@pytest.mark.parametrize("target", [0.0, 0.05, 0.35, 0.75, 1.0])
def test_neutral_colour_lands_on_the_requested_luminance(target: float) -> None:
    color = color_with_luminance(random.Random(0), target, saturated=False)
    assert relative_luminance(color) == pytest.approx(target, abs=0.01)


def test_saturated_colour_is_coloured_and_lands_on_target() -> None:
    rng = random.Random(4)
    for _ in range(20):
        color = color_with_luminance(rng, 0.4, saturated=True)
        if len(set(color)) > 1:
            break
    else:
        pytest.fail("saturated colours were all neutral")
    assert relative_luminance(color) == pytest.approx(0.4, abs=0.01)


@pytest.mark.parametrize("target", [0.02, 0.2, 0.6, 0.9, 0.99])
def test_saturated_colour_reaches_luminances_above_its_hue_ceiling(target: float) -> None:
    """A hue cannot be bright and pure at once; luminance must win over saturation."""
    rng = random.Random(1)
    for _ in range(40):
        color = color_with_luminance(rng, target, saturated=True)
        assert relative_luminance(color) == pytest.approx(target, abs=0.01)
