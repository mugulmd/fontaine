from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import pytest
from synthetic_fonts import build_font

from fontaine.config import ArrivalConfig, FontRule
from fontaine.contracts import FontFace
from fontaine.fonts import registry as font_registry
from fontaine.stream.arrival import ArrivalProcess, ScheduleError, resolve_schedule

ALNUM = string.ascii_letters + string.digits


def _pool(font_dir: Path, count: int = 4, *, families: int | None = None) -> list[FontFace]:
    """A pool of ``count`` faces spread over ``families`` families."""
    families = count if families is None else families
    for index in range(count):
        build_font(
            font_dir / f"face-{index}.ttf",
            chars=ALNUM,
            family=f"Fam{index % families}",
            subfamily=f"Style{index}",
        )
    return font_registry.scan(font_dir, charset="ascii_alnum").faces


def _process(faces: list[FontFace], **overrides: Any) -> ArrivalProcess:
    return ArrivalProcess(faces, ArrivalConfig(**overrides), seed=0)


def _counts(process: ArrivalProcess, n: int) -> dict[str, int]:
    process.take(n)
    return process.stats.face_counts


# ------------------------------------------------------------------- weights


def test_default_is_uniform_over_the_whole_pool(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    counts = _counts(_process(faces), 8_000)

    assert set(counts) == {face.face_id for face in faces}
    for count in counts.values():
        assert count == pytest.approx(2_000, rel=0.12)


def test_weight_scales_frequency_proportionally(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    heavy = faces[0].face_id
    counts = _counts(_process(faces, fonts={heavy: FontRule(weight=6.0)}), 9_000)

    # Weights 6 + 1 + 1 + 1 = 9, so the heavy font should take two thirds.
    assert counts[heavy] / 9_000 == pytest.approx(6 / 9, rel=0.05)
    for face in faces[1:]:
        assert counts[face.face_id] / 9_000 == pytest.approx(1 / 9, rel=0.15)


def test_default_weight_is_only_a_baseline(font_dir: Path) -> None:
    """Changing the default alone cannot change the shape — only the ratios can."""
    faces = _pool(font_dir, 4)
    counts = _counts(_process(faces, default_weight=7.5), 4_000)

    for count in counts.values():
        assert count == pytest.approx(1_000, rel=0.15)


def test_zero_weight_excludes_a_font_entirely(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    silent = faces[0].face_id
    counts = _counts(_process(faces, fonts={silent: FontRule(weight=0)}), 2_000)

    assert silent not in counts
    assert len(counts) == 3


def test_glob_applies_to_a_whole_family(font_dir: Path) -> None:
    faces = _pool(font_dir, 6, families=2)
    counts = _counts(_process(faces, fonts={"fam0:*": FontRule(weight=0)}), 3_000)

    assert all(not face_id.startswith("fam0:") for face_id in counts)
    assert any(face_id.startswith("fam1:") for face_id in counts)


def test_an_exact_face_id_beats_a_glob(font_dir: Path) -> None:
    faces = _pool(font_dir, 4, families=1)
    exact = faces[0].face_id
    schedule = {
        plan.face_id: plan
        for plan in resolve_schedule(
            faces,
            ArrivalConfig(fonts={"fam0:*": FontRule(weight=3.0), exact: FontRule(weight=9.0)}),
        )
    }

    assert schedule[exact].weight == 9.0
    assert schedule[exact].pattern == exact
    for face in faces[1:]:
        assert schedule[face.face_id].weight == 3.0
        assert schedule[face.face_id].pattern == "fam0:*"


def test_the_longer_glob_wins(font_dir: Path) -> None:
    faces = _pool(font_dir, 4, families=1)
    schedule = {
        plan.face_id: plan
        for plan in resolve_schedule(
            faces,
            ArrivalConfig(
                fonts={"fam0:*": FontRule(weight=2.0), "fam0:style0*": FontRule(weight=8.0)}
            ),
        )
    }

    assert schedule["fam0:style0"].weight == 8.0
    assert schedule["fam0:style1"].weight == 2.0


# ------------------------------------------------------------------ schedules


def test_start_holds_a_font_back_until_its_item(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    late = faces[0].face_id
    process = _process(faces, fonts={late: FontRule(start=500)})
    arrivals = process.take(2_000)

    appearances = [item.index for item in arrivals if item.face.face_id == late]
    assert appearances, "the font never appeared at all"
    assert min(appearances) >= 500
    # And it is a genuine novelty at that point, not something seen earlier.
    assert process.stats.face_first_seen[late] >= 500


def test_stop_retires_a_font_at_its_item(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    early = faces[0].face_id
    arrivals = _process(faces, fonts={early: FontRule(stop=800)}).take(2_000)

    appearances = [item.index for item in arrivals if item.face.face_id == early]
    assert appearances
    assert max(appearances) < 800


def test_a_window_bounds_a_font_on_both_sides(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    windowed = faces[0].face_id
    arrivals = _process(faces, fonts={windowed: FontRule(start=400, stop=900)}).take(2_000)

    appearances = [item.index for item in arrivals if item.face.face_id == windowed]
    assert appearances
    assert min(appearances) >= 400
    assert max(appearances) < 900


def test_the_schedule_records_what_was_asked_for(font_dir: Path) -> None:
    """The manifest needs the intended arrival, not only the observed one."""
    faces = _pool(font_dir, 4)
    late = faces[0].face_id
    process = _process(faces, fonts={late: FontRule(weight=0.05, start=1_000)})
    process.take(4_000)

    plan = next(item for item in process.schedule if item.face_id == late)
    assert plan.start == 1_000
    assert plan.weight == 0.05
    # A rare font's first actual appearance falls after the item it was allowed
    # from, which is exactly why both numbers are kept.
    assert process.stats.face_first_seen[late] > plan.start


def test_shares_reflect_the_active_set_at_each_point(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    late = faces[0].face_id
    process = _process(faces, fonts={late: FontRule(start=100)})

    assert late not in process.shares()
    assert sum(process.shares().values()) == pytest.approx(1.0)
    process.take(100)
    assert late in process.shares()
    assert process.shares()[late] == pytest.approx(0.25)


def test_weights_are_renormalized_when_the_active_set_changes(font_dir: Path) -> None:
    faces = _pool(font_dir, 4)
    process = _process(faces, fonts={faces[0].face_id: FontRule(stop=1_000)})
    arrivals = process.take(2_000)

    # Three fonts share everything after the fourth retires, so each takes a third.
    after = [item for item in arrivals if item.index >= 1_000]
    for face in faces[1:]:
        share = sum(1 for item in after if item.face.face_id == face.face_id) / len(after)
        assert share == pytest.approx(1 / 3, rel=0.15)


# -------------------------------------------------------------- bookkeeping


def test_same_seed_gives_the_same_sequence(font_dir: Path) -> None:
    faces = _pool(font_dir)
    first = [item.face.face_id for item in _process(faces).take(300)]
    second = [item.face.face_id for item in _process(faces).take(300)]

    assert first == second


def test_different_seeds_diverge(font_dir: Path) -> None:
    faces = _pool(font_dir)
    first = [item.face.face_id for item in ArrivalProcess(faces, seed=0).take(300)]
    second = [item.face.face_id for item in ArrivalProcess(faces, seed=1).take(300)]

    assert first != second


def test_counts_add_up_to_the_number_of_items(font_dir: Path) -> None:
    process = _process(_pool(font_dir))
    process.take(250)

    assert sum(process.stats.face_counts.values()) == 250
    assert process.stats.n_items == 250


def test_first_seen_flags_agree_with_the_recorded_ground_truth(font_dir: Path) -> None:
    process = _process(_pool(font_dir))
    items = process.take(300)

    for item in items:
        if item.first_seen:
            assert process.stats.face_first_seen[item.face.face_id] == item.index
    for face_id, step in process.stats.face_first_seen.items():
        assert min(item.index for item in items if item.face.face_id == face_id) == step


def test_faces_of_the_same_family_are_distinct_labels(font_dir: Path) -> None:
    """Labels are per face: two weights of one family are two classes to discover."""
    faces = _pool(font_dir, 6, families=2)
    process = ArrivalProcess(faces, seed=0)
    process.take(400)

    assert len(process.stats.face_counts) == 6


# ----------------------------------------------------------------- rejection


def test_a_pattern_matching_nothing_is_rejected(font_dir: Path) -> None:
    """Otherwise a renamed font silently turns a designed stream back into uniform."""
    faces = _pool(font_dir, 4)
    with pytest.raises(ScheduleError, match="match no font"):
        _process(faces, fonts={"no-such-font:regular": FontRule(weight=5)})


def test_an_empty_pool_is_rejected() -> None:
    with pytest.raises(ScheduleError, match="at least one face"):
        ArrivalProcess([])


def test_a_stream_with_nothing_available_at_the_start_is_rejected(font_dir: Path) -> None:
    faces = _pool(font_dir, 2)
    with pytest.raises(ScheduleError, match="item 0"):
        _process(faces, fonts={face.face_id: FontRule(start=100) for face in faces})


def test_a_schedule_that_empties_mid_stream_is_rejected(font_dir: Path) -> None:
    faces = _pool(font_dir, 2)
    process = _process(faces, fonts={face.face_id: FontRule(stop=50) for face in faces})

    with pytest.raises(ScheduleError, match="item 50"):
        process.take(100)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (FontRule(weight=-1), "weight cannot be negative"),
        (FontRule(start=-5), "start cannot be negative"),
        (FontRule(start=100, stop=100), "must be after start"),
        (FontRule(start=100, stop=50), "must be after start"),
    ],
)
def test_nonsensical_rules_are_rejected(font_dir: Path, rule: FontRule, message: str) -> None:
    faces = _pool(font_dir, 2)
    with pytest.raises(ScheduleError, match=message):
        _process(faces, fonts={faces[0].face_id: rule})


def test_a_negative_default_weight_is_rejected(font_dir: Path) -> None:
    faces = _pool(font_dir, 2)
    with pytest.raises(ScheduleError, match="default_weight"):
        _process(faces, default_weight=-1.0)
