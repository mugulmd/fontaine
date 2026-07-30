from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import pytest
from synthetic_fonts import build_font

from fontaine.config import ArrivalConfig
from fontaine.fonts import registry as font_registry
from fontaine.stream.arrival import ArrivalProcess

ALNUM = string.ascii_letters + string.digits


def _pool(font_dir: Path, count: int = 12, *, families: int | None = None) -> list:
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


def _process(faces: list, **overrides: Any) -> ArrivalProcess:
    return ArrivalProcess(faces, ArrivalConfig(**overrides), seed=0)


def test_first_item_is_always_a_new_font(font_dir: Path) -> None:
    process = _process(_pool(font_dir))
    first = process.step()

    assert first.first_seen is True
    assert first.index == 0
    assert first.n_seen == 1
    assert process.stats.face_first_seen[first.face.face_id] == 0


def test_same_seed_gives_the_same_sequence(font_dir: Path) -> None:
    faces = _pool(font_dir)
    first = [item.face.face_id for item in _process(faces).take(200)]
    second = [item.face.face_id for item in _process(faces).take(200)]

    assert first == second


def test_different_seeds_diverge(font_dir: Path) -> None:
    faces = _pool(font_dir)
    first = [item.face.face_id for item in ArrivalProcess(faces, seed=0).take(200)]
    second = [item.face.face_id for item in ArrivalProcess(faces, seed=1).take(200)]

    assert first != second


def test_discovery_is_progressive_not_immediate(font_dir: Path) -> None:
    process = _process(_pool(font_dir, 12), concentration=2.0)
    early = process.take(3)

    # A handful of items cannot have revealed the whole pool.
    assert len({item.face.face_id for item in early}) < 12
    assert not process.exhausted


def test_the_pool_bounds_distinct_faces_not_items(font_dir: Path) -> None:
    faces = _pool(font_dir, 6)
    process = _process(faces, concentration=50.0)
    items = process.take(500)

    assert len(items) == 500
    assert len({item.face.face_id for item in items}) == 6
    assert process.exhausted
    assert process.new_font_probability() == 0.0
    assert sum(item.first_seen for item in items) == 6


def test_first_seen_flags_agree_with_the_recorded_ground_truth(font_dir: Path) -> None:
    process = _process(_pool(font_dir))
    items = process.take(300)

    for item in items:
        if item.first_seen:
            assert process.stats.face_first_seen[item.face.face_id] == item.index
    # Every face's recorded first-seen step is the earliest it actually appears.
    for face_id, step in process.stats.face_first_seen.items():
        appearances = [item.index for item in items if item.face.face_id == face_id]
        assert min(appearances) == step


def test_counts_add_up_to_the_number_of_items(font_dir: Path) -> None:
    process = _process(_pool(font_dir))
    process.take(250)

    assert sum(process.stats.face_counts.values()) == 250
    assert sum(process.stats.label_counts.values()) == 250
    assert process.stats.n_items == 250


def test_a_new_face_of_a_known_family_is_not_a_new_label(font_dir: Path) -> None:
    """At family granularity, discovery is about families, not files."""
    faces = _pool(font_dir, 12, families=2)
    process = ArrivalProcess(faces, seed=0, label_granularity="family")
    items = process.take(400)

    assert len(process.stats.label_counts) == 2
    assert len(process.stats.face_counts) > 2
    # Some face arrived new while its family was already known.
    assert any(item.first_seen and not item.label_first_seen for item in items)


def test_popularity_is_rich_get_richer(font_dir: Path) -> None:
    faces = _pool(font_dir, 12)
    process = _process(faces, concentration=2.0, half_life=0)
    process.take(2_000)

    shares = sorted(process.stats.face_counts.values(), reverse=True)
    # Far from uniform: the leader takes much more than its 1/12 share.
    assert shares[0] / 2_000 > 3 / 12


def test_forgetting_keeps_discovery_alive(font_dir: Path) -> None:
    """Without forgetting, P(new) decays like 1/t and the stream ossifies."""
    faces = _pool(font_dir, 40)
    plain = _process(faces, concentration=4.0, half_life=0)
    forgetting = _process(faces, concentration=4.0, half_life=500)
    plain.take(20_000)
    forgetting.take(20_000)

    # Forgetting reaches the whole pool; plain stalls well short of it, with the
    # chance of anything new having collapsed towards zero.
    assert len(forgetting.stats.face_counts) == 40
    assert len(plain.stats.face_counts) < 40
    assert plain.new_font_probability() < 0.001


def test_forgetting_lets_a_font_fade(font_dir: Path) -> None:
    faces = _pool(font_dir, 12)
    process = _process(faces, concentration=4.0, half_life=200)
    process.take(3_000)

    popularity = process.popularity()
    assert popularity
    assert sum(popularity.values()) == pytest.approx(1.0)
    # Recency-weighted shares, so faces absent from the recent past sit near zero
    # even though their lifetime counts are non-zero.
    assert min(popularity.values()) < 1e-3
    assert all(process.stats.face_counts[face_id] > 0 for face_id in popularity)


def test_shuffling_changes_discovery_order_only(font_dir: Path) -> None:
    faces = _pool(font_dir, 12)
    ordered = _process(faces, shuffle_pool=False, concentration=50.0)
    ordered.take(500)

    first_seen = ordered.stats.face_first_seen
    discovery_order = sorted(first_seen, key=lambda face_id: first_seen[face_id])
    assert discovery_order == [face.face_id for face in faces]


def test_invalid_configuration_is_rejected(font_dir: Path) -> None:
    faces = _pool(font_dir, 3)
    with pytest.raises(ValueError, match="at least one face"):
        ArrivalProcess([])
    with pytest.raises(ValueError, match="concentration"):
        _process(faces, concentration=0.0)
    with pytest.raises(ValueError, match="half_life"):
        _process(faces, half_life=-5)
