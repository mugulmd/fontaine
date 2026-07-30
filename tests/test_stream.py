from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
from synthetic_fonts import build_font

from fontaine.config import (
    ArrivalConfig,
    BackgroundConfig,
    CorpusConfig,
    CropConfig,
    FontsConfig,
    Range,
    RenderConfig,
    StreamConfig,
    TypographyConfig,
)
from fontaine.fonts import registry as font_registry
from fontaine.store import reader, writer
from fontaine.stream.generator import StreamGenerator

ALNUM = string.ascii_letters + string.digits


@pytest.fixture
def stream_setup(font_dir: Path) -> tuple[StreamConfig, font_registry.FontRegistry]:
    for index in range(6):
        build_font(
            font_dir / f"face-{index}.ttf",
            chars=ALNUM,
            family=f"Fam{index % 3}",
            subfamily=f"Style{index}",
        )
    settings = StreamConfig(
        seed=3,
        fonts=FontsConfig(font_dir=font_dir),
        arrival=ArrivalConfig(concentration=6.0, half_life=100),
        render=RenderConfig(
            corpus=CorpusConfig(kinds={"word": 1.0}, casing={"as_is": 1.0}),
            typography=TypographyConfig(cap_height_px=Range(14, 20)),
            background=BackgroundConfig(sources={"solid": 1.0}, scrim_prob=0.0),
            crop=CropConfig(pad=Range(0.0, 0.2)),
        ),
    )
    registry = font_registry.scan(font_dir, charset="ascii_alnum")
    return settings, registry


def test_generator_yields_labelled_samples(stream_setup) -> None:
    settings, registry = stream_setup
    samples = list(StreamGenerator(settings, registry).take(20))

    assert len(samples) == 20
    assert [sample.index for sample in samples] == list(range(20))
    labels = {face.face_id for face in registry}
    for sample in samples:
        assert sample.label in labels
        assert sample.image.width > 0
        assert sample.metadata["face_id"] == sample.label
        assert "cap_height_px" in sample.metadata


def test_generator_carries_the_discovery_ground_truth(stream_setup) -> None:
    settings, registry = stream_setup
    generator = StreamGenerator(settings, registry)
    samples = list(generator.take(60))

    assert samples[0].metadata["first_seen"] is True
    for sample in samples:
        if sample.metadata["label_first_seen"]:
            assert generator.stats.label_first_seen[sample.label] == sample.index
    assert sum(sample.metadata["first_seen"] for sample in samples) == len(
        generator.stats.face_counts
    )


def test_generator_labels_at_family_granularity(font_dir: Path, stream_setup) -> None:
    settings, _ = stream_setup
    registry = font_registry.scan(
        settings.fonts.font_dir, charset="ascii_alnum", label_granularity="family"
    )
    samples = list(StreamGenerator(settings, registry).take(30))

    assert {sample.label for sample in samples} <= {"fam0", "fam1", "fam2"}
    # The face is still recorded, so a family-labelled stream can be re-scored.
    assert all(sample.metadata["face_id"] != sample.label for sample in samples)


def test_generator_rejects_an_empty_label_space(stream_setup) -> None:
    settings, _ = stream_setup
    empty = font_registry.FontRegistry(faces=[])
    with pytest.raises(ValueError, match="empty label space"):
        StreamGenerator(settings, empty)


def test_written_stream_has_the_expected_layout(stream_setup, tmp_path: Path) -> None:
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    report = writer.write_stream(StreamGenerator(settings, registry), 15, directory)

    assert report.n_items == 15
    assert report.n_skipped == 0
    assert (directory / writer.MANIFEST_NAME).is_file()
    assert len(list((directory / writer.ANNOTATIONS_NAME).read_text().splitlines())) == 15
    assert len(list(directory.glob("crops/*/*.png"))) == 15


def test_crops_are_sharded_by_index() -> None:
    assert writer.crop_path(0) == "crops/00000/00000000.png"
    assert writer.crop_path(9_999) == "crops/00000/00009999.png"
    assert writer.crop_path(10_000) == "crops/00001/00010000.png"
    assert writer.crop_path(10, shard_size=5) == "crops/00002/00000010.png"


def test_replay_is_indistinguishable_from_live_generation(stream_setup, tmp_path: Path) -> None:
    """The point of the writer/reader pair: one code path for frozen and live."""
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    writer.write_stream(StreamGenerator(settings, registry), 12, directory)

    live = list(StreamGenerator(settings, registry).take(12))
    replayed = list(reader.read_stream(directory))

    assert [sample.label for sample in replayed] == [sample.label for sample in live]
    assert [sample.index for sample in replayed] == [sample.index for sample in live]
    assert all(
        first.image.tobytes() == second.image.tobytes()
        for first, second in zip(live, replayed, strict=True)
    )


def test_annotations_can_be_replayed_without_decoding_images(stream_setup, tmp_path: Path) -> None:
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    writer.write_stream(StreamGenerator(settings, registry), 8, directory)

    records = list(reader.read_annotations(directory))

    assert len(records) == 8
    assert [record["index"] for record in records] == list(range(8))
    assert all(record["label"] for record in records)
    assert all(record["meta"]["cap_height_px"] for record in records)


def test_manifest_records_config_registry_and_ground_truth(stream_setup, tmp_path: Path) -> None:
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    writer.write_stream(StreamGenerator(settings, registry), 25, directory)

    manifest = reader.read_manifest(directory)
    assert manifest["seed"] == 3
    assert manifest["n_items"] == 25
    assert manifest["arrival"]["n_items"] == 25
    assert sum(manifest["arrival"]["label_counts"].values()) == 25

    restored = reader.read_config(directory)
    assert restored.seed == settings.seed
    assert restored.arrival.concentration == settings.arrival.concentration
    assert restored.render.typography.cap_height_px == settings.render.typography.cap_height_px

    label_space = reader.read_registry(directory)
    assert len(label_space) == len(registry)
    assert label_space.labels == registry.labels


def test_reading_an_incomplete_stream_fails_loudly(stream_setup, tmp_path: Path) -> None:
    """A directory without a manifest is an interrupted run, not a valid stream."""
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    writer.write_stream(StreamGenerator(settings, registry), 5, directory)
    (directory / writer.MANIFEST_NAME).unlink()

    with pytest.raises(reader.StreamNotFound, match="interrupted"):
        reader.read_manifest(directory)
    with pytest.raises(reader.StreamNotFound):
        reader.read_registry(tmp_path / "nowhere")


def test_annotation_records_are_self_describing(stream_setup, tmp_path: Path) -> None:
    settings, registry = stream_setup
    directory = tmp_path / "stream"
    writer.write_stream(StreamGenerator(settings, registry), 5, directory)

    lines = (directory / writer.ANNOTATIONS_NAME).read_text().splitlines()
    record = json.loads(lines[0])

    assert set(record) == {"index", "label", "image", "meta"}
    assert (directory / record["image"]).is_file()
    assert record["meta"]["text"]
