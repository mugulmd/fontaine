"""Materializing a stream to disk.

Layout::

    <stream>/manifest.json        config, font registry and discovery ground truth
    <stream>/annotations.jsonl    one line per item, in stream order
    <stream>/crops/00000/...png   the crops, sharded so no directory gets huge

The JSONL line order *is* the stream order, which makes a frozen stream readable
one line at a time without loading an index — the same way it is consumed live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

from fontaine import config as config_module
from fontaine.contracts import Sample
from fontaine.stream.generator import StreamGenerator

MANIFEST_NAME = "manifest.json"
ANNOTATIONS_NAME = "annotations.jsonl"
CROPS_DIR = "crops"
DEFAULT_SHARD_SIZE = 10_000


@dataclass(slots=True)
class StreamReport:
    """What a generation run produced."""

    directory: Path
    n_items: int
    n_labels: int
    n_faces: int
    n_skipped: int


def _package_version() -> str:
    try:
        return version("fontaine")
    except PackageNotFoundError:  # pragma: no cover - only when run from a checkout
        return "unknown"


def crop_path(index: int, shard_size: int = DEFAULT_SHARD_SIZE) -> str:
    """Relative path of a crop, sharded by index."""
    return f"{CROPS_DIR}/{index // shard_size:05d}/{index:08d}.png"


def write_stream(
    generator: StreamGenerator,
    count: int,
    directory: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    on_item: Callable[[Sample], None] | None = None,
) -> StreamReport:
    """Generate ``count`` items and write them under ``directory``.

    The manifest is written last: a directory with a manifest is a complete
    stream, so an interrupted run cannot be mistaken for a finished one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = 0

    with (directory / ANNOTATIONS_NAME).open("w") as annotations:
        for sample in generator.take(count):
            relative = crop_path(sample.index, shard_size)
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            sample.image.save(destination)
            annotations.write(
                json.dumps(
                    {
                        "index": sample.index,
                        "label": sample.label,
                        "image": relative,
                        "meta": sample.metadata,
                    }
                )
                + "\n"
            )
            written += 1
            if on_item is not None:
                on_item(sample)

    stats = generator.stats
    manifest = {
        "fontaine_version": _package_version(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": generator.config.seed,
        "n_items": written,
        "shard_size": shard_size,
        "label_granularity": generator.registry.label_granularity,
        "config": config_module.to_dict(generator.config),
        "registry": generator.registry.to_dict(),
        "arrival": stats.to_dict(),
        "skipped": [
            {"index": item.index, "face_id": item.face_id, "error": item.error}
            for item in generator.skipped
        ],
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    return StreamReport(
        directory=directory,
        n_items=written,
        n_labels=len(stats.label_counts),
        n_faces=len(stats.face_counts),
        n_skipped=len(generator.skipped),
    )
