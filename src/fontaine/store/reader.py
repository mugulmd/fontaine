"""Replaying a stream from disk.

Yields the same :class:`~fontaine.contracts.Sample` objects the live generator
does, so an online recognizer cannot tell the difference between training against
a frozen stream and against a live one.

Reading is lazy: one JSONL line and one PNG at a time, never the whole stream.
A stream that does not fit in memory is the normal case, not the exception.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from fontaine import config as config_module
from fontaine.config import StreamConfig
from fontaine.contracts import Sample
from fontaine.fonts.registry import FontRegistry
from fontaine.store.writer import ANNOTATIONS_NAME, MANIFEST_NAME


class StreamNotFound(FileNotFoundError):
    """Raised when a directory holds no complete stream."""


def read_manifest(directory: Path) -> dict[str, Any]:
    """The stream's manifest: config snapshot, registry and discovery ground truth."""
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise StreamNotFound(
            f"no {MANIFEST_NAME} in {directory} — the run may have been interrupted"
        )
    return json.loads(path.read_text())


def read_registry(directory: Path) -> FontRegistry:
    """The label space the stream was generated from."""
    return FontRegistry.from_dict(read_manifest(directory)["registry"])


def read_config(directory: Path) -> StreamConfig:
    """The exact configuration the stream was generated with."""
    return config_module.from_dict(read_manifest(directory)["config"], context="manifest")


def read_annotations(directory: Path) -> Iterator[dict[str, Any]]:
    """Replay the annotation records in order, without decoding any image.

    For passes that only need labels and metadata — scoring a discovery curve,
    slicing error rates by cap height — where decoding every PNG would dominate
    the runtime for nothing.
    """
    path = directory / ANNOTATIONS_NAME
    if not path.is_file():
        raise StreamNotFound(f"no {ANNOTATIONS_NAME} in {directory}")

    with path.open() as annotations:
        for line in annotations:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def read_stream(directory: Path) -> Iterator[Sample]:
    """Replay the stream in order, as the samples a live generator would yield.

    Every sample carries its image. Use :func:`read_annotations` when the images
    are not needed — this deliberately has no way to omit them, so a ``Sample``
    always means the same thing whatever produced it.
    """
    for record in read_annotations(directory):
        with Image.open(directory / record["image"]) as handle:
            # Load eagerly: the file handle closes when this block exits.
            image = handle.convert("RGB")
        yield Sample(
            index=record["index"],
            image=image,
            label=record["label"],
            metadata=record.get("meta", {}),
        )
