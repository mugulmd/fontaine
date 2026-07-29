"""Replaying a stream from disk.

Yields the same :class:`~fontaine.contracts.Sample` objects the live generator
does, so an online recognizer cannot tell the difference between training against
a frozen stream and against a live one.

Reading is lazy: one JSONL line and one PNG at a time, never the whole stream.
A stream that does not fit in memory is the normal case, not the exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

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


def read_stream(directory: Path, *, with_images: bool = True) -> Iterator[Sample]:
    """Replay the stream in order.

    ``with_images=False`` skips decoding the PNGs, for when only the labels and
    metadata are wanted — scoring a discovery curve, say.
    """
    path = directory / ANNOTATIONS_NAME
    if not path.is_file():
        raise StreamNotFound(f"no {ANNOTATIONS_NAME} in {directory}")

    with path.open() as annotations:
        for line in annotations:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image: Image.Image | None = None
            if with_images:
                with Image.open(directory / record["image"]) as handle:
                    # Load eagerly: the file handle closes when this block exits.
                    image = handle.convert("RGB")
            yield Sample(
                index=record["index"],
                image=image,  # type: ignore[arg-type]
                label=record["label"],
                metadata=record.get("meta", {}),
            )
