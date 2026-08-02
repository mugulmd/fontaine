"""Data contracts shared by the generator and the online recognizer.

This is the only module both programs are expected to import. Everything else
is private to one side or the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field


class FontFace(BaseModel):
    """A single renderable font face.

    ``face_id`` is a stable slugs derived from the font's ``name`` table,
    so it survives the file being moved or renamed.
    """

    face_id: str
    path: Path
    weight: int
    width_class: int
    italic: bool
    monospace: bool
    variable: bool
    units_per_em: int
    n_glyphs: int
    codepoints: frozenset[int] = Field(default_factory=frozenset)

    def covers(self, text: str) -> bool:
        """Whether every character of ``text`` has a glyph in this face."""
        return all(ord(char) in self.codepoints for char in text)

    def missing_from(self, text: str) -> str:
        """The characters of ``text`` this face has no glyph for, deduplicated."""
        seen: dict[str, None] = {}
        for char in text:
            if ord(char) not in self.codepoints:
                seen[char] = None
        return "".join(seen)


class Sample(BaseModel):
    """One item of the stream: a crop around a text box, and the font that drew it.

    ``metadata`` carries the full generation parameters (px size, contrast,
    background source, crop jitter, ...) so failures can be sliced after the
    fact. The recognizer must never read it — only ``image`` at prediction time
    and ``label`` when it is subsequently allowed to learn.
    """

    index: int
    image: Image.Image
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)
