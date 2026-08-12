"""Data contracts shared by the generator and the online recognizer.

This is the only module both programs are expected to import. Everything else
is private to one side or the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class FontFace(BaseModel):
    """A single renderable font face: one file, or one index inside a ``.ttc``.

    ``face_id`` is the classification label — a stable slug derived from the
    font's ``name`` table, so it survives the file being moved or renamed.
    """

    face_id: str
    family: str
    subfamily: str
    path: Path
    #: Index of this face inside a ``.ttc``/``.otc`` collection, 0 for a plain file.
    #: Needed to load the right face back out of the file.
    font_number: int = 0
    weight: int
    width_class: int
    italic: bool
    monospace: bool
    variable: bool
    units_per_em: int
    n_glyphs: int
    #: Left out of the serialized form: derivable from the file, and large. A
    #: replayed stream needs the labels, not the ability to re-render.
    codepoints: frozenset[int] = Field(default_factory=frozenset, exclude=True, repr=False)

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

    # PIL images have no pydantic schema, and there is nothing to validate about
    # one here: the renderer produced it.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    image: Image.Image
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Recognizer(ABC):
    """The one thing a challenger implements: a model that predicts, then learns.

    Subclass it in a file under ``models/`` and it becomes selectable as
    ``fontaine recognize --model <name>``. Nothing inside ``fontaine`` needs to
    change to add one, which is the point: the framework knows about this class
    and nothing else about your model.

    Three properties the evaluation loop relies on:

    * **A crop, not a** :class:`Sample`. The metadata records the exact cap
      height, contrast and background used to draw the item, so a model handed
      the whole sample could read the answer off the generator instead of the
      pixels. Only the image is passed.
    * **Test-then-train.** :meth:`predict` is always called before
      :meth:`learn` for the same item, and the item is scored on that
      prediction. Learning from an item you have already been scored on is
      free; there is no way to be scored twice.
    * **An open label set.** A font never seen before can arrive at any point,
      and the only announcement is a :meth:`learn` call carrying its label.
      Nothing tells you the label space up front — discovering it is the task.

    The loop hands the *same* image object to :meth:`predict` and then to
    :meth:`learn`, so work done for the prediction can be cached across the
    pair rather than repeated; ``models/baseline.py`` does exactly that.
    """

    #: How ``--model`` names this recognizer. Required on every concrete
    #: subclass: it is what the CLI matches against, and deriving it from the
    #: class name would make renaming the class silently rename the model.
    name: ClassVar[str]

    @abstractmethod
    def predict(self, image: Image.Image) -> str | None:
        """The ``face_id`` this crop was drawn with, or ``None`` to abstain.

        Abstaining is scored as a miss, not skipped — a model that never answers
        scores zero. It exists for the honest case of having seen no labels yet.
        """

    @abstractmethod
    def learn(self, image: Image.Image, label: str) -> None:
        """Fold one labelled crop in. ``label`` may be a font never seen before."""
