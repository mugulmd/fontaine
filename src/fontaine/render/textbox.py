"""Rendering one text-box crop.

The pipeline for a single item:

1. sample a target cap height, resolve the em size that gives it for this face
2. sample text, projected onto the glyphs this face actually has
3. size a working canvas with enough margin for the widest possible crop
4. build a background, optionally lay a scrim over the text area
5. measure the background luminance *under the text*, and pick a text colour at a
   sampled contrast ratio against it
6. draw the text on its own layer, and take the tight ink box from its alpha
7. crop the ink box with independent per-side padding — the text detector's
   imprecision, including boxes tight enough to clip the glyphs
8. apply capture degradations

The tight ink box comes from the rendered alpha rather than from font metrics:
metrics miss accent height and overshoot, and the crop has to sit on what was
actually drawn.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fontaine.config import RenderConfig
from fontaine.contracts import FontFace
from fontaine.render import background as background_module
from fontaine.render import degrade
from fontaine.render.color import (
    RGB,
    ContrastPlan,
    color_with_luminance,
    guaranteed_contrast,
    luminance_extremes,
    mean_luminance,
    plan_contrast,
    relative_luminance,
)
from fontaine.render.metrics import cap_height_ratio, em_size_for_cap_height, load_font
from fontaine.text.corpus import Corpus

Box = tuple[int, int, int, int]


class RenderError(RuntimeError):
    """Raised when a face and text combination produces no ink at all."""


@dataclass(slots=True)
class RenderedCrop:
    """A finished crop, and the full record of how it was made."""

    image: Image.Image
    metadata: dict[str, Any]


class CropRenderer:
    """Turns a face into a crop. Holds the config and corpus; carries no state."""

    def __init__(self, config: RenderConfig | None = None, corpus: Corpus | None = None) -> None:
        self.config = config or RenderConfig()
        self.corpus = corpus or Corpus(self.config.corpus)

    def render(self, face: FontFace, rng: random.Random) -> RenderedCrop:
        """Draw one crop in ``face``, drawing every choice from ``rng``."""
        typography = self.config.typography
        crop_config = self.config.crop

        cap_height = typography.cap_height_px.sample_log(rng)
        em_px = em_size_for_cap_height(face, cap_height)
        font = load_font(face, em_px)
        spacing = typography.letter_spacing.sample(rng) * cap_height

        sample = self.corpus.sample(rng, face.codepoints)
        layout = _layout(font, sample.text, spacing)
        if layout.width <= 0 or layout.height <= 0:
            raise RenderError(f"{face.face_id!r} renders nothing for {sample.text!r}")

        # Enough margin that even the loosest crop stays inside the canvas.
        widest_pad = max(abs(crop_config.pad.lo), abs(crop_config.pad.hi))
        margin = max(2, math.ceil(widest_pad * layout.height) + 4)
        canvas_size = (layout.width + 2 * margin, layout.height + 2 * margin)
        origin = (margin - layout.offset_x, margin - layout.offset_y)
        text_area: Box = (margin, margin, margin + layout.width, margin + layout.height)

        canvas = background_module.build(rng, canvas_size, self.config.background)
        image = canvas.image

        wanted_ratio = self.config.background.contrast_ratio.sample(rng)
        wants_darker = rng.random() < self.config.background.dark_text_prob
        wants_scrim = rng.random() < self.config.background.scrim_prob
        plan = plan_contrast(
            luminance_extremes(_patch(image, text_area)), ratio=wanted_ratio, darker=wants_darker
        )

        # A background whose range spans the text box leaves no colour that holds
        # everywhere; flatten it with a scrim rather than emit an unreadable crop.
        scrim: dict[str, Any] | None = None
        if wants_scrim or plan.achieved < self.config.background.min_contrast:
            scrim, plan = self._scrim(
                rng, image, text_area, margin, plan, ratio=wanted_ratio, forced=not wants_scrim
            )

        patch = _patch(image, text_area)
        color = color_with_luminance(
            rng, plan.luminance, saturated=rng.random() < self.config.background.saturated_prob
        )
        # Re-measure with the colour actually chosen: a saturated hue lands only
        # approximately on the planned luminance.
        achieved_ratio, reference = guaranteed_contrast(
            luminance_extremes(patch), relative_luminance(color)
        )

        neighbor: dict[str, Any] | None = None
        if rng.random() < crop_config.neighbor_bleed_prob:
            neighbor = self._draw_neighbors(rng, image, font, origin, layout, color, spacing, face)

        text_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        _draw_text(ImageDraw.Draw(text_layer), origin, sample.text, font, color, spacing)
        ink_box = text_layer.getchannel("A").getbbox()
        if ink_box is None:
            raise RenderError(f"{face.face_id!r} drew no ink for {sample.text!r}")
        image = Image.alpha_composite(image.convert("RGBA"), text_layer).convert("RGB")

        crop_box, pads = _jittered_box(rng, ink_box, crop_config.pad, canvas_size)
        crop = image.crop(crop_box)
        crop, degradations = degrade.apply(rng, crop, self.config.degrade)

        return RenderedCrop(
            image=crop,
            metadata={
                "text": sample.text,
                "text_kind": sample.kind,
                "text_dropped": sample.dropped,
                "cap_height_px": round(cap_height, 2),
                "em_px": em_px,
                "cap_ratio": round(cap_height_ratio(face), 4),
                "letter_spacing_px": round(spacing, 3),
                "ink_box": list(ink_box),
                "crop_box": list(crop_box),
                "crop_size": list(crop.size),
                "crop_pad": [round(pad, 4) for pad in pads],
                "background": canvas.source,
                "background_detail": canvas.detail,
                "background_luminance": round(mean_luminance(patch), 4),
                "background_reference_luminance": round(reference, 4),
                "text_color": list(color),
                "text_darker": plan.darker,
                "contrast_wanted": round(wanted_ratio, 2),
                "contrast_achieved": round(achieved_ratio, 2),
                "scrim": scrim,
                "neighbor": neighbor,
                "degradations": degradations,
            },
        )

    #: Opacities tried in turn when a scrim has to reach the contrast floor. The
    #: last is a fully opaque panel, which flattens any background — so the floor
    #: is always reachable rather than merely attempted.
    _SCRIM_ESCALATION = (0.85, 1.0)

    def _scrim(
        self,
        rng: random.Random,
        image: Image.Image,
        text_area: Box,
        margin: int,
        plan: ContrastPlan,
        *,
        ratio: float,
        forced: bool,
    ) -> tuple[dict[str, Any], ContrastPlan]:
        """Lay a panel behind the text, darkening or lightening until the floor is met.

        The text side is pinned to the panel's: a light panel under dark text, a
        dark panel under light text. Letting the side float would allow dark text
        to be planned against a black scrim, which no amount of opacity fixes.
        """
        # Light panel under dark text — the point is to move the background away
        # from the text, not towards it.
        color: RGB = (255, 255, 255) if plan.darker else (0, 0, 0)
        inset = rng.uniform(0.2, 1.0) * margin
        box: Box = (
            round(text_area[0] - inset),
            round(text_area[1] - inset),
            round(text_area[2] + inset),
            round(text_area[3] + inset),
        )
        opacities = [self.config.background.scrim_opacity.sample(rng), *self._SCRIM_ESCALATION]
        applied: list[float] = []
        for opacity in opacities:
            background_module.add_scrim(image, box, color, opacity)
            applied.append(round(opacity, 3))
            plan = plan_contrast(
                luminance_extremes(_patch(image, text_area)), ratio=ratio, darker=plan.darker
            )
            if plan.achieved >= self.config.background.min_contrast:
                break
        return (
            {"color": list(color), "box": list(box), "opacity": applied, "forced": forced},
            plan,
        )

    def _draw_neighbors(
        self,
        rng: random.Random,
        image: Image.Image,
        font: ImageFont.FreeTypeFont,
        origin: tuple[int, int],
        layout: _Layout,
        color: RGB,
        spacing: float,
        face: FontFace,
    ) -> dict[str, Any]:
        """Bleed a neighbouring line in at an edge, as a loose detector box does.

        Drawn on its own layer so it cannot affect the target's ink box: the crop
        must be positioned on the text being labelled, not on its neighbour.
        """
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        sides = rng.choice((("top",), ("bottom",), ("top", "bottom")))
        for side in sides:
            gap = self.config.crop.neighbor_gap.sample(rng) * layout.height
            delta = -(layout.height + gap) if side == "top" else layout.height + gap
            neighbor_text = self.corpus.sample(rng, face.codepoints).text
            _draw_text(
                draw,
                (origin[0], origin[1] + round(delta)),
                neighbor_text,
                font,
                color,
                spacing,
            )
        image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))
        return {"sides": list(sides)}


def _patch(image: Image.Image, box: Box) -> np.ndarray:
    return np.asarray(image.crop(box), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class _Layout:
    """Where the ink lands relative to the drawing origin, and how big it is."""

    width: int
    height: int
    offset_x: int
    offset_y: int


def _layout(font: ImageFont.FreeTypeFont, text: str, spacing: float) -> _Layout:
    left, top, right, bottom = (round(value) for value in font.getbbox(text))
    # getlength is the pen advance, which exceeds the ink box for a trailing
    # space or a glyph with right bearing; take whichever is wider.
    advance = font.getlength(text) + spacing * max(0, len(text) - 1)
    width = max(right, math.ceil(advance)) - min(0, left)
    return _Layout(
        width=max(1, width),
        height=max(1, bottom - top),
        offset_x=left,
        offset_y=top,
    )


def _draw_text(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    color: RGB,
    spacing: float,
) -> None:
    """Draw ``text``, per-character only when tracking is in play.

    The single-call path keeps the shaper's kerning and ligatures intact, so it is
    used whenever spacing is zero.
    """
    if spacing == 0:
        draw.text(origin, text, font=font, fill=color)
        return
    x, y = origin
    for position, char in enumerate(text):
        # Measuring the prefix keeps the kerned advances and adds tracking on top.
        offset = font.getlength(text[:position]) + spacing * position
        draw.text((x + offset, y), char, font=font, fill=color)


def _jittered_box(
    rng: random.Random, ink_box: Box, pad: Any, canvas_size: tuple[int, int]
) -> tuple[Box, tuple[float, float, float, float]]:
    """Pad the ink box independently on each side, in units of ink height."""
    left, top, right, bottom = ink_box
    ink_height = max(1, bottom - top)
    pads = (pad.sample(rng), pad.sample(rng), pad.sample(rng), pad.sample(rng))
    box = (
        round(left - pads[0] * ink_height),
        round(top - pads[1] * ink_height),
        round(right + pads[2] * ink_height),
        round(bottom + pads[3] * ink_height),
    )
    width, height = canvas_size
    clamped: Box = (
        max(0, min(box[0], width - 1)),
        max(0, min(box[1], height - 1)),
        min(width, max(box[2], 1)),
        min(height, max(box[3], 1)),
    )
    # A negative pad on both sides of a short crop could invert it; keep 1px.
    return (
        clamped[0],
        clamped[1],
        max(clamped[0] + 1, clamped[2]),
        max(clamped[1] + 1, clamped[3]),
    ), pads
