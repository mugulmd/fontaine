"""Contact sheets for eyeballing generated crops.

Crops come out at wildly different sizes, so each is scaled to a common height
for the grid — the same normalization a recognizer would do at its input. Cells
sit on a mid-grey sheet with a hairline border, so a crop's own edges stay visible
whether its background is light or dark.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

SHEET_BG = (43, 43, 43)
CELL_BG = (28, 28, 28)
BORDER = (90, 90, 90)
LABEL_COLOR = (222, 222, 222)


def contact_sheet(
    items: list[tuple[Image.Image, str]],
    *,
    columns: int = 6,
    cell_height: int = 56,
    max_cell_width: int = 260,
    padding: int = 10,
    label_size: int = 12,
) -> Image.Image:
    """Tile ``(crop, label)`` pairs into a single annotated image."""
    if not items:
        raise ValueError("nothing to lay out")

    font = ImageFont.load_default(size=label_size)
    label_height = label_size + 4
    scaled = []
    for image, label in items:
        cell, clipped = _scale_to_height(image, cell_height, max_cell_width)
        # Say so rather than silently showing a partial crop as if it were whole.
        scaled.append((cell, f"{label} ›" if clipped else label))

    cell_width = max(image.width for image, _ in scaled)
    columns = max(1, min(columns, len(scaled)))
    rows = (len(scaled) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            padding + columns * (cell_width + padding),
            padding + rows * (cell_height + label_height + padding),
        ),
        SHEET_BG,
    )
    draw = ImageDraw.Draw(sheet)

    for position, (image, label) in enumerate(scaled):
        row, column = divmod(position, columns)
        x = padding + column * (cell_width + padding)
        y = padding + row * (cell_height + label_height + padding)
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill=CELL_BG, outline=BORDER)
        # Centre the crop in its cell so narrow ones do not read as left-aligned.
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + (cell_height - image.height) // 2))
        draw.text(
            (x, y + cell_height + 2),
            _truncate(draw, label, font, cell_width),
            font=font,
            fill=LABEL_COLOR,
        )
    return sheet


def _scale_to_height(image: Image.Image, height: int, max_width: int) -> tuple[Image.Image, bool]:
    """Scale to a common height, clipping over-wide crops. Reports whether it clipped.

    Height is normalized rather than fitting the whole crop inside the cell, so
    letterforms stay judgeable at a glance — a long phrase scaled to fit width
    would be too small to assess.
    """
    scale = height / image.height
    width = max(1, round(image.width * scale))
    resampling = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST
    scaled = image.resize((width, height), resampling)
    if scaled.width <= max_width:
        return scaled, False
    # Keep the left edge: that is where the text starts.
    return scaled.crop((0, 0, max_width, height)), True


def _truncate(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text + "…"
