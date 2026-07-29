"""Command line entry points."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fontaine.config import DEFAULT_CONFIG_PATH, StreamConfig, load_stream_config
from fontaine.fonts import registry as font_registry
from fontaine.fonts.coverage import CHARSET_PRESETS, resolve_charset
from fontaine.render import background as background_module
from fontaine.render.textbox import CropRenderer, RenderError
from fontaine.rng import item_rng
from fontaine.viz.contact_sheet import contact_sheet

app = typer.Typer(help="Synthetic font-recognition streams.", no_args_is_help=True)
fonts_app = typer.Typer(help="Inspect the font universe.", no_args_is_help=True)
app.add_typer(fonts_app, name="fonts")

console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Stream config YAML.")]
FontDirOption = Annotated[
    Path | None, typer.Option("--font-dir", help="Override the config's font dir.")
]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", help="Surface fontTools' per-table warnings.")
]


def _load(config: Path) -> StreamConfig:
    try:
        return load_stream_config(config)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error


def _scan(settings: StreamConfig, *, verbose: bool = False) -> font_registry.FontRegistry:
    fonts = settings.fonts
    try:
        return font_registry.scan(
            fonts.font_dir,
            charset=fonts.admission_charset,
            label_granularity=fonts.label_granularity,
            exclude=fonts.exclude,
            include_variable=fonts.include_variable,
            verbose=verbose,
        )
    except NotADirectoryError as error:
        console.print(f"[red]{error}[/red]")
        console.print("Drop .ttf/.otf files in, or pass --font-dir.", style="dim")
        raise typer.Exit(code=1) from error


@fonts_app.command("scan")
def fonts_scan(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    font_dir: FontDirOption = None,
    charset: Annotated[
        str | None, typer.Option("--charset", help="Override the admission charset.")
    ] = None,
    include_variable: Annotated[
        bool, typer.Option("--include-variable", help="Keep variable fonts in the label space.")
    ] = False,
    list_faces: Annotated[
        bool, typer.Option("--list/--no-list", help="Print the per-face table.")
    ] = True,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the registry snapshot to this path.")
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """Build the label space from the font dir and report what was kept and dropped."""
    settings = _load(config)
    if font_dir is not None:
        settings.fonts.font_dir = font_dir
    if charset is not None:
        settings.fonts.admission_charset = charset
    if include_variable:
        settings.fonts.include_variable = True

    registry = _scan(settings, verbose=verbose)
    required = resolve_charset(settings.fonts.admission_charset)
    origin = " (preset)" if settings.fonts.admission_charset in CHARSET_PRESETS else " (literal)"
    console.print(
        f"[bold]{settings.fonts.font_dir}[/bold] — admission charset "
        f"[cyan]{settings.fonts.admission_charset}[/cyan]{origin}, {len(required)} required glyphs"
    )

    if list_faces and registry.faces:
        console.print(_faces_table(registry))
    if registry.rejected:
        console.print(_rejected_table(registry))
    if registry.unreadable:
        console.print(_unreadable_table(registry))

    console.print(
        f"\n[green]{len(registry.faces)}[/green] faces kept "
        f"across [green]{len(registry.families)}[/green] families "
        f"→ [bold]{len(registry.labels)}[/bold] labels "
        f"at [cyan]{registry.label_granularity}[/cyan] granularity"
    )
    if registry.rejected or registry.unreadable:
        console.print(
            f"[yellow]{len(registry.rejected)}[/yellow] faces rejected, "
            f"[yellow]{len(registry.unreadable)}[/yellow] files unreadable"
        )
    if not registry.faces:
        console.print("[red]empty label space — nothing to generate a stream from[/red]")
        raise typer.Exit(code=1)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(registry.to_dict(), indent=2))
        console.print(f"registry snapshot → [bold]{json_out}[/bold]")


@app.command("preview")
def preview(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    count: Annotated[int, typer.Option("--count", "-n", help="How many crops to render.")] = 48,
    out: Annotated[Path, typer.Option("--out", "-o", help="Contact sheet path.")] = Path(
        "data/preview.png"
    ),
    columns: Annotated[int, typer.Option("--columns", help="Cells per row.")] = 6,
    cell_height: Annotated[int, typer.Option("--cell-height", help="Row height in pixels.")] = 56,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the config's seed.")] = None,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Render a batch of crops to a contact sheet, to check the renders by eye.

    Faces are taken in registry order and cycled, so a large enough count shows
    every font in the label space at least once.
    """
    settings = _load(config)
    if font_dir is not None:
        settings.fonts.font_dir = font_dir
    if seed is not None:
        settings.seed = seed

    registry = _scan(settings, verbose=verbose)
    if not registry.faces:
        console.print("[red]empty label space — nothing to render[/red]")
        raise typer.Exit(code=1)

    images = background_module.count_images(settings.render.background)
    sources = background_module.available_sources(settings.render.background)
    console.print(
        f"{len(registry.faces)} faces, {images} background image(s) in "
        f"[bold]{settings.render.background.photo_dir}[/bold] — "
        f"sources: {', '.join(sorted(sources))}"
    )
    if not images and "photo" in settings.render.background.sources:
        console.print(
            "no PNGs found, falling back to synthetic canvases", style="yellow"
        )

    renderer = CropRenderer(settings.render)
    cells: list[tuple[object, str]] = []
    stats: Counter[str] = Counter()
    failures: list[str] = []
    for index in range(count):
        face = registry.faces[index % len(registry.faces)]
        rng = item_rng(settings.seed, index)
        try:
            crop = renderer.render(face, rng)
        except RenderError as error:
            failures.append(str(error))
            continue
        stats[crop.metadata["background"]] += 1
        stats[f"kind:{crop.metadata['text_kind']}"] += 1
        label = f"{face.label(registry.label_granularity)}  {crop.metadata['cap_height_px']:.0f}px"
        cells.append((crop.image, label))

    if not cells:
        console.print("[red]every render failed[/red]")
        for failure in failures[:5]:
            console.print(f"  {failure}", style="dim")
        raise typer.Exit(code=1)

    sheet = contact_sheet(cells, columns=columns, cell_height=cell_height)  # type: ignore[arg-type]
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    console.print(f"{len(cells)} crops → [bold]{out}[/bold] ({sheet.width}×{sheet.height})")
    backgrounds = {key: value for key, value in stats.items() if not key.startswith("kind:")}
    kinds = {key.removeprefix("kind:"): value for key, value in stats.items() if key.startswith("kind:")}
    console.print(f"backgrounds: {_summarize(backgrounds)}", style="dim")
    console.print(f"content kinds: {_summarize(kinds)}", style="dim")
    if failures:
        console.print(f"[yellow]{len(failures)} renders failed[/yellow]: {failures[0]}")


def _summarize(counts: dict[str, int]) -> str:
    return "  ".join(f"{key} {value}" for key, value in sorted(counts.items(), key=lambda item: -item[1]))


def _faces_table(registry: font_registry.FontRegistry) -> Table:
    table = Table(title=f"{len(registry.faces)} faces", title_justify="left", header_style="bold")
    # The label must never be elided — it is the class name everything downstream keys on.
    table.add_column("label", overflow="fold", style="cyan")
    table.add_column("wght", justify="right")
    table.add_column("wdth", justify="right")
    table.add_column("flags")
    table.add_column("glyphs", justify="right")
    table.add_column("file", overflow="fold", style="dim")
    for face in registry.faces:
        flags = "".join(
            (
                "I" if face.italic else "·",
                "M" if face.monospace else "·",
                "V" if face.variable else "·",
            )
        )
        name = face.path.name
        if face.font_number:
            name = f"{name}[{face.font_number}]"
        table.add_row(
            face.label(registry.label_granularity),
            str(face.weight),
            str(face.width_class),
            flags,
            str(face.n_glyphs),
            name,
        )
    return table


def _rejected_table(registry: font_registry.FontRegistry) -> Table:
    table = Table(
        title=f"{len(registry.rejected)} rejected faces",
        title_justify="left",
        header_style="bold yellow",
    )
    table.add_column("face", overflow="fold")
    table.add_column("reason")
    table.add_column("detail", overflow="fold")
    table.add_column("file", overflow="fold", style="dim")
    for item in registry.rejected:
        table.add_row(item.face.face_id, item.reason, item.detail, item.face.path.name)
    return table


def _unreadable_table(registry: font_registry.FontRegistry) -> Table:
    table = Table(
        title=f"{len(registry.unreadable)} unreadable files",
        title_justify="left",
        header_style="bold red",
    )
    table.add_column("file", overflow="fold")
    table.add_column("error", overflow="fold")
    for item in registry.unreadable:
        table.add_row(str(item.path), item.error)
    return table


if __name__ == "__main__":
    app()
