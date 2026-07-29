"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fontaine.config import DEFAULT_CONFIG_PATH, StreamConfig, load_stream_config
from fontaine.fonts import registry as font_registry
from fontaine.fonts.coverage import CHARSET_PRESETS, resolve_charset

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
