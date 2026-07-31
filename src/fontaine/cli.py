"""Command line entry points."""

from __future__ import annotations

import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from fontaine.config import DEFAULT_CONFIG_PATH, Range, StreamConfig, load_stream_config
from fontaine.evaluate import prequential
from fontaine.fonts import registry as font_registry
from fontaine.fonts.coverage import CHARSET_PRESETS, resolve_charset
from fontaine.recognize import features as feature_module
from fontaine.recognize import models as model_module
from fontaine.render import background as background_module
from fontaine.render.textbox import CropRenderer, RenderError
from fontaine.rng import item_rng
from fontaine.store import reader as store_reader
from fontaine.store.writer import write_stream
from fontaine.stream import arrival as arrival_module
from fontaine.stream.arrival import ArrivalProcess, ScheduleError
from fontaine.stream.generator import StreamGenerator
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


def _parse_range(value: str, option: str) -> Range:
    """Parse a ``LO:HI`` command-line override into a :class:`Range`."""
    try:
        low, high = (float(part) for part in value.split(":", 1))
        return Range(low, high)
    except ValueError as error:
        console.print(f"[red]{option} expects LO:HI, e.g. 18:64 — got {value!r}[/red]")
        raise typer.Exit(code=1) from error


def _arrivals(settings: StreamConfig, registry: font_registry.FontRegistry) -> ArrivalProcess:
    try:
        return ArrivalProcess(
            registry.faces,
            settings.arrival,
            seed=settings.seed,
            label_granularity=registry.label_granularity,
        )
    except ScheduleError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error


def _generator(settings: StreamConfig, registry: font_registry.FontRegistry) -> StreamGenerator:
    try:
        return StreamGenerator(settings, registry)
    except ValueError as error:  # ScheduleError included: an unusable config
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error


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
    cap_height: Annotated[
        str | None,
        typer.Option(
            "--cap-height",
            metavar="LO:HI",
            help="Override the config's cap height range, e.g. 18:64. The quickest "
            "knob for text that comes out too small.",
        ),
    ] = None,
    from_stream: Annotated[
        bool,
        typer.Option(
            "--stream/--all-faces",
            help="Draw items from the arrival process instead of cycling every face.",
        ),
    ] = False,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Render a batch of crops to a contact sheet, to check the renders by eye.

    By default faces are taken in registry order and cycled, so a large enough
    count shows every font in the label space at least once. ``--stream`` instead
    draws the first items of the actual stream, which shows what the recognizer
    really sees — dominated by whichever fonts arrived early.
    """
    settings = _load(config)
    if font_dir is not None:
        settings.fonts.font_dir = font_dir
    if seed is not None:
        settings.seed = seed
    if cap_height is not None:
        settings.render.typography.cap_height_px = _parse_range(cap_height, "--cap-height")

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
        console.print("no PNGs found, falling back to synthetic canvases", style="yellow")

    cells: list[tuple[Image.Image, str]] = []
    stats: Counter[str] = Counter()
    failures: list[str] = []

    if from_stream:
        generator = _generator(settings, registry)
        for sample in generator.take(count):
            stats[sample.metadata["background"]] += 1
            stats[f"kind:{sample.metadata['text_kind']}"] += 1
            marker = " NEW" if sample.metadata["label_first_seen"] else ""
            cells.append(
                (sample.image, f"{sample.label}  {sample.metadata['cap_height_px']:.0f}px{marker}")
            )
        failures = [item.error for item in generator.skipped]
    else:
        renderer = CropRenderer(settings.render)
        for index in range(count):
            face = registry.faces[index % len(registry.faces)]
            try:
                crop = renderer.render(face, item_rng(settings.seed, index))
            except RenderError as error:
                failures.append(str(error))
                continue
            stats[crop.metadata["background"]] += 1
            stats[f"kind:{crop.metadata['text_kind']}"] += 1
            label = face.label(registry.label_granularity)
            cells.append((crop.image, f"{label}  {crop.metadata['cap_height_px']:.0f}px"))

    if not cells:
        console.print("[red]every render failed[/red]")
        for failure in failures[:5]:
            console.print(f"  {failure}", style="dim")
        raise typer.Exit(code=1)

    sheet = contact_sheet(cells, columns=columns, cell_height=cell_height)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    console.print(f"{len(cells)} crops → [bold]{out}[/bold] ({sheet.width}×{sheet.height})")
    backgrounds = {key: value for key, value in stats.items() if not key.startswith("kind:")}
    kinds = {
        key.removeprefix("kind:"): value for key, value in stats.items() if key.startswith("kind:")
    }
    console.print(f"backgrounds: {_summarize(backgrounds)}", style="dim")
    console.print(f"content kinds: {_summarize(kinds)}", style="dim")
    if failures:
        console.print(f"[yellow]{len(failures)} renders failed[/yellow]: {failures[0]}")


@app.command("generate")
def generate(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    count: Annotated[int, typer.Option("--count", "-n", help="How many items to write.")] = 10_000,
    out: Annotated[Path, typer.Option("--out", "-o", help="Stream directory to create.")] = Path(
        "data/streams/v1"
    ),
    seed: Annotated[int | None, typer.Option("--seed", help="Override the config's seed.")] = None,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Materialize a stream to disk: crops, annotations and a manifest."""
    settings = _load(config)
    if font_dir is not None:
        settings.fonts.font_dir = font_dir
    if seed is not None:
        settings.seed = seed

    registry = _scan(settings, verbose=verbose)
    if not registry.faces:
        console.print("[red]empty label space — nothing to generate[/red]")
        raise typer.Exit(code=1)

    generator = _generator(settings, registry)
    console.print(f"seed {settings.seed} — {arrival_module.describe(generator.arrivals.schedule)}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("generating", total=count)
        report = write_stream(generator, count, out, on_item=lambda _: progress.advance(task))

    console.print(
        f"[green]{report.n_items}[/green] items → [bold]{report.directory}[/bold], "
        f"[green]{report.n_labels}[/green] labels from "
        f"[green]{report.n_faces}[/green] faces"
    )
    if report.n_skipped:
        console.print(f"[yellow]{report.n_skipped} items skipped[/yellow]")
    console.print(_schedule_table(generator.arrivals, report.n_items))
    console.print(_popularity_line(generator.stats.label_counts, report.n_items), style="dim")


@app.command("arrival")
def arrival(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    count: Annotated[
        int, typer.Option("--count", "-n", help="How many steps to simulate.")
    ] = 50_000,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the config's seed.")] = None,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Simulate the arrival process without rendering anything.

    Nothing is drawn, so this runs in a fraction of a second — the cheap way to
    check that the weights and schedules produce the stream you intended before
    spending minutes generating images.
    """
    settings = _load(config)
    if font_dir is not None:
        settings.fonts.font_dir = font_dir
    if seed is not None:
        settings.seed = seed

    registry = _scan(settings, verbose=verbose)
    if not registry.faces:
        console.print("[red]empty label space — nothing to simulate[/red]")
        raise typer.Exit(code=1)

    process = _arrivals(settings, registry)
    process.take(count)

    console.print(f"{count:,} steps — {arrival_module.describe(process.schedule)}")
    console.print(_schedule_table(process, count))
    console.print(_popularity_line(process.stats.label_counts, count), style="dim")

    missing = [
        plan.face_id
        for plan in process.schedule
        if plan.weight > 0 and plan.face_id not in process.stats.face_counts
    ]
    if missing:
        console.print(
            f"[yellow]{len(missing)} font(s) never appeared[/yellow] despite a non-zero "
            f"weight: {', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''} — "
            f"lengthen the stream, raise their weight, or check their start",
            style="yellow",
        )


@app.command("recognize")
def recognize(
    stream: Annotated[
        Path | None,
        typer.Option("--stream", "-s", help="A saved stream directory to replay."),
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    count: Annotated[
        int, typer.Option("--count", "-n", help="Items to evaluate when generating live.")
    ] = 5_000,
    model_name: Annotated[
        str, typer.Option("--model", "-m", help=f"One of: {', '.join(model_module.MODELS)}.")
    ] = model_module.DEFAULT_MODEL,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Stop after this many items of a saved stream.")
    ] = None,
    classes: Annotated[
        bool, typer.Option("--classes/--no-classes", help="Print the per-font table.")
    ] = True,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Score an online recognizer over a stream, predicting before it is told.

    Reads a saved stream with ``--stream``, or generates one on the fly from the
    config. Both paths hand the model the same items, so the numbers are comparable.
    """
    if model_name not in model_module.MODELS:
        console.print(f"[red]unknown model {model_name!r}[/red]")
        for name, description in model_module.MODELS.items():
            console.print(f"  [cyan]{name}[/cyan] — {description}", style="dim")
        raise typer.Exit(code=1)

    schedule: dict[str, int] | None = None
    if stream is not None:
        try:
            manifest = store_reader.read_manifest(stream)
        except store_reader.StreamNotFound as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from error
        total = manifest["n_items"] if limit is None else min(limit, manifest["n_items"])
        schedule = prequential.schedule_from_manifest(
            manifest, manifest.get("label_granularity", "face")
        )
        samples = store_reader.read_stream(stream)
        if limit is not None:
            samples = islice(samples, limit)
        source = f"replaying [bold]{stream}[/bold]"
    else:
        settings = _load(config)
        if font_dir is not None:
            settings.fonts.font_dir = font_dir
        registry = _scan(settings, verbose=verbose)
        generator = _generator(settings, registry)
        schedule = {
            plan.face_id: plan.start
            for plan in generator.arrivals.schedule
            if registry.label_granularity == "face"
        }
        samples = generator.take(count)
        total = count
        source = f"generating live from [bold]{config}[/bold]"

    console.print(f"{source} — model [cyan]{model_name}[/cyan], {total:,} items")
    model = model_module.build(model_name)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("evaluating", total=total)
        result = prequential.run(
            samples,
            model,
            feature_module.describe,
            schedule=schedule,
            on_item=lambda _: progress.advance(task),
        )

    console.print(_headline_table(result))
    if classes:
        console.print(_classes_table(result))
    if result.confusions:
        worst = "   ".join(
            f"{true.split(':')[0]}→{predicted.split(':')[0]} {n}"
            for (true, predicted), n in result.worst_confusions(5)
        )
        console.print(f"most confused: {worst}", style="dim")


def _headline_table(result: prequential.Result) -> Table:
    """Accuracy beside what a model that learned nothing would have scored."""
    table = Table(title="prequential score", title_justify="left", header_style="bold")
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_column("meaning", style="dim", overflow="fold")

    lift = result.accuracy / result.majority_accuracy if result.majority_accuracy else float("inf")
    rows = [
        (
            "accuracy",
            f"{result.accuracy:.1%}",
            "predicted before being told, over the whole stream",
        ),
        ("recent accuracy", f"{result.rolling_accuracy:.1%}", f"last {prequential.WINDOW} items"),
        ("balanced accuracy", f"{result.balanced_accuracy:.1%}", "averaged over fonts, not items"),
        ("macro F1", f"{result.macro_f1:.1%}", "penalises ignoring the rare fonts"),
        (
            "majority baseline",
            f"{result.majority_accuracy:.1%}",
            "always answer the commonest font",
        ),
        ("chance baseline", f"{result.chance_accuracy:.1%}", "guess uniformly among fonts seen"),
        ("lift over majority", f"{lift:.1f}x", "below 1 means the model learned nothing"),
        ("abstentions", f"{result.abstentions:,}", "no answer yet, before any label was seen"),
    ]
    for name, value, meaning in rows:
        table.add_row(name, value, meaning)
    return table


def _classes_table(result: prequential.Result) -> Table:
    """Per-font accuracy and how long each took to be recognised."""
    table = Table(title="per font", title_justify="left", header_style="bold")
    # Short headers so the label, which is the class name, never has to wrap.
    table.add_column("font", overflow="fold", style="cyan")
    table.add_column("items", justify="right")
    table.add_column("acc", justify="right")
    table.add_column("sched", justify="right")
    table.add_column("seen", justify="right")
    table.add_column("right", justify="right")
    table.add_column("lag", justify="right")

    for report in sorted(result.classes.values(), key=lambda item: -item.seen):
        lag = report.discovery_lag
        table.add_row(
            report.label,
            f"{report.seen:,}",
            f"{report.accuracy:.0%}",
            f"{report.scheduled_start:,}" if report.scheduled_start is not None else "-",
            f"{report.first_seen:,}" if report.first_seen is not None else "-",
            f"{report.first_correct:,}" if report.first_correct is not None else "[dim]never[/dim]",
            f"{lag:,}" if lag is not None else "[dim]-[/dim]",
        )
    return table


def _schedule_table(process: ArrivalProcess, n_items: int) -> Table:
    """What was asked for per font, beside what the run actually produced."""
    table = Table(title="arrival schedule", title_justify="left", header_style="bold")
    table.add_column("font", overflow="fold", style="cyan")
    table.add_column("weight", justify="right")
    table.add_column("window", justify="right")
    table.add_column("share", justify="right")
    table.add_column("items", justify="right")
    table.add_column("first seen", justify="right")

    counts = process.stats.face_counts
    for plan in sorted(process.schedule, key=lambda item: (-item.weight, item.face_id)):
        window = "all" if plan.start == 0 and plan.stop is None else f"{plan.start:,}-"
        if plan.stop is not None:
            window = f"{plan.start:,}-{plan.stop:,}"
        seen = process.stats.face_first_seen.get(plan.face_id)
        count = counts.get(plan.face_id, 0)
        table.add_row(
            plan.face_id,
            f"{plan.weight:g}" if plan.weight else "[dim]0[/dim]",
            window,
            f"{count / n_items:.1%}" if n_items else "-",
            f"{count:,}",
            f"{seen:,}" if seen is not None else "[dim]never[/dim]",
        )
    return table


def _popularity_line(counts: dict[str, int], n_items: int) -> str:
    if not counts or not n_items:
        return "no items"
    ranked = sorted(counts.values(), reverse=True)
    top = sum(ranked[:5]) / n_items
    singletons = sum(1 for value in ranked if value == 1)
    return (
        f"popularity: top 5 labels hold {top:.0%} of items, "
        f"{singletons} label(s) appeared exactly once"
    )


def _summarize(counts: dict[str, int]) -> str:
    return "  ".join(
        f"{key} {value}" for key, value in sorted(counts.items(), key=lambda item: -item[1])
    )


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
