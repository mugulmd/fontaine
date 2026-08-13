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

from fontaine.assets import fetch as asset_fetch
from fontaine.assets import manifest as asset_manifest
from fontaine.config import DEFAULT_CONFIG_PATH, Range, StreamConfig, load_stream_config
from fontaine.evaluate import prequential
from fontaine.fonts import registry as font_registry
from fontaine.fonts.coverage import CHARSET_PRESETS, resolve_charset
from fontaine.recognize import discovery
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
models_app = typer.Typer(help="Inspect the recognizers in models/.", no_args_is_help=True)
assets_app = typer.Typer(help="Fetch and verify the pinned assets.", no_args_is_help=True)
app.add_typer(fonts_app, name="fonts")
app.add_typer(models_app, name="models")
app.add_typer(assets_app, name="assets")

console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Stream config YAML.")]
FontDirOption = Annotated[
    Path | None, typer.Option("--font-dir", help="Override the config's font dir.")
]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", help="Surface fontTools' per-table warnings.")
]
ModelDirOption = Annotated[
    Path, typer.Option("--model-dir", help="Directory the recognizers are read from.")
]
ManifestOption = Annotated[Path, typer.Option("--manifest", help="Asset manifest YAML.")]


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
        return ArrivalProcess(registry.faces, settings.arrival, seed=settings.seed)
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
            exclude=fonts.exclude,
            include_variable=fonts.include_variable,
            verbose=verbose,
        )
    except NotADirectoryError as error:
        console.print(f"[red]{error}[/red]")
        console.print("Drop .ttf/.otf files in, or pass --font-dir.", style="dim")
        raise typer.Exit(code=1) from error


def _manifest(path: Path) -> asset_manifest.AssetManifest:
    try:
        return asset_manifest.load_manifest(path)
    except asset_manifest.ManifestError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error


def _selection(
    manifest: asset_manifest.AssetManifest, only: str | None
) -> tuple[asset_manifest.Asset, ...]:
    """The assets a ``--only`` glob picks out, refusing a pattern that matches nothing.

    Same reasoning as the arrival schedule's unmatched-pattern error: a typo that
    silently selects zero assets looks exactly like a fetch with nothing to do.
    """
    selected = manifest.matching(only)
    if not selected:
        console.print(
            f"[red]--only {only!r} matched none of the {len(manifest.assets)} assets[/red]"
        )
        raise typer.Exit(code=1)
    return selected


@assets_app.command("fetch")
def assets_fetch(
    manifest_path: ManifestOption = asset_manifest.DEFAULT_MANIFEST_PATH,
    only: Annotated[
        str | None, typer.Option("--only", help="Glob over paths or file names, e.g. '*.ttf'.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even assets that already verify.")
    ] = False,
) -> None:
    """Download every pinned asset and verify it against its checksum.

    Safe to re-run: an asset already matching its checksum is left alone, so this
    costs nothing on a warm asset directory. Nothing is installed until its digest
    matches, so an interrupted or misdirected fetch leaves the asset directory as
    it was rather than half-populated.
    """
    manifest = _manifest(manifest_path)
    selected = _selection(manifest, only)
    console.print(
        f"{len(selected)} asset(s) from [bold]{manifest_path}[/bold] — "
        f"[dim]{manifest.release}[/dim]"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("fetching", total=len(selected))
        report = asset_fetch.fetch_all(
            manifest,
            assets=selected,
            force=force,
            on_start=lambda asset: progress.update(task, description=asset.name),
            on_done=lambda _: progress.advance(task),
        )

    style = "yellow" if report.failed else "green"
    console.print(f"[{style}]{asset_fetch.summarize(report)}[/{style}]")

    if report.failed:
        console.print(_failures_table(report))
        # The two failures call for opposite fixes, so only the hint that applies is
        # printed: a mismatch is a manifest to re-pin, a 404 is a release to fill in.
        if any("mismatch" in failure.reason for failure in report.failed):
            console.print(
                "a checksum mismatch means the manifest and the release disagree about the "
                "bytes — if the change was intended, re-pin it with `fontaine assets hash`",
                style="dim",
            )
        else:
            console.print(
                f"nothing answered for those — check the release actually holds them, "
                f"under exactly these file names: {manifest.release}",
                style="dim",
            )
        raise typer.Exit(code=1)


@assets_app.command("status")
def assets_status(
    manifest_path: ManifestOption = asset_manifest.DEFAULT_MANIFEST_PATH,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="List every asset, not just the ones out of sync.")
    ] = False,
) -> None:
    """Check what is on disk against the manifest, without touching the network.

    Exits non-zero when the asset tree is not the pinned one, so it works as a
    check before a run whose numbers are meant to be comparable with someone else's.
    """
    manifest = _manifest(manifest_path)
    checks = asset_manifest.check_all(manifest)
    extra = asset_manifest.untracked(manifest)

    shown = checks if verbose else tuple(item for item in checks if not item.ok)
    if shown:
        console.print(_assets_table(shown, total=len(checks)))

    n_ok = sum(1 for item in checks if item.ok)
    console.print(
        f"[green]{n_ok}[/green]/{len(checks)} assets match [bold]{manifest_path}[/bold]"
        + ("" if verbose or not n_ok else " — pass --verbose to list them all")
    )

    # Outside the table, and one digest per line: this is the value to paste into
    # the manifest when the change was deliberate, and a column narrow enough to
    # wrap it mid-string would make it useless for exactly that.
    changed = [item for item in checks if item.state is asset_manifest.State.CHANGED]
    for item in changed:
        console.print(f"\n{item.asset.path} is not the pinned file. If that was intended:")
        console.print(f"    sha256: {item.actual}", style="cyan", highlight=False)
    if changed:
        console.print(
            "otherwise `fontaine assets fetch --force` puts the pinned bytes back", style="dim"
        )

    if extra:
        console.print(
            f"[yellow]{len(extra)} untracked file(s)[/yellow] in the asset directories: "
            + ", ".join(str(path) for path in extra[:6])
            + (" ..." if len(extra) > 6 else "")
        )
        console.print(
            "an unpinned font still enters the label space, so it changes the classes "
            "while every checksum here keeps verifying — pin it or remove it",
            style="dim",
        )

    unlicensed = [item.asset.path for item in checks if item.asset.license is None]
    if unlicensed:
        console.print(
            f"[yellow]{len(unlicensed)} asset(s) with no licence recorded[/yellow] — the release "
            f"redistributes these bytes, so the terms have to travel with them",
            style="yellow",
        )

    if n_ok != len(checks) or extra:
        raise typer.Exit(code=1)


@assets_app.command("hash")
def assets_hash(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files, or directories to scan for files not yet pinned."),
    ],
    manifest_path: ManifestOption = asset_manifest.DEFAULT_MANIFEST_PATH,
) -> None:
    """Print paste-ready manifest entries for files already on disk.

    This is how a new asset gets pinned: drop the file in, run this, paste the
    block into the manifest and fill in the licence and source. Given a directory,
    it emits an entry for every file the manifest does not already pin, which is
    the shape of adding several fonts at once.
    """
    manifest = _manifest(manifest_path)
    pinned = {asset.path: asset for asset in manifest.assets}
    root = Path.cwd()

    targets: list[Path] = []
    for given in paths:
        resolved = given.resolve()
        if not resolved.is_relative_to(root):
            console.print(f"[red]{given} is outside the repo, so it cannot be an asset path[/red]")
            raise typer.Exit(code=1)
        relative = resolved.relative_to(root)
        if resolved.is_dir():
            targets.extend(
                entry.relative_to(root)
                for entry in sorted(resolved.rglob("*"))
                if entry.is_file()
                and not entry.name.startswith(".")
                and entry.relative_to(root) not in pinned
            )
        elif resolved.is_file():
            targets.append(relative)
        else:
            console.print(f"[red]no such file: {given}[/red]")
            raise typer.Exit(code=1)

    if not targets:
        console.print("nothing to hash — every file given is already pinned", style="dim")
        return

    for path in targets:
        existing = pinned.get(path)
        if existing is not None:
            console.print(
                f"# {path} is already pinned as {existing.sha256[:12]}… — "
                f"replace just its sha256 line",
                style="dim",
            )
        # Printed without rich's markup parsing, which would eat any bracket in a path.
        console.print(asset_manifest.entry_yaml(path), markup=False, highlight=False)


def _assets_table(checks: tuple[asset_manifest.Check, ...], *, total: int) -> Table:
    """Per-asset state, and what the state means."""
    table = Table(
        title=f"{len(checks)} of {total} assets", title_justify="left", header_style="bold"
    )
    table.add_column("asset", overflow="fold", style="cyan")
    table.add_column("state")
    table.add_column("licence")
    table.add_column("detail", style="dim", overflow="fold")

    styles = {
        asset_manifest.State.OK: "green",
        asset_manifest.State.MISSING: "yellow",
        asset_manifest.State.CHANGED: "red",
    }
    for item in checks:
        table.add_row(
            str(item.asset.path),
            f"[{styles[item.state]}]{item.state.value}[/{styles[item.state]}]",
            item.asset.license or "[yellow]none[/yellow]",
            asset_fetch.describe_state(item.state),
        )
    return table


def _failures_table(report: asset_fetch.FetchReport) -> Table:
    """What went wrong per failed asset, since different failures call for different fixes."""
    table = Table(
        title=f"{len(report.failed)} assets could not be fetched",
        title_justify="left",
        header_style="bold red",
    )
    table.add_column("asset", overflow="fold")
    table.add_column("why", overflow="fold", style="dim")
    for failure in report.failed:
        table.add_row(str(failure.asset.path), failure.reason)
    return table


@models_app.command("list")
def models_list(model_dir: ModelDirOption = discovery.DEFAULT_MODEL_DIR) -> None:
    """List the recognizers found in the model directory."""
    try:
        found = discovery.discover(model_dir)
    except discovery.ModelError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    table = Table(title=f"{len(found)} models in {model_dir}", title_justify="left")
    table.add_column("name", style="cyan")
    table.add_column("class")
    table.add_column("what it is", style="dim", overflow="fold")
    for name, model in sorted(found.items()):
        table.add_row(name, model.__qualname__, discovery.describe(model))
    console.print(table)

    if not found:
        console.print(
            f"drop a file in [bold]{model_dir}[/bold] subclassing "
            "fontaine.contracts.Recognizer — see models/baseline.py",
            style="dim",
        )


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

    console.print(f"\n[green]{len(registry.faces)}[/green] faces kept — one label each")
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
        json_out.write_text(json.dumps(registry.model_dump(mode="json"), indent=2))
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
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Render a batch of crops to a contact sheet, to check the renders by eye.

    Faces are taken in registry order and cycled, so a count of at least the size
    of the label space shows every font at least once. The ``arrival`` config is
    deliberately ignored: this is a check that every font renders, so a font made
    rare or scheduled to arrive late still has to show up here.
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
        cells.append((crop.image, f"{face.face_id}  {crop.metadata['cap_height_px']:.0f}px"))

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
        f"[green]{report.n_faces}[/green] faces"
    )
    if report.n_skipped:
        console.print(f"[yellow]{report.n_skipped} items skipped[/yellow]")
    console.print(_schedule_table(generator.arrivals, report.n_items))
    console.print(_popularity_line(generator.stats.face_counts, report.n_items), style="dim")


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
    console.print(_popularity_line(process.stats.face_counts, count), style="dim")

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
    limit: Annotated[
        int | None, typer.Option("--limit", help="Stop after this many items of a saved stream.")
    ] = None,
    model_name: Annotated[
        str, typer.Option("--model", "-m", help="Which recognizer from the model dir to score.")
    ] = "baseline",
    model_dir: ModelDirOption = discovery.DEFAULT_MODEL_DIR,
    classes: Annotated[
        bool, typer.Option("--classes/--no-classes", help="Print the per-font table.")
    ] = True,
    font_dir: FontDirOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Score an online recognizer over a stream, predicting before it is told.

    Reads a saved stream with ``--stream``, or generates one on the fly from the
    config. Both paths hand the model the same items, so the numbers are comparable.

    ``--model`` names any recognizer under ``models/``; ``fontaine models list``
    prints what is there.
    """
    # Built before the stream, so a typo in --model fails immediately rather than
    # after minutes of generation.
    try:
        model = discovery.load(model_name, model_dir)
    except discovery.ModelError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1) from error

    if stream is not None:
        try:
            manifest = store_reader.read_manifest(stream)
        except store_reader.StreamNotFound as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from error
        total = manifest["n_items"] if limit is None else min(limit, manifest["n_items"])
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
        samples = generator.take(count)
        total = count
        source = f"generating live from [bold]{config}[/bold]"

    console.print(f"{source} — {total:,} items — model [cyan]{model.name}[/cyan]")

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
            on_item=lambda _: progress.advance(task),
        )

    console.print(_headline_table(result))
    if classes:
        console.print(_classes_table(result))
    worst = result.overall.worst_confusions(5)
    if worst:
        summary = "   ".join(
            f"{true.split(':')[0]}→{predicted.split(':')[0]} {n}" for (true, predicted), n in worst
        )
        console.print(f"most confused: {summary}", style="dim")


def _headline_table(result: prequential.Result) -> Table:
    """The lifetime and recent scores side by side, beside what learning nothing gets.

    Both columns come from the same three formulas over two confusion matrices. Read
    across a row rather than down: a model still climbing shows a recent column well
    above its overall one, which is exactly what separates a slow learner from a
    model that has converged somewhere mediocre.
    """
    table = Table(title="prequential score", title_justify="left", header_style="bold")
    table.add_column("measure")
    table.add_column("overall", justify="right")
    table.add_column(f"last {prequential.WINDOW:,}", justify="right")
    table.add_column("meaning", style="dim", overflow="fold")

    overall, recent = result.overall, result.recent
    lift = overall.accuracy / result.majority_accuracy if result.majority_accuracy else float("inf")
    rows = [
        (
            "accuracy",
            f"{overall.accuracy:.1%}",
            f"{recent.accuracy:.1%}",
            "predicted before being told",
        ),
        (
            "balanced accuracy",
            f"{overall.balanced_accuracy:.1%}",
            f"{recent.balanced_accuracy:.1%}",
            "averaged over fonts, not items",
        ),
        (
            "macro F1",
            f"{overall.macro_f1:.1%}",
            f"{recent.macro_f1:.1%}",
            "penalises ignoring the rare fonts",
        ),
        (
            "majority baseline",
            f"{result.majority_accuracy:.1%}",
            "",
            "always answer the commonest font",
        ),
        (
            "chance baseline",
            f"{result.chance_accuracy:.1%}",
            "",
            "guess uniformly among fonts seen",
        ),
        ("lift over majority", f"{lift:.1f}x", "", "below 1 means the model learned nothing"),
    ]
    for name, whole, window, meaning in rows:
        table.add_row(name, whole, window, meaning)
    return table


def _classes_table(result: prequential.Result) -> Table:
    """Per-font recall and precision, lifetime beside the recent window.

    Recall says how much of a font the model catches, precision how much of what it
    answers is right — a model that guesses one font for everything has high recall
    there and precision on the floor, and only the pair shows it.
    """
    table = Table(title="per font", title_justify="left", header_style="bold")
    # Short headers so the label, which is the class name, never has to wrap.
    table.add_column("font", overflow="fold", style="cyan")
    table.add_column("items", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("prec", justify="right")
    table.add_column("recent", justify="right")

    overall, recent = result.overall, result.recent
    for label in overall.labels:
        # A font can be absent from the rolling window entirely — it stopped
        # arriving, or has not arrived yet — and has no recent recall to show.
        in_window = recent.support(label) > 0
        table.add_row(
            label,
            f"{overall.support(label):,}",
            f"{overall.recall(label):.0%}",
            f"{overall.precision(label):.0%}",
            f"{recent.recall(label):.0%}" if in_window else "[dim]-[/dim]",
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
        table.add_row(
            face.face_id,
            str(face.weight),
            str(face.width_class),
            flags,
            str(face.n_glyphs),
            face.path.name,
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
