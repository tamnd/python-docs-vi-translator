"""The ``pydocvi`` command line.

This module wires arguments to library functions and does nothing else. No other
module in the package imports it, and every command here should stay short
enough to read in one screen.
"""

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from pydocvi import __version__, batch, classify, config, sync
from pydocvi.catalog import Catalog
from pydocvi.logs import configure_logging
from pydocvi.memory import Memory

app = typer.Typer(
    name="pydocvi",
    help="Translate the CPython documentation into Vietnamese gettext catalogs.",
    no_args_is_help=True,
    add_completion=False,
)
tm_app = typer.Typer(help="Inspect the translation memory.", no_args_is_help=True)
app.add_typer(tm_app, name="tm")

console = Console()


class ExitCode:
    """Stable exit codes.

    The runbook and both repositories' CI depend on telling a failed check apart
    from a usage error, and both apart from an unreachable fleet.
    """

    OK = 0
    CHECK_FAILED = 1
    USAGE = 2
    FLEET_UNREACHABLE = 3


@app.callback()
def root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Raise the log level.")] = False,
) -> None:
    """Keep the app a command group.

    Typer collapses a single-command app into a bare command, which would make
    ``pydocvi version`` a usage error until the second command lands.
    """
    configure_logging(verbose=verbose)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command(name="sync")
def sync_command(
    human: Annotated[
        bool, typer.Option("--human", help="Load human translations into the memory.")
    ] = False,
    branch: Annotated[str, typer.Option(help="Upstream branch to record.")] = sync.DEFAULT_BRANCH,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would change and write nothing.")
    ] = False,
) -> None:
    """Measure upstream, write the pin, and optionally load human translations."""
    where = config.paths()
    if not where.upstream.exists():
        console.print(f"[red]no upstream checkout at {where.upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    catalogs = sync.read_corpus(where.upstream)
    pin = sync.pin(
        where.upstream,
        repo="tamnd/python-docs-vi",
        branch=branch,
        commit=sync.head_commit(where.upstream),
    )

    table = Table(title=f"{pin.repo} @ {pin.branch} {pin.commit[:8]}", box=None)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("files", f"{pin.files:,}")
    table.add_row("entries", f"{pin.entries:,}")
    table.add_row("English words", f"{pin.words:,}")
    table.add_row("msgid characters", f"{pin.characters:,}")
    table.add_row("already translated", f"{pin.translated:,}")
    console.print(table)

    memory = Memory.load(where.memory)
    changes = sync.diff(memory, catalogs)
    console.print(
        f"upstream strings not in the memory: {len(changes.added):,}   "
        f"orphaned segments: {len(changes.orphaned):,}"
    )

    if human:
        stored = sync.load_human(memory, catalogs)
        console.print(f"human segments stored: {stored:,}")

    if dry_run:
        console.print("[yellow]dry run, nothing written[/yellow]")
        return

    where.upstream_pin.parent.mkdir(parents=True, exist_ok=True)
    where.upstream_pin.write_text(pin.as_yaml(), encoding="utf-8")
    where.reports.mkdir(parents=True, exist_ok=True)
    (where.reports / "sync.md").write_text(changes.as_markdown(), encoding="utf-8")
    memory.save(where.memory)
    console.print(f"wrote {where.upstream_pin} and {where.memory}")


@app.command(name="classify")
def classify_command(
    report: Annotated[
        bool, typer.Option("--report", help="Write the breakdown to reports/classify.md.")
    ] = False,
) -> None:
    """Count what the corpus is made of, and what a model will never see."""
    where = config.paths()
    catalogs = _corpus(where.upstream)
    counts = classify.counts([entry.msgid for cat in catalogs for entry in cat])

    table = Table(title="entries by kind", box=None)
    table.add_column("kind")
    table.add_column("entries", justify="right")
    table.add_column("share", justify="right")
    for kind, count in _kinds(counts):
        table.add_row(kind, f"{count:,}", f"{count / counts.total:.1%}")
    table.add_row("[bold]total", f"[bold]{counts.total:,}", "")
    console.print(table)
    console.print(f"never sent to a model: {counts.passthrough:,}")

    if report:
        where.reports.mkdir(parents=True, exist_ok=True)
        target = where.reports / "classify.md"
        target.write_text(_classify_markdown(counts), encoding="utf-8")
        console.print(f"wrote {target}")


def _kinds(counts: classify.Counts) -> list[tuple[str, int]]:
    return [
        ("prose", counts.prose),
        ("no-op", counts.noop),
        ("doctest", counts.doctest),
        ("literal block", counts.literal_block),
        ("version marker", counts.version_marker),
    ]


def _classify_markdown(counts: classify.Counts) -> str:
    lines = [
        "# Classification",
        "",
        "| kind | entries | share |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {kind} | {count} | {count / counts.total:.1%} |" for kind, count in _kinds(counts)
    ]
    lines += [
        f"| **total** | **{counts.total}** | |",
        "",
        f"{counts.passthrough} entries are copied through without a model call.",
        "",
    ]
    return "\n".join(lines)


@app.command(name="batch")
def batch_command(
    stats: Annotated[bool, typer.Option("--stats", help="Print the batching numbers.")] = False,
    top: Annotated[int, typer.Option(help="How many files to list by batch count.")] = 10,
) -> None:
    """Cut the corpus into batches and report what came out."""
    where = config.paths()
    batches = batch.build(_corpus(where.upstream), root=where.upstream)
    measured = batch.stats(batches)

    console.print(f"{measured.batches:,} batches over {measured.files:,} files")
    if not stats:
        return

    table = Table(box=None)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("entries batched", f"{measured.entries:,}")
    table.add_row("msgid characters", f"{measured.characters:,}")
    table.add_row("protected spans", f"{measured.spans:,}")
    table.add_row("entries per batch", f"{measured.entries_per_batch:.1f}")
    table.add_row("characters per batch", f"{measured.characters_per_batch:,.0f}")
    table.add_row("batches over a cap alone", f"{measured.oversized:,}")
    console.print(table)

    heaviest = sorted(batch.by_file(batches).items(), key=lambda pair: -pair[1])[:top]
    files = Table(title=f"heaviest {top} files", box=None)
    files.add_column("file")
    files.add_column("batches", justify="right")
    files.add_column("share", justify="right")
    for path, count in heaviest:
        files.add_row(path, f"{count:,}", f"{count / measured.batches:.1%}")
    console.print(files)


def _corpus(upstream: Path) -> list[Catalog]:
    """Read the upstream corpus, or exit saying it is not there."""
    if not upstream.exists():
        console.print(f"[red]no upstream checkout at {upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    return sync.read_corpus(upstream)


@tm_app.command("stats")
def tm_stats() -> None:
    """Counts by provenance."""
    memory = Memory.load(config.paths().memory)
    table = Table(box=None)
    table.add_column("source")
    table.add_column("segments", justify="right")
    for source, count in memory.counts().items():
        table.add_row(source, f"{count:,}")
    table.add_row("[bold]total", f"[bold]{len(memory):,}")
    console.print(table)


@tm_app.command("show")
def tm_show(segment: str) -> None:
    """Print one segment by its id."""
    found = Memory.load(config.paths().memory).get(segment)
    if found is None:
        console.print(f"[red]no segment {segment}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    console.print(found)


def main() -> NoReturn:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(130)
    raise AssertionError("unreachable: typer exits via SystemExit")
