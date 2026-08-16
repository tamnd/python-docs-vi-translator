"""The ``pydocvi`` command line.

This module wires arguments to library functions and does nothing else. No other
module in the package imports it, and every command here should stay short
enough to read in one screen.
"""

import sys
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from pydocvi import __version__, config, sync
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
