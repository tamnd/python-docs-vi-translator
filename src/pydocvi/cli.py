"""The ``pydocvi`` command line.

This module wires arguments to library functions and does nothing else. No other
module in the package imports it, and every command here should stay short
enough to read in one screen.
"""

import sys
from typing import NoReturn

import typer

from pydocvi import __version__
from pydocvi.logs import configure_logging

app = typer.Typer(
    name="pydocvi",
    help="Translate the CPython documentation into Vietnamese gettext catalogs.",
    no_args_is_help=True,
    add_completion=False,
)


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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Raise the log level."),
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


def main() -> NoReturn:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(130)
    raise AssertionError("unreachable: typer exits via SystemExit")
