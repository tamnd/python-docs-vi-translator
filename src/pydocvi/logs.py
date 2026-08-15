"""Logging setup.

A translation pass runs for hours, and the log is the only record of what
happened during it. Every fleet log line carries structured extras such as the
batch id, the route and the attempt number, which is what makes a trace
recoverable after the fact.
"""

import logging

from rich.logging import RichHandler


def configure_logging(*, verbose: bool = False) -> None:
    """Install a rich handler on the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(
        RichHandler(
            rich_tracebacks=True,
            show_path=verbose,
            omit_repeated_times=False,
            log_time_format="%H:%M:%S",
        )
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
