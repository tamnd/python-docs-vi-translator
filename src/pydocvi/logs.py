"""Logging setup.

A translation pass runs for hours, and the log is the only record of what
happened during it. Every fleet log line carries structured extras such as the
batch id, the route and the attempt number, which is what makes a trace
recoverable after the fact.
"""

import logging

from rich.logging import RichHandler

from pydocvi.client import redact

#: Every attribute a bare record already has, worked out by making one rather
#: than by listing them, so a Python release that adds a field does not turn that
#: field into something this module prints as though a call site had asked for it.
_BUILT_IN = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "asctime",
    "message",
    "taskName",
}


class WithExtras(logging.Formatter):
    """A formatter that prints the ``extra`` the call site bothered to attach.

    Without this the extras are attached and then dropped, which is how a
    half-hour bench run produced the line "bench call failed" six times and never
    once said why, and how "call failed, retrying on the same route" managed not
    to name the route. Both of those had the reason in hand and threw it away.

    Redacted, because an extra can hold a URL and this is the point where it
    reaches somebody's terminal.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = {name: value for name, value in vars(record).items() if name not in _BUILT_IN}
        if not extras:
            return message
        said = " ".join(f"{name}={value}" for name, value in sorted(extras.items()))
        return redact(f"{message}  {said}")


def configure_logging(*, verbose: bool = False) -> None:
    """Install a rich handler on the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = RichHandler(
        rich_tracebacks=True,
        show_path=verbose,
        omit_repeated_times=False,
        log_time_format="%H:%M:%S",
    )
    handler.setFormatter(WithExtras())
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
