"""The formatter that prints what a call site attached.

A translation pass runs for hours and the log is the only record of it, so a line
that says something failed and not why is close to worthless. Two of those cost
most of an afternoon during the first real bench run, which is why these exist.
"""

import logging

from conftest import FAKE_KEY
from pydocvi.logs import WithExtras, configure_logging


def record(message: str, **extras: object) -> logging.LogRecord:
    """A record the way ``log.warning(msg, extra=...)`` makes one."""
    made = logging.LogRecord("pydocvi.test", logging.WARNING, "f.py", 10, message, None, None)
    for name, value in extras.items():
        setattr(made, name, value)
    return made


class TestWithExtras:
    def test_an_extra_reaches_the_line(self) -> None:
        line = WithExtras().format(record("call failed", route="server1", reason="HTTP 502"))
        assert "call failed" in line
        assert "route=server1" in line
        assert "reason=HTTP 502" in line

    def test_a_line_with_nothing_attached_is_left_alone(self) -> None:
        assert WithExtras().format(record("tunnel up")) == "tunnel up"

    def test_the_extras_are_in_a_stable_order(self) -> None:
        """Sorted rather than insertion ordered, so two runs of the same failure
        produce two identical lines and a diff of two logs is readable."""
        line = WithExtras().format(record("call failed", route="a", attempt=2, batch="3f2a"))
        assert line.index("attempt=") < line.index("batch=") < line.index("route=")

    def test_a_key_that_finds_its_way_into_an_extra_is_redacted(self) -> None:
        """This is the point where an extra reaches somebody's terminal, and a
        base URL is exactly the kind of thing a call site attaches."""
        line = WithExtras().format(record("call failed", url=f"http://h/v1?key={FAKE_KEY}"))
        assert FAKE_KEY not in line
        assert "***" in line

    def test_a_field_every_record_has_is_not_printed_as_an_extra(self) -> None:
        """Worked out by making a record rather than by listing the fields, so a
        Python release that adds one does not start printing it."""
        line = WithExtras().format(record("tunnel up"))
        assert "lineno=" not in line
        assert "pathname=" not in line

    def test_configure_logging_installs_it(self) -> None:
        configure_logging()
        installed = logging.getLogger().handlers
        assert installed
        assert all(isinstance(handler.formatter, WithExtras) for handler in installed)
