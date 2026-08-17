"""Shared fixtures.

Nothing here touches the network or the fleet. Fixtures that need a real corpus
are guarded by the ``corpus`` marker and skip when the checkout is absent.

The fakes below are the reason the suite runs in seconds rather than sleeping
through a doubling backoff and a five-minute cooldown. A fake clock, a fake
command runner and a fake completions client cover every path that would
otherwise need a live tunnel.
"""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pydocvi.audit.model import Body, Corpus, Finding
from pydocvi.catalog import Catalog, Entry
from pydocvi.client import Answer, Usage
from pydocvi.fleet import Ran
from pydocvi.memory import Segment
from pydocvi.routes import Route

UPSTREAM = Path.home() / "github" / "tamnd" / "python-docs-vi"

#: A key of the right shape for the tests that check keys are handled.
#: Assembled at import rather than written out, because ``make secrets`` refuses
#: to let any tracked file contain a key-shaped string and that rule is worth
#: more without an allowlist in it.
FAKE_KEY = "sk-" + "not-a-real-key-0123456789"


@pytest.fixture(autouse=True)
def _never_a_real_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every path at ``tmp_path`` before any test runs.

    Autouse, and belt and braces on top of the fixtures that already do this,
    because the cost of getting it wrong is asymmetric. A test that forgets to
    override ``PYDOCVI_CONTENT`` does not fail: it passes, having written a pin
    or a glossary into a real checkout, and the next person finds it in ``git
    status`` days later without knowing what put it there.
    """
    for name, path in (
        ("UPSTREAM", tmp_path / "upstream"),
        ("CONTENT", tmp_path / "content"),
        ("WORK", tmp_path / "work"),
    ):
        monkeypatch.setenv(f"PYDOCVI_{name}", str(path))


@pytest.fixture(scope="session")
def upstream() -> Path:
    """A local checkout of the upstream Transifex mirror."""
    if not (UPSTREAM / "bugs.po").exists():
        pytest.skip(f"no upstream checkout at {UPSTREAM}")
    return UPSTREAM


@pytest.fixture
def data_dir() -> Path:
    """Hand-written fixtures. None of them is longer than forty lines."""
    return Path(__file__).parent / "data"


class FakeClock:
    """Time that moves only when something waits for it.

    A cooldown doubling from five minutes to an hour takes two hours of real
    sleeping to test properly, which means it would not be tested properly.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.time = start
        self.slept: list[float] = []

    def now(self) -> float:
        return self.time

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds

    def advance(self, seconds: float) -> None:
        self.time += seconds


class FakeRunner:
    """A command runner with canned answers, keyed on a substring.

    Keyed on a substring rather than the whole command because the assertions
    that matter are about which program was invoked and with which flag, not
    about argument order.
    """

    def __init__(self, answers: dict[str, Ran] | None = None) -> None:
        self.answers = answers or {}
        self.commands: list[list[str]] = []

    def run(self, command: Sequence[str], *, timeout: float = 30.0) -> Ran:
        self.commands.append(list(command))
        joined = " ".join(command)
        for key, answer in self.answers.items():
            if key in joined:
                return answer
        return Ran(code=0)

    def ran(self, needle: str) -> bool:
        return any(needle in " ".join(command) for command in self.commands)


class FakeClient:
    """Completions from a list, or from a function of the prompt."""

    def __init__(
        self,
        answers: Sequence[str | BaseException] | Callable[[str], str | BaseException] = (),
        *,
        seconds: float = 120.0,
        served: str = "",
        delay: float = 0.0,
    ) -> None:
        self.answers = answers
        self.seconds = seconds
        self.served = served
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        #: The system message of every call, kept apart from ``calls`` so that
        #: the tests written before there was one still read the same way.
        self.systems: list[str | None] = []

    async def complete(self, route: Route, prompt: str, *, system: str | None = None) -> Answer:
        self.calls.append((route.name, prompt))
        self.systems.append(system)
        if self.delay:
            await asyncio.sleep(self.delay)
        if callable(self.answers):
            reply = self.answers(prompt)
        elif self.answers:
            reply = self.answers[(len(self.calls) - 1) % len(self.answers)]
        else:
            reply = "1 Một."
        if isinstance(reply, BaseException):
            raise reply
        return Answer(
            text=reply,
            route=route.name,
            model=route.model,
            seconds=self.seconds,
            usage=Usage(prompt_tokens=100, completion_tokens=50),
            served=self.served,
        )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


def make_route(name: str = "server3", **overrides: object) -> Route:
    """A route with the shape of a real one and none of its addresses."""
    values: dict[str, object] = {
        "name": name,
        "base_url": f"http://127.0.0.1:{overrides.pop('port', 8103)}/v1",
        "model": "gpt-5",
        "host": name,
        "local_port": 8103,
        "concurrency": 1,
        "timeout": 600.0,
    }
    values.update(overrides)
    return Route(**values)  # type: ignore[arg-type]


@pytest.fixture
def route() -> Route:
    return make_route()


def entry(msgid: str, msgstr: str = "", **overrides: object) -> Entry:
    """One entry, fuzzy by default because everything this tool writes is.

    Fuzzy by default so that a test about placeholders does not have to think
    about ``S04``, which is a different check with its own tests.
    """
    values: dict[str, object] = {"msgid": msgid, "msgstr": msgstr, "flags": ("fuzzy",)}
    values.update(overrides)
    return Entry(**values)  # type: ignore[arg-type]


def catalog_of(*entries: Entry, name: str = "library/one.po", root: Path | None = None) -> Catalog:
    """A catalog with a plausible header and the entries handed in."""
    header = Entry(msgid="", msgstr="Project-Id-Version: Python 3.15\nLanguage: vi\n")
    return Catalog(
        path=(root or Path("/corpus")) / name,
        header=header,
        entries=tuple(entries),
    )


def corpus_of(*catalogs: Catalog, root: Path | None = None, **overrides: object) -> Corpus:
    """A corpus around some catalogs, with nothing else in it.

    Everything a check might want is left out unless a test asks for it, which
    is the same shape :func:`pydocvi.audit.assemble` produces when a file is
    missing, and the same shape the checks are written to survive.
    """
    values: dict[str, object] = {"root": root or Path("/corpus"), "catalogs": catalogs}
    values.update(overrides)
    return Corpus(**values)  # type: ignore[arg-type]


def upstream_of(*catalogs: Catalog, root: Path | None = None) -> dict[str, Catalog]:
    """The pin, keyed the way :func:`pydocvi.audit.assemble` keys it.

    By path relative to the upstream root, because that is the only thing a
    content catalog and an upstream catalog have in common: they live in
    different checkouts and the checks pair them up by name.
    """
    where = root or Path("/corpus")
    return {one.path.relative_to(where).as_posix(): one for one in catalogs}


def findings(check: Body, corpus: Corpus) -> list[Finding]:
    """What one check finds, called directly rather than through the registry.

    Directly, because the decorator hands the function straight back, and a test
    that went through the registry would be testing the registry as well as the
    rule and would fail for two unrelated reasons.
    """
    return list(check(corpus))


def machine_segment(msgid: str, msgstr: str, **provenance: object) -> Segment:
    return Segment.from_entry(Entry(msgid=msgid, msgstr=msgstr), source="machine", **provenance)
