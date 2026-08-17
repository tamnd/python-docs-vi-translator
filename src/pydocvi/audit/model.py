"""What a check is, what it is handed, and what it produces.

A finding names a file, a line and an entry, always. A check that can only say
that something is wrong somewhere in ``library/`` is not a check, it is a mood,
and a reviewer with 87 008 entries in front of them skips it.
"""

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydocvi.apply import Stamp
from pydocvi.catalog import Catalog, Entry
from pydocvi.glossary import Glossary
from pydocvi.memory import Memory

#: Where a generated report keeps the numbers a machine reads.
#:
#: ``A01`` and ``H06`` both compare a document's claims against a recount, and
#: neither can do it by reading the prose: a coverage table is written to be
#: read by a person, with rounded percentages and a paragraph explaining what
#: moved. This marker is the same device ``GLOSSARY.md`` uses for its term
#: table, for the same reason. The prose around it belongs to whoever is
#: writing it and moving a section does not change what the tool reads.
COUNTS = re.compile(r"<!--\s*counts:\s*(?P<body>\{.*?\})\s*-->", re.DOTALL)


def counts(markdown: str) -> dict[str, int] | None:
    """The counts a generated document records, or ``None`` if it records none.

    ``None`` rather than an empty mapping, because "this file has no marker" and
    "this file claims nothing was translated" are different situations and only
    one of them is a check that cannot run.
    """
    match = COUNTS.search(markdown)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group("body"))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return {str(key): int(value) for key, value in loaded.items() if isinstance(value, int)}


class Group(StrEnum):
    """The six families, in the order the report prints them."""

    STRUCTURE = "structure"
    PLACEHOLDERS = "placeholders"
    GLOSSARY = "glossary"
    LANGUAGE = "language"
    AVAILABILITY = "availability"
    HYGIENE = "hygiene"

    @property
    def prefix(self) -> str:
        """The letter every check in the group is numbered with."""
        return self.name[0]


def plural(count: int, noun: str) -> str:
    """A count and its noun, with the ``s`` only when it belongs there.

    Small, and shared rather than written twice, because the audit prints these
    counts on every CI run and into every report. "1 checks" is the kind of
    thing that makes a reader wonder what else was not looked at.
    """
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
    """One thing wrong, in one place.

    ``english`` and ``got`` are both here because a finding a reviewer has to go
    and look up is a finding a reviewer skips. ``P03 translated role target``
    with a file and a line is a bug report; the same line with the English
    beside what came back is a fix.
    """

    check: str
    path: str
    line: int = 0
    detail: str
    english: str = ""
    got: str = ""
    segment: str = ""

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}  {self.detail}"

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.check,
            "file": self.path,
            "line": self.line,
            "detail": self.detail,
            "english": self.english,
            "got": self.got,
            "segment": self.segment,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Corpus:
    """Everything the checks are allowed to look at.

    Assembled once and handed to all of them, because forty checks each reading
    548 catalogs off a disk is forty times the slowest part of the run.

    Nothing in here is a model, a route or a key. CI runs this on every push and
    it never calls the fleet, which is what makes it fast and what makes it
    worth trusting. Every hard check was written to be computable from the
    catalogs plus the upstream pin plus the glossary and nothing else.
    """

    root: Path
    catalogs: tuple[Catalog, ...] = ()
    upstream: dict[str, Catalog] = field(default_factory=dict)
    memory: Memory | None = None
    glossary: Glossary | None = None
    markdown: str | None = None
    tracked: tuple[Path, ...] = ()
    coverage: str | None = None
    quality: str | None = None
    readme: str | None = None
    queue: Path | None = None
    pin: Path | None = None
    upstream_root: Path | None = None
    stamp: Stamp | None = None

    def relative(self, path: Path) -> str:
        """A path as it reads in a finding, which is relative to the repo."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def translated(self) -> Iterator[tuple[Catalog, Entry]]:
        """Every entry that came back with something in it.

        The subject of most of the audit. An untranslated entry cannot break a
        placeholder rule or a glossary rule, and reporting one as passing is
        how a corpus that is 4 % done reads as clean.
        """
        for one in self.catalogs:
            for entry in one:
                if entry.msgstr:
                    yield one, entry

    def paired(self) -> Iterator[tuple[Catalog, Entry, Entry | None]]:
        """Every translated entry beside the upstream entry it claims to be.

        The upstream side is ``None`` when there is no such entry, which is
        ``S02``'s business to report and everyone else's to skip.
        """
        for one, entry in self.translated():
            source = self.upstream.get(self.relative(one.path))
            yield one, entry, None if source is None else source.by_id().get(entry.id)


#: A check's body. Yields rather than returns, so a check over 87 008 entries
#: reports its first finding before it has finished counting.
type Body = Callable[[Corpus], Iterator[Finding]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Check:
    """One rule, its identifier, and whether failing it stops the build."""

    id: str
    group: Group
    hard: bool
    title: str
    body: Body

    def run(self, corpus: Corpus) -> Result:
        return Result(check=self, findings=tuple(self.body(corpus)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Result:
    """What one check came to."""

    check: Check
    findings: tuple[Finding, ...] = ()

    @property
    def failed(self) -> bool:
        return bool(self.findings)

    def __str__(self) -> str:
        count = len(self.findings)
        if not count:
            return f"{self.check.id} passed"
        return f"{self.check.id} ({count} {'entry' if count == 1 else 'entries'})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """Every result of one run, and the verdict that follows from them."""

    results: tuple[Result, ...] = ()
    fail_soft: bool = False

    def failing(self, *, hard: bool | None = None) -> list[Result]:
        return [
            one for one in self.results if one.failed and (hard is None or one.check.hard is hard)
        ]

    @property
    def ok(self) -> bool:
        """Exit 0 iff every hard check passes.

        Soft checks are reported and do not change the verdict unless somebody
        asked for it. ``L05``, ``L06`` and ``L08`` are soft on purpose: each has
        real exceptions, a heading that is genuinely an instruction or a
        sentence Vietnamese splits in two, and a hard rule that a correct
        translation breaks is not a rule.

        This is expected to be red for a long time. That is the job.
        """
        if self.failing(hard=True):
            return False
        return not (self.fail_soft and self.failing())

    def of_group(self, group: Group) -> list[Result]:
        return [one for one in self.results if one.check.group is group]

    @property
    def findings(self) -> list[Finding]:
        return [finding for one in self.results for finding in one.findings]

    def as_json(self) -> str:
        """The finding list, for CI annotations."""
        payload = {
            "ok": self.ok,
            "checks": len(self.results),
            "failing": [one.check.id for one in self.failing()],
            "findings": [finding.as_dict() for finding in self.findings],
        }
        return json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


class UnknownCheckError(ValueError):
    """A name that is neither a check nor a group.

    Carries the near misses, because the difference between ``P03`` and ``S03``
    is one keystroke and the difference in what they mean is total.
    """

    def __init__(self, name: str, known: Sequence[str]) -> None:
        near = sorted(one for one in known if one[-2:].casefold() == name[-2:].casefold())
        suffix = f". Did you mean {', '.join(near)}?" if near else ""
        super().__init__(f"no check or group called {name!r}{suffix}")


@dataclass(slots=True, kw_only=True)
class Registry:
    """Every check there is, in the order they were declared.

    Declaration order is report order, which is why this is a list and not the
    set it would otherwise be. The groups are meant to be read top to bottom: a
    structural failure explains most of what follows it, and printing the
    ``L02`` above the ``S02`` that caused it wastes the reader's first minute.
    """

    checks: list[Check] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.checks)

    def add(self, check: Check) -> Check:
        if any(one.id == check.id for one in self.checks):
            raise ValueError(f"{check.id} is registered twice")
        self.checks.append(check)
        return check

    def check(
        self, identifier: str, group: Group, *, hard: bool, title: str
    ) -> Callable[[Body], Body]:
        """Register a check and hand the function straight back.

        The function rather than the ``Check``, so that a check stays callable
        on its own in a test, without going through the registry and without
        the five other things a registry lookup would drag in.
        """

        def register(body: Body) -> Body:
            if not identifier.startswith(group.prefix):
                raise ValueError(f"{identifier} does not belong to group {group}")
            self.add(Check(id=identifier, group=group, hard=hard, title=title, body=body))
            return body

        return register

    def selected(self, *, only: Sequence[str] = (), skip: Sequence[str] = ()) -> list[Check]:
        """The checks a person asked for.

        Both flags take ids, group names and the one-letter group prefixes,
        because ``--skip glossary`` while the glossary is being rewritten is the
        request somebody actually has, and making them spell out six ids to ask
        it is the friction that ends in the whole audit being switched off
        instead. ``--only P,G`` is the same request typed by somebody who reads
        the check ids more often than the group names.
        """
        names = (
            {one.id for one in self.checks}
            | {str(one) for one in Group}
            | {one.prefix for one in Group}
        )
        wanted = {one.casefold() for one in only}
        unwanted = {one.casefold() for one in skip}
        known = {one.casefold() for one in names}
        for name in (*only, *skip):
            #: Reported as it was typed rather than as it was folded, because a
            #: message that changes the input is a message a reader has to
            #: reconcile with what they wrote before they can see the typo.
            if name.casefold() not in known:
                raise UnknownCheckError(name, sorted(names))
        return [
            one
            for one in self.checks
            if (not wanted or _matches(one, wanted)) and not _matches(one, unwanted)
        ]


def _matches(check: Check, names: set[str]) -> bool:
    return (
        check.id.casefold() in names
        or str(check.group) in names
        or check.group.prefix.casefold() in names
    )
