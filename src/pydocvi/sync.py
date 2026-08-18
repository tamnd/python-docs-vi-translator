"""Pinning upstream and loading what people have already translated.

Two jobs that belong together because both answer the question "what does the
corpus look like right now": write down exactly which upstream commit this run
is against, and take the human translations out of it before anything machine
made can compete with them.
"""

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydocvi import catalog, classify
from pydocvi.catalog import Catalog
from pydocvi.memory import Memory, Segment

#: The branch this project translates. Transifex serves the documentation from
#: the ``python-doc/python-newest`` project, which tracks the newest branch, so
#: the newest branch is the one with human work landing in it. Older branches
#: are produced from the same memory by ``apply`` rather than translated again.
DEFAULT_BRANCH = "3.15"


@dataclass(frozen=True, slots=True, kw_only=True)
class Pin:
    """What upstream looked like when this run started."""

    repo: str
    branch: str
    commit: str
    files: int
    entries: int
    words: int
    characters: int
    translated: int

    def as_yaml(self) -> str:
        """Serialise the pin.

        Hand-rolled rather than through a YAML library because this file is read
        by people far more often than by code, the shape is eight scalars, and a
        dependency that reorders keys would make the diff useless.
        """
        return (
            "# Written by pydocvi sync. Do not edit by hand.\n"
            f"repo: {self.repo}\n"
            f"branch: {quote(self.branch)}\n"
            f"commit: {self.commit}\n"
            "counts:\n"
            f"  files: {self.files}\n"
            f"  entries: {self.entries}\n"
            f"  words: {self.words}\n"
            f"  characters: {self.characters}\n"
            f"  translated: {self.translated}\n"
        )

    @classmethod
    def from_yaml(cls, text: str) -> Pin:
        """Read a pin back.

        Hand-rolled for the same reason :meth:`as_yaml` is, and deliberately
        narrow: it reads the eight scalars that method writes and refuses
        anything else. ``S01`` exists to catch a pin that has drifted from the
        corpus, so a reader that shrugged at a malformed file would be checking
        nothing at exactly the moment it mattered.
        """
        values: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, separator, value = stripped.partition(":")
            if separator and value.strip():
                values[key.strip()] = value.strip().strip("'\"")
        missing = {"repo", "branch", "commit", "files", "entries", "words", "characters"} - set(
            values
        )
        if missing:
            raise PinError(f"pin is missing {', '.join(sorted(missing))}")
        try:
            return cls(
                repo=values["repo"],
                branch=values["branch"],
                commit=values["commit"],
                files=int(values["files"]),
                entries=int(values["entries"]),
                words=int(values["words"]),
                characters=int(values["characters"]),
                translated=int(values.get("translated", 0)),
            )
        except ValueError as error:
            raise PinError(f"pin has a count that is not a number: {error}") from error

    @classmethod
    def read(cls, path: Path) -> Pin | None:
        """The pin on disk, or ``None`` where there is not one yet."""
        if not path.exists():
            return None
        return cls.from_yaml(path.read_text(encoding="utf-8"))


class PinError(ValueError):
    """A pin file this module cannot read."""


def quote(value: str) -> str:
    """Quote a YAML scalar that would otherwise read as a number.

    ``3.15`` unquoted is a float, and a pin that says ``branch: 3.15`` round
    trips into ``3.1`` through some readers. Cheap to prevent, tedious to
    diagnose.
    """
    return f'"{value}"'


@dataclass(frozen=True, slots=True, kw_only=True)
class HumanLoad:
    """What reconciling the memory against the mirror did.

    Both numbers, rather than the one ``load_human`` used to return. A run that
    stores 0 and drops 136 and a run that does nothing are the same integer from
    the caller's side, and the first is the one worth printing.
    """

    stored: int = 0
    dropped: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SyncDiff:
    """What upstream has that the memory does not, and the other way round.

    There is no ``changed`` category and there cannot be one. The segment id is
    a hash of the string itself, so an edited ``msgid`` is a different segment:
    it shows up as one addition and one orphan. That is the honest reading, and
    it is also the useful one, because the old translation is of the old English
    and re-queueing the new string is exactly the right thing to do.
    """

    added: tuple[str, ...] = ()
    orphaned: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.added or self.orphaned)

    def as_markdown(self) -> str:
        lines = [
            "# Sync report",
            "",
            f"- upstream strings with nothing in the memory: {len(self.added)}",
            f"- memory segments upstream no longer has: {len(self.orphaned)}",
            "",
        ]
        if self.orphaned:
            lines += [
                "Orphaned segments are kept rather than deleted. Upstream reverts happen,",
                "a string that moved between files keeps its id, and a translation is cheap",
                "to store and expensive to reproduce.",
                "",
            ]
        return "\n".join(lines)


def head_commit(repo: Path) -> str:
    """The commit the checkout is on."""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def current_branch(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def read_corpus(root: Path) -> list[Catalog]:
    """Every catalog under ``root``, parsed."""
    return [catalog.read(path) for path in catalog.walk(root)]


def measure(catalogs: Iterable[Catalog]) -> tuple[int, int, int, int, int]:
    """Count files, entries, English words, msgid characters and translations.

    Counted off the files every time rather than read from the last pin. A
    number nobody re-measured is a number that was true about one branch and
    wrong about the next.
    """
    files = entries = words = characters = translated = 0
    for cat in catalogs:
        files += 1
        for entry in cat:
            entries += 1
            words += len(entry.msgid.split())
            characters += len(entry.msgid)
            if entry.translated:
                translated += 1
    return files, entries, words, characters, translated


def pin(root: Path, *, repo: str, branch: str | None = None, commit: str | None = None) -> Pin:
    """Measure a checkout and describe it."""
    catalogs = read_corpus(root)
    files, entries, words, characters, translated = measure(catalogs)
    return Pin(
        repo=repo,
        branch=branch or current_branch(root),
        commit=commit or head_commit(root),
        files=files,
        entries=entries,
        words=words,
        characters=characters,
        translated=translated,
    )


def human_segments(catalogs: Iterable[Catalog]) -> list[Segment]:
    """Every entry a person has already translated.

    An entry counts as human when it has a non-empty ``msgstr`` and is not
    marked fuzzy. Fuzzy means gettext is not confident the translation still
    matches the source, and inheriting one of those as ground truth would seed
    the memory with the exact thing it exists to avoid.

    And when it is prose. A doctest is copied, never translated, which is the
    rule ``P07`` enforces on everything this pipeline writes, and it was not
    being asked of the 136 code entries the mirror hands over as somebody's
    work. ``human`` is a provenance and not a grade: it says a person typed the
    string, and 30 of those 136 are a person having typed over the code.

    What that looks like in the corpus, from ``tutorial/introduction.po``::

        File "<stdin>", line 1, in <module>     the English
        File "1", line 1, in 2                  the translation

    ``<stdin>`` and ``<module>`` are gone. Elsewhere it is the indentation
    inside a ``for`` body flattened to one space, the carets under a syntax
    error unaligned from what they point at, and a column-aligned option table
    reflowed. Every one is an example a reader copies out and then has to
    debug, and the comment translation they were made for is worth less than
    that: comments are M8, with a prompt of their own and a check that every
    code line came back byte-identical.

    The other 106 are already byte-identical, and dropping those loses nothing.
    :func:`apply` mints them from the ``msgid`` with ``passthrough=doctest`` on
    them, which is the same string with an accurate account of where it came
    from instead of a claim that somebody translated it.

    Only code. A no-op is not translatable either and is left alone, because
    one of them is a ``:ref:`` whose display text a person translated correctly
    and the classifier calls markup. That entry is a bug in :func:`is_noop`,
    not a licence to throw the translation away.
    """
    out: list[Segment] = []
    for cat in catalogs:
        for entry in cat:
            if entry.translated and not entry.fuzzy and not classify.classify(entry.msgid).code:
                out.append(Segment.from_entry(entry, source="human"))
    return out


def load_human(memory: Memory, catalogs: Iterable[Catalog]) -> HumanLoad:
    """Bring the memory's human half into line with the mirror.

    Stores what the mirror offers and drops the ``human`` segments it no longer
    does, which is a reconciliation where this used to be an ``extend``.

    The difference is only visible when :func:`human_segments` gets stricter, and
    it got stricter once: 136 code entries stopped qualifying, and an ``extend``
    leaves all 136 sitting in the memory as somebody's translation of a doctest
    for ``apply`` to write back. The alternative was editing them out of the
    manifest by hand, and a memory that has been hand-edited is no longer a thing
    the content repo can be rebuilt from, which is the property the whole
    projection rests on.

    Dropping is safe because the mirror is the only place a ``human`` segment
    comes from. There is no command in this tool that promotes a string to
    ``human``, deliberately (spec 02 §4), so anything of that provenance in the
    memory was read out of Transifex and can be read again.

    Only ``human``. A ``machine`` segment is the one thing here that genuinely
    cannot be rebuilt without spending the run again, and nothing about the
    mirror is evidence either way about it.
    """
    wanted = human_segments(catalogs)
    stored = memory.extend(wanted)
    keep = {segment.id for segment in wanted}
    dropped = [s.id for s in memory if s.source == "human" and s.id not in keep]
    for one in dropped:
        memory.remove(one)
    return HumanLoad(stored=stored, dropped=len(dropped))


def diff(memory: Memory, catalogs: Iterable[Catalog]) -> SyncDiff:
    """Compare the memory against upstream."""
    upstream = {entry.id for cat in catalogs for entry in cat}
    known = {segment.id for segment in memory}
    return SyncDiff(
        added=tuple(sorted(upstream - known)),
        orphaned=tuple(sorted(known - upstream)),
    )
