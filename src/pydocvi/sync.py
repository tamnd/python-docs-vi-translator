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

from pydocvi import catalog
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


def quote(value: str) -> str:
    """Quote a YAML scalar that would otherwise read as a number.

    ``3.15`` unquoted is a float, and a pin that says ``branch: 3.15`` round
    trips into ``3.1`` through some readers. Cheap to prevent, tedious to
    diagnose.
    """
    return f'"{value}"'


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
    """
    out: list[Segment] = []
    for cat in catalogs:
        for entry in cat:
            if entry.translated and not entry.fuzzy:
                out.append(Segment.from_entry(entry, source="human"))
    return out


def load_human(memory: Memory, catalogs: Iterable[Catalog]) -> int:
    """Load human translations into the memory. Returns how many were stored."""
    return memory.extend(human_segments(catalogs))


def diff(memory: Memory, catalogs: Iterable[Catalog]) -> SyncDiff:
    """Compare the memory against upstream."""
    upstream = {entry.id for cat in catalogs for entry in cat}
    known = {segment.id for segment in memory}
    return SyncDiff(
        added=tuple(sorted(upstream - known)),
        orphaned=tuple(sorted(known - upstream)),
    )
