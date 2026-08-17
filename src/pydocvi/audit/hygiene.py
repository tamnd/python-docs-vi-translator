"""``H01`` to ``H06``: what is in the repository that should not be.

All hard, all cheap, and none of them about translation. This group exists
because the content repo is 548 catalogs and 87 008 entries, and at that size
nobody reviews a diff closely enough to notice a 40 MB compiled catalog or a
key pasted into a manifest. A check does notice, every push, in under a second.

``H03`` is the one that matters and the one with no second chance. A key that
reaches a public repository is a key that has to be rotated whatever anybody
does afterwards, because it is in the reflog, in every fork and in whatever
scraped the push within the minute. Deleting the commit does not undo it.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from pydocvi.audit.model import Corpus, Finding, Group, Registry, counts

registry = Registry()
check = registry.check

#: Compiled catalogs. Generated from the ``.po`` files, machine-specific, and
#: large. Sphinx builds them at publication time and a tracked one is only ever
#: out of date with the source beside it.
COMPILED = ".mo"

#: A tracked file this size is a mistake unless it is a catalog. Half a megabyte
#: of anything else in a documentation repo is an image nobody meant to add or
#: an archive somebody unpacked in the wrong directory.
LARGE = 512 * 1024

#: A catalog this size is also a mistake, but the bound has to clear the real
#: corpus and the real corpus has ``whatsnew/changelog.po`` in it at 5 126 KB.
#: The spec said 4 MB and the spec had not measured; a ceiling that fails on
#: the largest legitimate file in the repository is a ceiling that gets raised
#: in a hurry by whoever is trying to land something else.
LARGE_CATALOG = 8 * 1024 * 1024

#: The one tracked file that is meant to be big and to keep getting bigger.
#: Every other large file is something nobody chose; this one is every segment
#: the project has, and ``apply --check`` is not checkable without it. Named
#: rather than given a ceiling, because any number here is a number that gets
#: raised in a hurry by whoever is trying to land a translation run.
#:
#: One path, not a pattern, and not the start of an allowlist. It grows with the
#: corpus and the answer at that point is to shard it by top-level directory,
#: which is translator issue #41, not to widen this.
EXPECTED_LARGE = "manifests/memory.json"

#: Directories and files that belong to a working copy rather than to the
#: project. ``.DS_Store`` is on the list because there is one in the content
#: repo's ``MACHINE/`` directory today.
UNWANTED = ("venv/", ".venv/", "locales/", "__pycache__/", ".DS_Store", ".mypy_cache/")

#: The key shape, applied to every tracked file with no exemption anywhere. The
#: pattern is the one ``make secrets`` uses, character for character on purpose:
#: a check stricter than the Makefile would fail builds the Makefile passes, and
#: a looser one would be worse than nothing.
KEYS = (re.compile(r"sk-[A-Za-z0-9._-]{8,}"),)

#: The two shapes that are not API keys and are just as bad to publish. Held
#: apart from :data:`KEYS` because these are checked everywhere except inside a
#: catalog, and the reason is what this corpus is. It is the Python
#: documentation translated, so ``library/ssl.po`` contains the literal PEM
#: header as documentation text and ``library/contextvars.po`` contains a
#: ``curl`` example against a loopback port. Both matched on the first real run
#: and both were correct content. A ``msgid`` comes from CPython and cannot
#: carry our key, a ``msgstr`` is a translation of that ``msgid``, and neither
#: is anywhere a person pastes credentials. The key rule above still applies to
#: catalogs, because that one costs nothing to keep absolute.
EXPOSURES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b"),
)

#: Addresses that carry no information about where anything runs. Loopback is
#: every tutorial's example server, and the three RFC 5737 blocks exist so that
#: documentation has addresses it can print. Matching them tells a reader
#: nothing and costs the check its credibility.
EXAMPLE_HOSTS = re.compile(r"^(?:127\.|0\.0\.0\.0|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)")

#: Suffixes worth reading as text for ``H04``. Anything else is either generated
#: or binary, and reporting a trailing space inside a PNG helps nobody.
TEXT = frozenset({".po", ".pot", ".md", ".yaml", ".yml", ".py", ".toml", ".cfg", ".txt"})


@check("H01", Group.HYGIENE, hard=True, title="no compiled catalog is tracked")
def h01_no_mo_files(corpus: Corpus) -> Iterator[Finding]:
    """No ``.mo`` file is tracked.

    They are a build artefact of the ``.po`` beside them, they are binary, and
    a tracked one drifts from its source the first time anybody edits the
    source and forgets to recompile.
    """
    for path in corpus.tracked:
        if path.suffix == COMPILED:
            yield Finding(
                check="H01",
                path=corpus.relative(path),
                detail="a compiled catalog belongs to the build, not the repository",
            )


@check("H02", Group.HYGIENE, hard=True, title="no oversized file is tracked")
def h02_no_large_files(corpus: Corpus) -> Iterator[Finding]:
    """Nothing over half a megabyte except a catalog, and no catalog over eight.

    Two ceilings because the two cases mean different things. A large catalog is
    plausible and is checked against a generous bound; a large anything else is
    almost always something that was never meant to be committed.
    """
    for path in corpus.tracked:
        if not path.exists() or corpus.relative(path) == EXPECTED_LARGE:
            continue
        size = path.stat().st_size
        catalog = path.suffix in {".po", ".pot"}
        ceiling = LARGE_CATALOG if catalog else LARGE
        if size > ceiling:
            yield Finding(
                check="H02",
                path=corpus.relative(path),
                detail=f"{size / 1024:,.0f} KB, over the {ceiling / 1024:,.0f} KB ceiling",
            )


@check("H03", Group.HYGIENE, hard=True, title="no secret-shaped string is tracked")
def h03_no_secrets(corpus: Corpus) -> Iterator[Finding]:
    """No key, private key header or host and port in a tracked file.

    Deliberately shape-based and deliberately without an allowlist. An allowlist
    is how this check dies: one fixture is exempted, the exemption is copied,
    and a year later nobody can say which of the twenty entries are real. Test
    fixtures that need a key-shaped value assemble it at import time instead,
    which costs one line and keeps the rule absolute.

    The key rule holds everywhere. The private key header and the host and port
    are not applied inside a catalog, for the reason recorded on
    :data:`EXPOSURES`: the corpus is CPython's documentation, so it contains
    prose about keys and loopback addresses, and a check that fires on the
    ``ssl`` page is a check that gets muted.

    The finding never quotes what it matched. A check that printed the secret
    into a CI log to tell you the secret was in a file would have published it a
    second time, somewhere with a longer retention.
    """
    for path in corpus.tracked:
        catalog = path.suffix in {".po", ".pot"}
        patterns = KEYS if catalog else KEYS + EXPOSURES
        for number, line in _lines(path):
            if any(_matches(pattern, line) for pattern in patterns):
                yield Finding(
                    check="H03",
                    path=corpus.relative(path),
                    line=number,
                    detail="a secret-shaped string is on this line, and is not quoted here",
                )


@check("H04", Group.HYGIENE, hard=True, title="text files are well formed")
def h04_text_is_well_formed(corpus: Corpus) -> Iterator[Finding]:
    """UTF-8, LF, one trailing newline, no trailing whitespace.

    Every one of these is invisible in a rendered page and every one of them
    produces diff noise, which in a 548 file repository is the difference
    between a review that finds a mistranslation and a review that gives up.

    Line width is deliberately not checked here, although the spec asked for it
    at 79 columns. :mod:`pydocvi.catalog` records that no wrapping rule
    reproduces the inherited corpus, 86.31 per cent being the best any of them
    manages, so a hard column bound fails on thousands of correct lines, which
    is exactly the way a check stops being read. ``S08`` covers the ground that
    matters anyway: it renders every catalog and requires byte identity with
    what is committed, and wrapping is part of what it compares.
    """
    for path in corpus.tracked:
        if path.suffix not in TEXT or not path.exists():
            continue
        where = corpus.relative(path)
        raw = path.read_bytes()
        if not raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            yield Finding(check="H04", path=where, detail=f"is not UTF-8: {error.reason}")
            continue
        if b"\r" in raw:
            yield Finding(check="H04", path=where, detail="has carriage returns")
        if not text.endswith("\n"):
            yield Finding(check="H04", path=where, detail="does not end in a newline")
        elif text.endswith("\n\n"):
            yield Finding(check="H04", path=where, detail="ends in more than one newline")
        for number, line in enumerate(text.split("\n"), start=1):
            if line != line.rstrip():
                yield Finding(
                    check="H04", path=where, line=number, detail="has trailing whitespace"
                )


@check("H05", Group.HYGIENE, hard=True, title="no working-copy directory is tracked")
def h05_no_working_copy(corpus: Corpus) -> Iterator[Finding]:
    """No ``venv/``, ``locales/``, ``__pycache__/`` or ``.DS_Store``.

    ``locales/`` is on the list and the other three are obvious. It is where
    ``sphinx-intl`` puts a working copy of the catalogs, so a tracked one is a
    second, silently diverging set of the same 548 files, and a reviewer has no
    way to tell which of the two the build used.
    """
    for path in corpus.tracked:
        where = corpus.relative(path)
        for unwanted in UNWANTED:
            if unwanted.endswith("/") and f"{unwanted}" in f"{where}/":
                yield Finding(
                    check="H05", path=where, detail=f"is inside {unwanted}, which is not tracked"
                )
                break
            if not unwanted.endswith("/") and path.name == unwanted:
                yield Finding(check="H05", path=where, detail=f"{unwanted} is not tracked")
                break


@check("H06", Group.HYGIENE, hard=True, title="the README agrees with the coverage report")
def h06_readme_matches_coverage(corpus: Corpus) -> Iterator[Finding]:
    """The README's coverage table says what ``reports/coverage.md`` says.

    The README is the only one of the two anybody reads without being asked to,
    which makes it the one most worth keeping true and the one most likely to go
    stale. Both carry the same generated counts, so this compares them rather
    than reading either as prose.
    """
    if corpus.readme is None or corpus.coverage is None:
        return
    stated = counts(corpus.readme)
    if stated is None:
        yield Finding(check="H06", path="README.md", detail="has no coverage counts to check")
        return
    wanted = counts(corpus.coverage)
    if wanted is None:
        yield Finding(
            check="H06", path="reports/coverage.md", detail="has no coverage counts to check"
        )
        return
    for tier in sorted(set(stated) | set(wanted)):
        if stated.get(tier) != wanted.get(tier):
            yield Finding(
                check="H06",
                path="README.md",
                detail=(
                    f"tier {tier}: the README says {stated.get(tier, 'nothing')}, "
                    f"the report says {wanted.get(tier, 'nothing')}"
                ),
            )


def _matches(pattern: re.Pattern[str], line: str) -> bool:
    """Whether a line matches, once the example addresses are discounted.

    Every match counts except an address out of the documentation blocks. Those
    are filtered here rather than by making the address pattern cleverer,
    because the pattern is meant to be readable by whoever is deciding whether a
    finding is real, and an IPv4 regex that excludes four ranges by lookahead is
    not readable by anybody.
    """
    for found in pattern.finditer(line):
        if EXAMPLE_HOSTS.match(found.group()):
            continue
        return True
    return False


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    """Every line of a tracked file, or nothing at all if it is not text.

    Binary files are skipped rather than reported. ``H02`` is the check with an
    opinion about what binary files may be here; this one only needs to look
    inside the ones it can read.
    """
    if not path.exists() or not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return
    yield from enumerate(text.splitlines(), start=1)
