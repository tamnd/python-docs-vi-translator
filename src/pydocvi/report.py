"""The two generated reports: how much is done, and how good it is.

Both are written off the committed corpus rather than off the run that produced
it, with one exception that is named where it happens. A report that a run
writes about itself says what the run believed; a report that recounts the
corpus says what a reader will actually get, and those two have already
disagreed once on this project.

The exception is the refusal rate. A refusal that was fixed on rung 2 leaves no
trace in the corpus at all, so the only place that number can come from is the
tally the run wrote down, and :class:`~pydocvi.translate.Tally` is persisted for
exactly this.

Nothing here decides anything. The audit decides, and these two files are what
it decides against: ``A01`` recounts coverage and fails when this file is stale,
``H06`` compares the README against it, and ``A02`` and ``A05`` require that
every dead job and every heavy refusal appear in the quality report. Generating
the evidence and checking it are deliberately two programs.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydocvi import apply, batch, invariants, segment
from pydocvi.audit import Corpus
from pydocvi.catalog import Entry
from pydocvi.queue import Job, Stage, State
from pydocvi.translate import Tally

#: The fence the generated coverage table sits between in ``README.md``, the
#: same device ``GLOSSARY.md`` uses for its term table. The prose around it is
#: the half no machine can write and moving a section does not disturb it.
TABLE_OPEN = "<!-- generated: coverage -->"
TABLE_CLOSE = "<!-- /generated: coverage -->"

#: Words, for the word column. Runs of anything that is not a space, which is
#: how every count this project has published so far was made and therefore the
#: only definition that keeps the numbers comparable.
_WORDS = re.compile(r"\S+")


class ReportError(ValueError):
    """A generated file that cannot be regenerated in place."""


class Written(StrEnum):
    """How an entry came to say what it says.

    Read off the entry rather than off the memory, because the entry is what
    ships. An entry the memory believes was translated and that the catalog does
    not carry is ``A03``'s business and it is not coverage.
    """

    HUMAN = "human"
    MACHINE = "machine"
    PASSTHROUGH = "passthrough"
    LEGACY = "legacy"
    UNTRANSLATED = "untranslated"


#: Column order, which is also precedence order in the prose: only the first may
#: be called translated.
COLUMNS: tuple[Written, ...] = tuple(Written)


def written_as(entry: Entry) -> Written:
    """Which column one entry belongs in.

    The four rules in order, and the order matters. A provenance comment is the
    strongest evidence there is, because this tool wrote it and wrote it at the
    moment it knew. Absence of one is weaker evidence and is read through the
    fuzzy flag: everything this tool writes is fuzzy, so a translated entry with
    no marker and no fuzzy flag is somebody's reviewed work, and a fuzzy one
    with no marker came from upstream before this tool existed.
    """
    if not entry.msgstr:
        return Written.UNTRANSLATED
    for comment in entry.comments:
        if comment.startswith(apply.MARKER):
            if apply.PASSTHROUGH_FIELD in comment:
                return Written.PASSTHROUGH
            return Written.MACHINE
    return Written.LEGACY if entry.fuzzy else Written.HUMAN


@dataclass(slots=True, kw_only=True)
class Count:
    """Entries and words in one section, split by how they were written."""

    entries: Counter[Written] = field(default_factory=Counter)
    words: Counter[Written] = field(default_factory=Counter)

    def add(self, entry: Entry) -> None:
        where = written_as(entry)
        self.entries[where] += 1
        #: Counted on the English in every column including the untranslated
        #: one, so that "words done" and "words to do" add up to one corpus.
        #: Counting the Vietnamese would make the total move when a translation
        #: is longer than its source, which is most of the time.
        self.words[where] += len(_WORDS.findall(entry.msgid))

    @property
    def total(self) -> int:
        return sum(self.entries.values())

    @property
    def done(self) -> int:
        """Entries carrying anything at all, which is what ``A01`` recounts."""
        return self.total - self.entries[Written.UNTRANSLATED]

    @property
    def translated(self) -> int:
        """Entries a person has agreed with, which is the only honest number.

        Separate from :attr:`done` and named differently on purpose. The machine
        column is a corpus and the passthrough column is work nobody did, and
        collapsing the three is how a project comes to claim a million
        translated words.
        """
        return self.entries[Written.HUMAN]

    def share(self, of: Written) -> float:
        return self.entries[of] / self.total if self.total else 0.0


def by_tier(corpus: Corpus) -> dict[int, Count]:
    """Counts per tier, in tier order."""
    return _tally(corpus, lambda path: batch.tier_of(path))


def by_section(corpus: Corpus) -> dict[str, Count]:
    """Counts per top-level directory, with the root files under their own name.

    A section is what a reader recognises. ``library/`` is one thing to them
    however many files it is to us, and a table with 548 rows is a table nobody
    reads to the end of.
    """
    return _tally(corpus, lambda path: path.split("/")[0] if "/" in path else path)


def by_file(corpus: Corpus) -> dict[str, Count]:
    """Counts per catalog, for the quality report's detail table."""
    return _tally(corpus, lambda path: path)


def coverage(corpus: Corpus) -> str:
    """``reports/coverage.md``.

    Five columns because collapsing them is the failure this file exists to
    prevent. The repo holds roughly a million words of 2026 Google output and
    ``ROADMAP.md`` says "zero words formally translated", and both statements
    were defensible only because nobody had generated this table.
    """
    tiers = by_tier(corpus)
    whole = _sum(tiers.values())
    lines = [
        "# Coverage",
        "",
        f"{whole.total:,} entries, {sum(whole.words.values()):,} English words, "
        f"across {len(corpus.catalogs):,} catalogs.",
        "",
        f"**{whole.translated:,} entries are translated**, which is "
        f"{whole.share(Written.HUMAN):.1%}. That is the human column and nothing else. "
        f"The {whole.entries[Written.MACHINE]:,} machine entries are a corpus to review, "
        f"not a translation, and the {whole.entries[Written.PASSTHROUGH]:,} passthrough "
        "entries are strings that never needed translating.",
        "",
        counts_marker(tiers),
        "",
        "## By tier",
        "",
        *_table("Tier", {f"{number}": one for number, one in sorted(tiers.items())}),
        "",
        "## By section",
        "",
        *_table("Section", dict(sorted(by_section(corpus).items()))),
        "",
    ]
    return "\n".join(lines)


def counts_marker(tiers: dict[int, Count]) -> str:
    """The line ``A01`` and ``H06`` read, and the only line here they read.

    A generated table is written for a person, with rounded percentages and a
    paragraph explaining what moved, and no check can compare that against a
    recount. This is the machine-readable half, and it holds entries done rather
    than entries translated because that is what the audit recounts off the
    catalogs.
    """
    body = ", ".join(f'"{number}": {one.done}' for number, one in sorted(tiers.items()))
    return f"<!-- counts: {{{body}}} -->"


def render(markdown: str, corpus: Corpus) -> str:
    """Put the coverage table back into ``README.md``, prose untouched.

    The README is the only one of the two files anybody reads without being
    asked to, which makes it the one most worth keeping true and the one most
    likely to go stale. ``H06`` is what stops it going stale; this is what makes
    fixing it a command rather than an afternoon.
    """
    start = markdown.find(TABLE_OPEN)
    end = markdown.find(TABLE_CLOSE)
    if start < 0 or end < 0 or end < start:
        raise ReportError(f"{TABLE_OPEN} and {TABLE_CLOSE} must both be present, in that order")
    tiers = by_tier(corpus)
    body = "\n".join([counts_marker(tiers), "", *_table("Tier", _named(tiers))])
    return f"{markdown[: start + len(TABLE_OPEN)]}\n\n{body}\n\n{markdown[end:]}"


def quality(corpus: Corpus, *, tallies: Sequence[Tally] = (), limit: int = 20) -> str:
    """``reports/quality.md``.

    Everything in here is either a recount off the committed corpus or a number
    a run wrote down, and the two are labelled as such. Where there is no run on
    record the section says so rather than printing zeros, because a rate of
    0.0 % reads as a perfect run and an absent one reads as absent.
    """
    lines = [
        "# Quality",
        "",
        *_invariants(corpus),
        "",
        *_refusals(tallies, limit=limit),
        "",
        *_adherence(corpus, limit=limit),
        "",
        *_dead(corpus),
        "",
        *_routes(tallies),
        "",
    ]
    return "\n".join(lines)


def _invariants(corpus: Corpus) -> list[str]:
    """Hard invariant pass rate per rule, recounted over what was committed.

    Over machine entries only. A human entry that breaks ``P08`` is a person's
    decision about their own sentence, and the number this section reports is
    whether the pipeline's own output holds, which a corpus-wide rate would
    dilute with work the pipeline did not do.
    """
    seen = 0
    broke: Counter[str] = Counter()
    for _, entry in corpus.translated():
        if written_as(entry) is not Written.MACHINE:
            continue
        seen += 1
        for violation in invariants.check_entry(
            entry.msgid, entry.msgstr, segment.spans_of(entry.msgid)
        ):
            broke[violation.rule] += 1
    lines = [
        "## Invariants, recounted",
        "",
        f"{seen:,} machine-written entries in the corpus. Every one of them passed the "
        "invariants when it was accepted, so anything below is a rule that was tightened "
        "afterwards or an entry something else edited.",
        "",
    ]
    if not seen:
        return [*lines[:2], "No machine-written entry is in the corpus yet."]
    if not broke:
        return [*lines, "Every rule passes on every entry."]
    lines += ["| Rule | Failing | Pass rate |", "| --- | ---: | ---: |"]
    for rule, count in sorted(broke.items()):
        lines.append(f"| `{rule}` | {count:,} | {1 - count / seen:.2%} |")
    return lines


def _refusals(tallies: Sequence[Tally], *, limit: int) -> list[str]:
    """Refusal rate by rule and by rung, from the runs' own records.

    The one section not recounted from the corpus, and it cannot be: an entry
    refused on rung 1 and accepted on rung 2 is in the corpus exactly once, with
    no sign that it cost three calls. This is the number that says whether the
    prompt is working, so it is the number worth persisting.
    """
    if not tallies:
        return [
            "## Refusals",
            "",
            "No translation run on record. Refusal rates come from the run that produced "
            "the entries and cannot be recovered from the corpus afterwards.",
        ]
    whole = _merge(tallies)
    lines = [
        "## Refusals",
        "",
        f"{len(tallies)} runs, {whole.batches:,} batches, {whole.accepted:,} entries accepted, "
        f"{whole.refused:,} refused, {whole.rejected:,} batches refused whole, "
        f"{whole.dead:,} entries out of rungs. Acceptance rate {whole.rate:.2%}.",
        "",
        "| Rule | Refusals | Share |",
        "| --- | ---: | ---: |",
    ]
    total = sum(whole.by_rule.values())
    worst = sorted(whole.by_rule.items(), key=lambda pair: (-pair[1], pair[0]))
    for rule, count in worst[:limit]:
        lines.append(f"| `{rule}` | {count:,} | {count / total:.1%} |")
    if len(worst) > limit:
        lines.append(f"| and {len(worst) - limit} more rules | | |")
    lines += ["", "| Rung | Refusals |", "| --- | ---: |"]
    for rung, count in sorted(whole.by_attempt.items()):
        lines.append(f"| {rung} | {count:,} |")
    return lines


def _adherence(corpus: Corpus, *, limit: int) -> list[str]:
    """Glossary adherence, worst terms first.

    The same walk ``G02`` and ``G03`` do, counted instead of listed. A soft
    check with 4 000 findings tells a reviewer nothing; the twenty terms behind
    most of them tells them what to fix or what to argue with, and at this stage
    arguing with the row is as likely to be right.
    """
    if corpus.glossary is None:
        return ["## Glossary adherence", "", "No glossary on this checkout."]
    matcher = corpus.glossary.matcher()
    missed: Counter[str] = Counter()
    seen = 0
    for one, entry in corpus.translated():
        seen += 1
        where = corpus.relative(one.path)
        for term in matcher.missing(entry.msgid, entry.msgstr, where=where, msgctxt=entry.msgctxt):
            missed[term.en] += 1
    lines = [
        "## Glossary adherence",
        "",
        f"{len(corpus.glossary)} terms against {seen:,} translated entries. "
        f"{sum(missed.values()):,} misses over {len(missed)} terms.",
        "",
    ]
    if not missed:
        return [*lines[:3], "", "Every term that appeared was rendered."]
    lines += ["| Term | Misses |", "| --- | ---: |"]
    for english, count in sorted(missed.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]:
        lines.append(f"| {english} | {count:,} |")
    return lines


def _dead(corpus: Corpus) -> list[str]:
    """Every dead job, with its reason and its batch.

    ``A02`` requires each one to appear here by id, so this section is both the
    report and the thing that makes the check passable. Listed in full and never
    truncated: a truncated list would silently fail the check for the entries
    below the cut, which is the opposite of what a cap is for.
    """
    jobs = _dead_jobs(corpus.queue) if corpus.queue is not None else []
    if not jobs:
        # An absent queue and an empty one say the same thing here, and they have
        # to. This report is committed, and CI regenerates it and fails if a byte
        # moved, so a section that reads differently depending on whether a
        # scratch directory happens to exist makes the freshness gate fire on
        # every run for a reason that has nothing to do with the corpus. It did.
        return ["## Dead entries", "", "Nothing in the dead letter queue."]
    lines = [
        "## Dead entries",
        "",
        f"{len(jobs)} jobs out of rungs, "
        f"{sum(_entries(one) for one in jobs):,} entries between them.",
        "",
        "| Job | File | Entries | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    for one in sorted(jobs, key=lambda job: job.id):
        where = str(one.payload.get("file", "unknown"))
        reason = one.error or "no reason recorded"
        lines.append(f"| `{one.id}` | {where} | {_entries(one):,} | {reason} |")
    return lines


def _routes(tallies: Sequence[Tally]) -> list[str]:
    """Calls and wall clock by route, from the runs' own records.

    Named as an absence when there has not been a run, on the same principle as
    the wall-clock estimate: a table of zeros looks measured and is not.
    """
    if not tallies:
        return ["## Routes", "", "No translation run on record."]
    whole = _merge(tallies)
    if not whole.by_route:
        return ["## Routes", "", "The runs on record made no calls."]
    lines = [
        "## Routes",
        "",
        f"{whole.calls:,} calls over {whole.seconds / 3600:.1f} hours.",
        "",
        "| Route | Calls | Share |",
        "| --- | ---: | ---: |",
    ]
    total = sum(whole.by_route.values())
    for route, count in sorted(whole.by_route.items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"| {route} | {count:,} | {count / total:.1%} |")
    return lines


def _merge(tallies: Sequence[Tally]) -> Tally:
    """Every run on record as one tally."""
    whole = Tally(run=f"{len(tallies)} runs")
    for one in tallies:
        whole.batches += one.batches
        whole.accepted += one.accepted
        whole.refused += one.refused
        whole.rejected += one.rejected
        whole.dead += one.dead
        whole.calls += one.calls
        whole.seconds += one.seconds
        for name, count in one.by_rule.items():
            whole.by_rule[name] = whole.by_rule.get(name, 0) + count
        for rung, count in one.by_attempt.items():
            whole.by_attempt[rung] = whole.by_attempt.get(rung, 0) + count
        for route, count in one.by_route.items():
            whole.by_route[route] = whole.by_route.get(route, 0) + count
    return whole


def _dead_jobs(queue: Path) -> list[Job]:
    target = queue / str(Stage.TRANSLATE) / str(State.DEAD)
    if not target.exists():
        return []
    found = []
    for path in sorted(target.glob("*.json")):
        try:
            found.append(Job.from_json(path.read_text(encoding="utf-8")))
        except OSError, ValueError:
            continue
    return found


def _entries(job: Job) -> int:
    """How many entries died with one job."""
    found = job.payload.get("segments")
    return len(found) if isinstance(found, list) else 0


def _tally[K](corpus: Corpus, key: Callable[[str], K]) -> dict[K, Count]:
    """Counts grouped by whatever ``key`` makes of a corpus-relative path."""
    tally: dict[K, Count] = {}
    for one in corpus.catalogs:
        count = tally.setdefault(key(corpus.relative(one.path)), Count())
        for entry in one:
            if not entry.is_header:
                count.add(entry)
    return tally


def _sum(counts: Iterable[Count]) -> Count:
    whole = Count()
    for one in counts:
        whole.entries.update(one.entries)
        whole.words.update(one.words)
    return whole


def _named(tiers: dict[int, Count]) -> dict[str, Count]:
    return {f"{number}": one for number, one in sorted(tiers.items())}


def _table(heading: str, rows: dict[str, Count]) -> list[str]:
    """One row per section, five columns and a total.

    Entries rather than words in the columns, with words alongside, because
    entries are what gets reviewed and words are what gets read. Both are
    quoted in different places and a table with one of them invites the other
    to be estimated from it.
    """
    lines = [
        f"| {heading} | Entries | Words | "
        + " | ".join(one.value.title() for one in COLUMNS)
        + " |",
        "| --- | ---: | ---: |" + " ---: |" * len(COLUMNS),
    ]
    for name, count in rows.items():
        cells = " | ".join(f"{count.entries[one]:,}" for one in COLUMNS)
        lines.append(f"| {name} | {count.total:,} | {sum(count.words.values()):,} | {cells} |")
    whole = _sum(rows.values())
    cells = " | ".join(f"{whole.entries[one]:,}" for one in COLUMNS)
    lines.append(f"| **Total** | {whole.total:,} | {sum(whole.words.values()):,} | {cells} |")
    return lines
