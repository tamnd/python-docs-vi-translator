"""``A01`` to ``A05``: what is missing, and whether anybody wrote down why.

The group about absence rather than about correctness. Every other check reads
an entry and asks whether it is right; these ask about the entries that are not
there, which is the half of the corpus nothing else in the audit can see.

``A02`` is the one that earns the group. A job that died is the pipeline's only
record that a string was tried and could not be produced, and the failure mode
it guards against is not a wrong translation but a silent one: a run that
reports ``dead 0`` while 28 entries went nowhere. That happened, it took a
second run and a queue inspection to notice, and this is where it would have
been a line in a report instead.
"""

import re
from collections.abc import Iterator, Sequence

from pydocvi import batch, queue
from pydocvi.audit.model import Corpus, Finding, Group, Registry, counts

registry = Registry()
check = registry.check

#: Above this share of a file's entries, dead jobs stop being individual
#: failures and start being something wrong with the file: an encoding, a
#: pathological entry length, a construct the protector does not handle.
DEAD_CEILING = 0.01

#: Above this share of a batch refused, the batch is worth naming in the report
#: rather than counting. A batch losing one entry in five is not bad luck.
REFUSAL_CEILING = 0.20


@check("A01", Group.AVAILABILITY, hard=True, title="the coverage report is current")
def a01_coverage_is_current(corpus: Corpus) -> Iterator[Finding]:
    """Per-tier coverage in ``reports/coverage.md`` matches a recount.

    The same rule as ``S01`` applied to a different file, and it is here for the
    same reason: coverage is the number this project is judged by, it is quoted
    in the README and in every milestone comment, and a stale one is a confident
    claim about a corpus that has moved on.
    """
    if corpus.coverage is None:
        return
    recorded = counts(corpus.coverage)
    if recorded is None:
        yield Finding(
            check="A01",
            path="reports/coverage.md",
            detail="no machine-readable counts, so nothing can be checked against it",
        )
        return
    for tier, (done, total) in sorted(_by_tier(corpus).items()):
        claimed = recorded.get(str(tier))
        if claimed is None:
            yield Finding(
                check="A01", path="reports/coverage.md", detail=f"tier {tier} is not reported"
            )
        elif claimed != done:
            yield Finding(
                check="A01",
                path="reports/coverage.md",
                detail=f"tier {tier} reported as {claimed:,} of {total:,}, a recount finds {done:,}",
            )


@check("A02", Group.AVAILABILITY, hard=True, title="every dead job is accounted for")
def a02_dead_jobs_are_reported(corpus: Corpus) -> Iterator[Finding]:
    """A job in the dead letter queue is named in ``reports/quality.md``.

    Dying is allowed. Some entries cannot be translated by this pipeline and
    saying so is a real answer. Dying without a line in a report is not: it
    leaves an entry that nobody translated, nobody refused and nobody can find,
    and the corpus reads as though the string was never there.
    """
    if corpus.queue is None:
        return
    quality = corpus.quality or ""
    for one in _dead(corpus):
        if one.id in quality:
            continue
        yield Finding(
            check="A02",
            path="reports/quality.md",
            detail=f"job {one.id} died and is not in the report",
            english=str(one.payload.get("file", "")),
            got=one.error or "no reason recorded",
        )


@check("A03", Group.AVAILABILITY, hard=True, title="the memory and the catalogs agree")
def a03_no_untranslated_with_a_record(corpus: Corpus) -> Iterator[Finding]:
    """No entry has an empty ``msgstr`` and a machine record in the memory.

    That combination means ``apply`` and the memory disagree about what has been
    done, and it is the shape of a lost write: the translation exists, it was
    paid for, and the catalog it belongs in never got it. Left alone it reads
    downstream as an entry still to do, and the next run pays for it again.
    """
    if corpus.memory is None:
        return
    for one in corpus.catalogs:
        for entry in one:
            if entry.msgstr or entry.is_header:
                continue
            known = corpus.memory.lookup(entry.msgid, entry.msgctxt)
            if known is None or known.source == "human":
                continue
            yield Finding(
                check="A03",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"the memory has a {known.source} translation and the catalog is empty",
                english=entry.msgid,
                got=known.msgstr,
                segment=entry.id,
            )


@check("A04", Group.AVAILABILITY, hard=False, title="dead entries per file below a ceiling")
def a04_dead_per_file(corpus: Corpus) -> Iterator[Finding]:
    """No file has more than one entry in a hundred dead.

    Per file rather than corpus-wide, because that is the number that separates
    the two explanations. Scattered deaths are individual hard strings and are
    expected; a hundred in one file is something about that file, and averaging
    them into a corpus of 87 008 hides it completely.
    """
    if corpus.queue is None:
        return
    sizes = {corpus.relative(one.path): len(one) for one in corpus.catalogs}
    tally: dict[str, int] = {}
    for one in _dead(corpus):
        where = str(one.payload.get("file", ""))
        tally[where] = tally.get(where, 0) + len(_segments(one))
    for where, count in sorted(tally.items()):
        total = sizes.get(where, 0)
        if total and count / total > DEAD_CEILING:
            yield Finding(
                check="A04",
                path=where,
                detail=f"{count:,} of {total:,} entries dead, over the {DEAD_CEILING:.0%} ceiling",
            )


@check("A05", Group.AVAILABILITY, hard=False, title="heavily refused batches are named")
def a05_refusal_rates_are_reported(corpus: Corpus) -> Iterator[Finding]:
    """A batch that lost more than a fifth of its entries is named in the report.

    A refusal rate is the pipeline criticising itself, and a high one is
    information about the prompt rather than about the entries. Counting these
    into a corpus-wide average is how a prompt that fails on one kind of entry
    stays undiagnosed: the average moves by a fraction of a point and the batch
    that produced it is never looked at.
    """
    if corpus.queue is None:
        return
    quality = corpus.quality or ""
    for one in _dead(corpus):
        segments = _segments(one)
        refused = _refused(one.error or "")
        if not segments or refused is None:
            continue
        rate = refused / len(segments)
        if rate > REFUSAL_CEILING and one.id not in quality:
            yield Finding(
                check="A05",
                path=str(one.payload.get("file", "")),
                detail=(
                    f"job {one.id} refused {refused} of {len(segments)} entries "
                    f"({rate:.0%}) and is not named in the report"
                ),
                got=one.error or "",
            )


def _dead(corpus: Corpus) -> list[queue.Job]:
    """Every job in every stage's dead letter queue."""
    if corpus.queue is None:
        return []
    return [job for one in queue.queues(corpus.queue) for job in one.jobs(queue.State.DEAD)]


def _segments(job: queue.Job) -> Sequence[object]:
    found = job.payload.get("segments")
    return found if isinstance(found, list) else ()


def _refused(error: str) -> int | None:
    """How many entries a dead job's reason says were refused.

    Read out of the message rather than stored as a field, because the message
    is what a reviewer reads and a number that disagreed with it would be worse
    than no number at all.
    """
    match = re.search(r"(\d+) entries refused", error)
    return int(match.group(1)) if match else None


def _by_tier(corpus: Corpus) -> dict[int, tuple[int, int]]:
    """Translated and total entries per tier, recounted off the catalogs."""
    tally: dict[int, tuple[int, int]] = {}
    for one in corpus.catalogs:
        tier = batch.tier_of(corpus.relative(one.path))
        done, total = tally.get(tier, (0, 0))
        for entry in one:
            if entry.is_header:
                continue
            total += 1
            done += bool(entry.msgstr)
        tally[tier] = (done, total)
    return tally
