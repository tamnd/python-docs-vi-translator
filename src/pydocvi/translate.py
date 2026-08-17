"""From an answer to translations, and from a refusal to the next attempt.

Two halves. The first is pure: :func:`read` turns one answer into accepted
entries and refusals, :func:`advice` says what to tell the model about them, and
:func:`again` says what to send next. Not one of those calls a model, opens a
file or looks at a clock, which is why the retry ladder can be tested exhaustively
without a fleet.

The second half is :class:`Run`, which is the stage: it holds the memory, the
glossary and the queue, and hands the worker a ``build`` and a ``handle``. The
worker owns routes, leases, concurrency and cancellation and knows nothing about
placeholders; this owns placeholders and knows nothing about routes.

The rule the first half exists to enforce is that failure is per entry. A batch
where entry 17 lost a marker gives back 39 translations and one refusal. At a
minute and a half a call, throwing away 39 good entries because of one bad one
produces no better corpus and a much longer run. The three exceptions come from
the invariants themselves: alignment is a property of the whole answer, and
narration means the model was not doing the task at all.

Two attempt counters run side by side and they count different things. The
queue's counts claims, and a claim dies to a dropped tunnel or a Ctrl-C. The
ladder's counts rungs, and a rung dies to a translation that broke a rule. A
batch that has burned three rungs has not run out of claims, it has run out of
things left to try, which is why exhausting the ladder buries the job rather
than releasing it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

from pydocvi import invariants, parse, render
from pydocvi.batch import Batch, Item, batch_id
from pydocvi.catalog import SegmentId
from pydocvi.client import Answer
from pydocvi.glossary import Glossary
from pydocvi.memory import Memory, Segment
from pydocvi.queue import Job, Queue, Stage, job_id
from pydocvi.segment import restore

log = logging.getLogger(__name__)

#: Batches between writes of the memory. Nine hours of calls held in a
#: dictionary is nine hours a Ctrl-C can throw away, and the write is
#: milliseconds against a call that is minutes.
SAVE_EVERY = 10


class Attempt(IntEnum):
    """Where an entry is on the ladder.

    Three rungs and then dead. A fourth attempt has never fixed anything on this
    transport; what fixes it is somebody reading the trace.
    """

    FIRST = 1
    NAMED = 2
    ALONE = 3
    DEAD = 4


#: What to say about each rule on the second rung, addressed to the model.
#: Naming the failure is the whole difference between attempt 2 and attempt 1,
#: and a generic "try again" is worth nothing: the same prompt against the same
#: session returns the same answer.
ADVICE = {
    "P01": "came back with a marker wrong. Every ⟦n⟧ in a string must appear "
    "exactly once in your translation of it, copied character for character.",
    "P02": "came back with a marker wrong. Every ⟦n⟧ in a string must appear "
    "exactly once in your translation of it, copied character for character.",
    "P04": "lost the space or the newline at the start or the end. Keep both "
    "exactly as they are in the original.",
    "P05": "came back empty or unchanged. Translate the string into Vietnamese.",
    "P06": "came back with a note or an apology in it. Give the translation and nothing else.",
    "P07": "came back inside a code fence. Give the translation and nothing else.",
    "P08": "came back in English. Write it in Vietnamese, with the tone marks.",
    "P09": "gained or lost a format specifier. Every %s, %(name)s, {} and "
    "{name} in a string must appear in your translation of it, and nothing "
    "shaped like one that was not there already.",
}

#: For a rule with nothing specific to say, which should stay empty.
GENERIC = "came back wrong."


@dataclass(frozen=True, slots=True, kw_only=True)
class Accepted:
    """One translation that passed every hard check, markup back in place."""

    segment: SegmentId
    index: int
    msgid: str
    msgstr: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Refused:
    """One entry that did not, named by the rule that refused it."""

    segment: SegmentId
    index: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: entry {self.index}: {self.detail}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """What one answer produced."""

    batch: str
    accepted: tuple[Accepted, ...] = ()
    refused: tuple[Refused, ...] = ()
    rejected: str = ""

    #: The rule behind ``rejected``, kept apart from the sentence describing it.
    #: The retry has to name the failure to the model and reading the rule back
    #: out of a formatted string would be a parser for this module's own output.
    rejected_rule: str = ""

    @property
    def whole_batch_refused(self) -> bool:
        return bool(self.rejected)

    @property
    def rate(self) -> float:
        """The share of the batch that came back usable.

        Counted in entries and not in refusals. One entry can break two rules and
        is reported twice, because the second fact changes what the retry says,
        but it is still one entry of the batch and a rate that counted it twice
        would fall as the reporting got better.
        """
        total = len(self.accepted) + len(self.entries_refused)
        return len(self.accepted) / total if total else 0.0

    @property
    def entries_refused(self) -> frozenset[SegmentId]:
        """The entries behind the refusals, each once however many rules it broke."""
        return frozenset(one.segment for one in self.refused)


def read(batch: Batch, text: str) -> Outcome:
    """Read one answer against the batch that produced it.

    The order matters. Alignment is checked first because an answer whose
    indices are not exactly ``1..N`` may have every translation on the wrong
    entry, and an entry-by-entry check of that would report nine failures and
    hide the one fact that explains them.
    """
    answer = parse.parse(text, len(batch))
    for violation in invariants.check_answer(answer, len(batch)):
        return Outcome(batch=batch.id, rejected=str(violation), rejected_rule=violation.rule)

    accepted: list[Accepted] = []
    refused: list[Refused] = []
    for index, item in enumerate(batch.items, 1):
        translated = answer.entries[index]
        broken = invariants.check_entry(item.msgid, translated, item.protected.spans, index=index)
        fatal = next((one for one in broken if one.rejects_batch), None)
        if fatal is not None:
            return Outcome(batch=batch.id, rejected=str(fatal), rejected_rule=fatal.rule)
        if broken:
            refused.extend(
                Refused(segment=item.segment, index=index, rule=one.rule, detail=one.detail)
                for one in broken
            )
            continue
        accepted.append(
            Accepted(
                segment=item.segment,
                index=index,
                msgid=item.msgid,
                msgstr=restore(translated, item.protected.spans, original=item.msgid),
            )
        )

    return Outcome(batch=batch.id, accepted=tuple(accepted), refused=tuple(refused))


def advice(refused: Sequence[Refused]) -> str:
    """What to tell the model about the entries it got wrong.

    One sentence per rule that fired rather than one per entry, because thirty
    entries that all lost a marker are one mistake made thirty times, and a
    prompt that says so thirty times is a prompt about its own length.
    """
    rules = sorted({one.rule for one in refused})
    if not rules:
        return ""
    said = [f"Some strings in the last attempt {ADVICE.get(rule, GENERIC)}" for rule in rules]
    return "\n".join(said)


def again(batch: Batch, refused: Sequence[Refused], *, attempt: Attempt) -> list[Batch]:
    """The batches to send on the next rung, in order."""
    return retry(batch, _failed(batch, refused), attempt=attempt)


def retry(batch: Batch, items: Sequence[Item], *, attempt: Attempt) -> list[Batch]:
    """The batches to send on the next rung, from the items that need one.

    Rung 2 is the failed entries together, which is one call instead of thirty
    and keeps the neighbouring-paragraph context that made the first batch worth
    packing that way. Rung 3 is one entry per call, because an entry that has
    now failed twice is not failing for a reason its neighbours share, and the
    only thing left to give it is the whole call to itself.

    Taking items rather than refusals because a batch condemned by ``P03``,
    ``P06`` or ``P07`` has no per-entry refusals at all: the answer was wrong as
    a whole and every entry in it needs sending again.
    """
    if not items or attempt >= Attempt.DEAD:
        return []
    if attempt is Attempt.ALONE:
        return [_batch(batch.path, [item]) for item in items]
    return [_batch(batch.path, list(items))]


def _failed(batch: Batch, refused: Sequence[Refused]) -> list[Item]:
    """The items behind a set of refusals, in the batch's own order.

    By segment rather than by index, because the indices of the next attempt are
    not the indices of this one and an entry retried under the wrong number is
    a translation written onto another string.
    """
    wanted = {one.segment for one in refused}
    return [item for item in batch.items if item.segment in wanted]


def _batch(path: str, items: Sequence[Item]) -> Batch:
    """A retry batch, identified like any other.

    Content-addressed on the entries it holds, so the retry of a batch is a
    different id from the batch, and two runs that fail the same way file their
    traces under the same name.
    """
    return Batch(
        id=batch_id(path, [item.segment for item in items]),
        path=path,
        items=tuple(items),
    )


@dataclass(slots=True, kw_only=True)
class Tally:
    """Refusals by rule and by rung, which is what says whether the prompt works.

    Counted as the run goes rather than reconstructed from the corpus after it,
    because a refusal that was fixed on rung 2 leaves no trace in the corpus at
    all and it is exactly the number that says the ladder is earning its calls.
    """

    batches: int = 0
    accepted: int = 0
    refused: int = 0
    rejected: int = 0
    dead: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    by_attempt: dict[int, int] = field(default_factory=dict)

    def record(self, outcome: Outcome, *, attempt: Attempt) -> None:
        self.batches += 1
        self.accepted += len(outcome.accepted)
        if outcome.whole_batch_refused:
            self.rejected += 1
            self._count(outcome.rejected_rule, attempt, 1)
            return
        self.refused += len(outcome.entries_refused)
        for one in outcome.refused:
            self._count(one.rule, attempt, 1)

    def _count(self, rule: str, attempt: Attempt, many: int) -> None:
        self.by_rule[rule] = self.by_rule.get(rule, 0) + many
        self.by_attempt[int(attempt)] = self.by_attempt.get(int(attempt), 0) + many

    @property
    def rate(self) -> float:
        """The share of entries seen that came back usable, over the whole run."""
        total = self.accepted + self.refused
        return self.accepted / total if total else 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    """What a run would do, before it has spent anything.

    ``queued`` and ``known`` are separate because the difference is the whole
    value of a content-addressed queue: a second pass over a tier queues only
    what the first pass did not finish, and a plan that printed one number would
    hide that the other 3 000 batches are already done.
    """

    tier: int | None = None
    batches: int = 0
    entries: int = 0
    queued: int = 0
    known: int = 0

    def __str__(self) -> str:
        where = f"tier {self.tier}" if self.tier is not None else "the selection"
        known = f", {self.known:,} already known" if self.known else ""
        return (
            f"{where}: {self.batches:,} batches, {self.entries:,} entries, "
            f"{self.queued:,} queued{known}"
        )


class Run:
    """One translation run: the queue on one side, the memory on the other.

    Everything the worker needs from this stage is :meth:`build` and
    :meth:`handle`, and everything durable it produces is in the queue and the
    memory. Nothing is held in this object that a Ctrl-C would lose, which is
    what makes ``--resume`` a matter of pointing it at the same directory rather
    than a feature with its own state file.
    """

    def __init__(
        self,
        *,
        queue: Queue,
        memory: Memory,
        glossary: Glossary,
        batches: Sequence[Batch],
        run: str,
        prompt: str = render.TRANSLATE,
        save_every: int = SAVE_EVERY,
    ) -> None:
        self.queue = queue
        self.memory = memory
        self.glossary = glossary
        self.run = run
        self.prompt = prompt
        self.save_every = save_every
        self.tally = Tally()
        self._items = {item.segment: item for batch in batches for item in batch.items}
        self._since_save = 0

    def plan(
        self, batches: Sequence[Batch], *, tier: int | None = None, write: bool = True
    ) -> Plan:
        """Queue what is not already known, and say what that came to.

        ``write=False`` is what ``--dry-run`` needs. A dry run that queued the
        work would leave the queue in exactly the state a real run leaves it in,
        so the next command without the flag would find the estimate it was
        given already committed to, which is the opposite of what somebody asks
        a dry run for.
        """
        jobs = [self._job(batch, attempt=Attempt.FIRST) for batch in batches]
        queued = (
            self.queue.extend(jobs)
            if write
            else sum(1 for job in jobs if self.queue.locate(job.id) is None)
        )
        return Plan(
            tier=tier,
            batches=len(batches),
            entries=sum(len(batch) for batch in batches),
            queued=queued,
            known=len(jobs) - queued,
        )

    def untranslated(self, batches: Sequence[Batch]) -> list[Batch]:
        """The batches with at least one entry the memory does not already hold.

        A batch every entry of which is already translated is a call nobody
        needs to pay for. Whole batches rather than repacked leftovers, because
        repacking would change every batch id and orphan every trace filed under
        the old ones.
        """
        return [
            batch
            for batch in batches
            if any(item.segment not in self.memory for item in batch.items)
        ]

    def build(self, job: Job) -> render.Prompt:
        """The two messages for one job."""
        return render.render(
            self.batch(job),
            self.glossary,
            advice=str(job.payload.get("advice", "")),
            name=self.prompt,
        )

    async def handle(self, job: Job, answer: Answer) -> None:
        """Keep what passed, and queue the next rung for what did not."""
        batch = self.batch(job)
        attempt = _rung(job)
        outcome = read(batch, answer.text)
        self.tally.record(outcome, attempt=attempt)
        self._remember(outcome, batch=batch, answer=answer)

        if outcome.whole_batch_refused:
            log.warning(
                "batch refused whole",
                extra={"batch": batch.id, "attempt": int(attempt), "why": outcome.rejected},
            )
            failed = list(batch.items)
            note = _say(outcome.rejected_rule)
        else:
            failed = _failed(batch, outcome.refused)
            note = advice(outcome.refused)
        self._ladder(job, batch, failed, note=note, attempt=attempt)

    def batch(self, job: Job) -> Batch:
        """Rebuild a job's batch from its payload.

        The payload carries the file and the segment ids and never the entries
        themselves. That is what makes a run resumable across a restart: the
        items are recovered from the corpus on disk, which is the same corpus,
        rather than from a serialised copy that would go stale the moment
        upstream moved.
        """
        path = str(job.payload["file"])
        segments = _segments(job)
        return Batch(
            id=batch_id(path, segments),
            path=path,
            items=tuple(self._items[one] for one in segments),
        )

    def save(self, *, force: bool = False) -> None:
        """Write the memory, every so often and at the end.

        Nine hours of calls held in a dictionary is nine hours a Ctrl-C can
        throw away. The store is a few tens of megabytes and writing it is
        milliseconds against a call that is minutes, so the interval is small.
        """
        self._since_save += 1
        if force or self._since_save >= self.save_every:
            self.memory.save()
            self._since_save = 0

    def _remember(self, outcome: Outcome, *, batch: Batch, answer: Answer) -> None:
        stored = self.memory.extend(
            Segment(
                id=one.segment,
                msgid=one.msgid,
                msgstr=one.msgstr,
                source="machine",
                model=answer.answered_by,
                prompt=render.fingerprint(self.prompt),
                glossary=self.glossary.version,
                batch=batch.id,
                run=self.run,
            )
            for one in outcome.accepted
        )
        log.info(
            "batch read",
            extra={
                "batch": batch.id,
                "accepted": len(outcome.accepted),
                "stored": stored,
                "refused": len(outcome.entries_refused),
            },
        )
        self.save()

    def _ladder(
        self, job: Job, batch: Batch, failed: Sequence[Item], *, note: str, attempt: Attempt
    ) -> None:
        """Queue the next rung, or bury what has run out of rungs."""
        if not failed:
            return
        nxt = Attempt(attempt + 1)
        following = retry(batch, failed, attempt=nxt)
        if not following:
            self.tally.dead += len(failed)
            self.queue.bury(job, error=f"{len(failed)} entries refused after {int(attempt)} rungs")
            return
        self.queue.extend([self._job(one, attempt=nxt, advice=note) for one in following])

    def _job(self, batch: Batch, *, attempt: Attempt, advice: str = "") -> Job:
        payload: dict[str, object] = {
            "file": batch.path,
            "segments": [str(item.segment) for item in batch.items],
            "attempt": int(attempt),
            "advice": advice,
            "prompt": render.fingerprint(self.prompt),
            "glossary": self.glossary.version,
        }
        return Job(id=job_id(Stage.TRANSLATE, payload), stage=Stage.TRANSLATE, payload=payload)


def _say(rule: str) -> str:
    """The advice for one rule, when the rule condemned the whole batch."""
    return f"Some strings in the last attempt {ADVICE.get(rule, GENERIC)}"


def _segments(job: Job) -> list[SegmentId]:
    """The segment ids out of a payload that came off a disk nobody validated.

    A job file is JSON on a filesystem a person can edit, so the types in it are
    whatever is there rather than whatever was written. Reading it defensively
    costs one function and saves a run that dies on hour eight with a
    ``TypeError`` in a worker task.
    """
    found = job.payload.get("segments")
    if not isinstance(found, list):
        raise JobError(f"job {job.id} has no list of segments")
    return [str(one) for one in found]


def _rung(job: Job) -> Attempt:
    """Which rung a job is on, defaulting to the first."""
    found = job.payload.get("attempt", Attempt.FIRST)
    if not isinstance(found, int) or found not in set(Attempt):
        raise JobError(f"job {job.id} is on no rung this ladder has: {found!r}")
    return Attempt(found)


class JobError(ValueError):
    """A job file that this stage cannot make a batch out of."""
