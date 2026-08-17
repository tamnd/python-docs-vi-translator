"""Asking a model about candidate terms, and refusing most of what comes back.

A batch of forty candidates goes out with the definition sentence beside each
one and comes back as forty short phrases. Six checks then run over the answer,
and a line that fails any of them is dropped and reported rather than repaired.
Dropping is cheap: it is one term to ask about again or to write by hand, and a
term written by hand is what the glossary wants anyway.

The single most important thing in this module is not any of the checks. It is
that the model is given a way to say it does not know, and told that saying so
is the wanted answer. A model with no way out invents, and an invented rendering
is precisely the failure that survives every mechanical check: one plausible
phrase, in the right script, that no Vietnamese programmer uses.
"""

import asyncio
import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from pydocvi import parse
from pydocvi.client import Completions
from pydocvi.glossary import KEEP, Rejection, Term
from pydocvi.invariants import vietnamese
from pydocvi.mine import Candidate
from pydocvi.routes import Route

log = logging.getLogger(__name__)

#: Candidates in one call, the same cap the translation batches use and for the
#: same reason: past about forty items a numbered list stops being a list the
#: model transcribes and starts being one it summarises.
BATCH_SIZE = 40

#: The longest a rendering may be. Six words is already generous for a term and
#: an answer longer than that is a definition, which is not what was asked for.
PHRASE_WORDS = 6

#: What a model says when it does not know. Spelled in capitals so that it
#: cannot collide with a rendering, and named in the prompt twice.
UNSURE = "UNSURE"

#: Separates the repeated English from the rendering on an answer line.
SEPARATOR = "="

#: Sentence-final punctuation, which a phrase does not have.
_FINAL = re.compile(r"[.!?;:,]\s*$")

_WORD = re.compile(r"\w+", re.UNICODE)

PROMPT = """\
You are compiling a Vietnamese terminology list for the Python documentation.

For each numbered English term below, give the Vietnamese term that Vietnamese \
Python programmers actually use. Answer with one line per number, in this form:

<number>. <the English term, copied exactly> {separator} <the answer>

The answer is one of three things:

1. A Vietnamese phrase. At most {words} words, no final punctuation, with the \
tone marks written.
2. The word {keep}, if Vietnamese programmers use the English word itself. \
Much of Python's vocabulary is like this and {keep} is a normal answer.
3. The word {unsure}, if you do not know what Vietnamese Python programmers \
call this. {unsure} is a wanted answer and it is better than a guess. A \
plausible phrase that nobody uses is the one mistake that cannot be caught \
later.

Give no preamble, no explanation and no line that is not one of the numbered \
answers.

Terms:

{terms}
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Batch:
    """One call's worth of candidates."""

    index: int
    candidates: tuple[Candidate, ...]

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def id(self) -> str:
        """Content addressed, so the same batch after a re-run is the same batch."""
        payload = "\0".join(candidate.en for candidate in self.candidates)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def batches(candidates: Sequence[Candidate], *, size: int = BATCH_SIZE) -> list[Batch]:
    """Cut the candidate list into calls, in the order it arrived.

    In order, which means by trust and then alphabetically, so the first call
    asks about the terms people already rendered and the last asks about the
    ones a machine contradicted itself on. A run stopped halfway has therefore
    curated the half worth curating.
    """
    return [
        Batch(index=number, candidates=tuple(candidates[start : start + size]))
        for number, start in enumerate(range(0, len(candidates), size), start=1)
    ]


def prompt(batch: Batch) -> str:
    """The call.

    The definition sentence goes in beside every candidate that has one, because
    a model asked to render "annotation" bare renders the English word and a
    model shown CPython's definition of it renders the Python concept.
    """
    lines = []
    for number, candidate in enumerate(batch.candidates, start=1):
        lines.append(f"{number}. {candidate.en}")
        if candidate.definition:
            lines.append(f"   context: {candidate.definition}")
        if candidate.seen:
            lines.append(f"   already rendered as: {', '.join(candidate.seen)}")
    return PROMPT.format(
        separator=SEPARATOR,
        words=PHRASE_WORDS,
        keep=KEEP,
        unsure=UNSURE,
        terms="\n".join(lines),
    )


def prompt_id() -> str:
    """A hash of the template, stamped onto every row it produced.

    A rendering is an answer to a question, and changing the question makes the
    answers worth re-asking. Without this the only record of which prompt
    produced a term is the date, which is not a record.
    """
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """What one curation call produced.

    Three outcomes rather than two. ``accepted`` is a row to review, ``declined``
    is the model saying it does not know, and ``dropped`` is a line that broke a
    check. Declining is not a failure and is counted apart from one, because a
    run where the model declined 60 terms is a run that behaved well and a
    report that called those 60 failures would get the prompt changed.
    """

    batch: str
    index: int = 0
    accepted: tuple[Term, ...] = ()
    declined: tuple[str, ...] = ()
    dropped: tuple[Rejection, ...] = ()

    @property
    def answered(self) -> int:
        return len(self.accepted) + len(self.declined)

    @property
    def unanswered(self) -> bool:
        """Whether the call produced no answer at all, as against a poor one.

        A batch that comes back like this is worth sending to another host,
        because nothing about it is the batch's fault. One that came back with
        rows dropped under a ``G`` rule is not worth moving, because the second
        host would drop the same rows for the same reason.
        """
        return bool(self.dropped) and all(one.rule == "call" for one in self.dropped)

    @classmethod
    def failed(cls, batch: Batch, detail: str) -> Self:
        """A call that never produced an answer, as a reply rather than a raise.

        Every candidate in the batch is dropped under the rule ``call``, so the
        terms a failure cost are listed by name in the report instead of going
        missing between the candidate count and the accepted count.
        """
        return cls(
            batch=batch.id,
            index=batch.index,
            dropped=tuple(
                Rejection(rule="call", en=candidate.en, detail=detail)
                for candidate in batch.candidates
            ),
        )


def read(batch: Batch, text: str) -> Reply:
    """Parse one answer and run ``G-a`` through ``G-d`` over every line.

    Numbering is parsed by the same code that parses a translation batch, so a
    model that repeats an index or invents one is caught by the module that
    already knows how, rather than by a second parser that has to be kept in
    agreement with the first.
    """
    answer = parse.parse(text, len(batch))
    accepted: list[Term] = []
    declined: list[str] = []
    dropped: list[Rejection] = [
        Rejection(rule="format", en=_named(batch, problem.index), detail=str(problem))
        for problem in answer.problems
    ]

    for index, body in sorted(answer.entries.items()):
        candidate = batch.candidates[index - 1]
        term, rejection = _row(candidate, body)
        if rejection is not None:
            dropped.append(rejection)
        elif term is None:
            declined.append(candidate.en)
        else:
            accepted.append(term)

    return Reply(
        batch=batch.id,
        index=batch.index,
        accepted=tuple(accepted),
        declined=tuple(declined),
        dropped=tuple(dropped),
    )


def _named(batch: Batch, index: int | None) -> str:
    if index is None or not 1 <= index <= len(batch):
        return "answer"
    return batch.candidates[index - 1].en


def _row(candidate: Candidate, body: str) -> tuple[Term | None, Rejection | None]:
    """One answer line, checked. A term, a decline, or a rejection."""
    english, separator, rendering = body.partition(SEPARATOR)
    if not separator:
        return None, Rejection(rule="G-a", en=candidate.en, detail=f"no {SEPARATOR!r} in {body!r}")

    failure = _ga(candidate, english) or _gb(candidate, rendering)
    if failure is not None:
        return None, failure

    answer = rendering.strip()
    if answer == UNSURE:
        return None, None
    if answer == KEEP:
        return Term(en=candidate.en, vi=candidate.en, keep_en=True), None

    failure = _gc(candidate, answer) or _gd(candidate, answer)
    if failure is not None:
        return None, failure
    return Term(en=candidate.en, vi=answer), None


def _ga(candidate: Candidate, english: str) -> Rejection | None:
    """``G-a``: the English is repeated byte-identically.

    So that the answer is provably about the term that was asked. Without this
    an off-by-one in the model's numbering silently attaches every rendering to
    the term above it, and the result is a glossary that is wrong in a way no
    later check can see, because every row in it is a real Vietnamese phrase.
    """
    if english.strip() == candidate.en:
        return None
    return Rejection(
        rule="G-a",
        en=candidate.en,
        detail=f"the answer repeats it as {english.strip()!r}",
    )


def _gb(candidate: Candidate, rendering: str) -> Rejection | None:
    """``G-b``: one phrase, not a sentence."""
    answer = rendering.strip()
    if not answer:
        return Rejection(rule="G-b", en=candidate.en, detail="the answer is empty")
    if "\n" in rendering.strip("\n"):
        return Rejection(rule="G-b", en=candidate.en, detail="the answer runs over lines")
    if _FINAL.search(answer):
        return Rejection(rule="G-b", en=candidate.en, detail=f"{answer!r} ends a sentence")
    if len(answer.split()) > PHRASE_WORDS:
        return Rejection(
            rule="G-b",
            en=candidate.en,
            detail=f"{answer!r} is {len(answer.split())} words, the cap is {PHRASE_WORDS}",
        )
    return None


def _gc(candidate: Candidate, answer: str) -> Rejection | None:
    """``G-c``: written in Vietnamese, or explicitly kept in English.

    Vietnamese without tone marks is not Vietnamese, and a rendering with none
    is either the English word again, in which case the answer was the
    keep-English one, or a transliteration nobody writes.
    """
    if vietnamese(answer):
        return None
    return Rejection(
        rule="G-c",
        en=candidate.en,
        detail=f"{answer!r} has no tone mark and is not the keep-English answer",
    )


def _gd(candidate: Candidate, answer: str) -> Rejection | None:
    """``G-d``: not the English with the words in a different order.

    "context manager" coming back as "manager context" is the model having
    answered the shape of the question rather than the question. It passes
    ``G-a``, it passes ``G-b``, and only this rule sees it.
    """
    if _words(answer) != _words(candidate.en):
        return None
    return Rejection(
        rule="G-d",
        en=candidate.en,
        detail=f"{answer!r} is the English rearranged",
    )


def _words(text: str) -> frozenset[str]:
    """The words of a phrase, folded and stripped of marks.

    Stripped so that a rendering that only added tone marks to the English is
    still recognised as the English. Adding marks to "manager" does not make it
    a Vietnamese word, it makes it a word that passes ``G-c``.
    """
    bare = "".join(
        character
        for character in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(character)
    )
    return frozenset(_WORD.findall(bare))


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """Every reply from one curation run, added up.

    ``declined`` is reported beside ``accepted`` rather than folded into the
    failures, and the ratio between them is the number to watch. A run that
    declines nothing is a run where the model has stopped telling the truth
    about what it knows.
    """

    accepted: tuple[Term, ...] = ()
    declined: tuple[str, ...] = ()
    dropped: tuple[Rejection, ...] = ()
    batches: int = 0

    @property
    def asked(self) -> int:
        return len(self.accepted) + len(self.declined) + len(self.dropped)

    @property
    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.dropped:
            counts[rejection.rule] = counts.get(rejection.rule, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def kept(self) -> int:
        return sum(1 for term in self.accepted if term.keep_en)


def report(outcome: Outcome, *, prompt: str) -> str:
    """The Markdown a run leaves behind.

    The declined list is written out in full and the accepted list is not. The
    terms a model would not guess at are the ones a person has to write by hand,
    and a count of them is not a work list.
    """
    lines = [
        "# Glossary curation",
        "",
        f"Prompt `{prompt}`, {outcome.batches} calls, {outcome.asked:,} terms asked about.",
        "",
        "| outcome | terms |",
        "| --- | ---: |",
        f"| accepted | {len(outcome.accepted):,} |",
        f"| of those, keep-English | {outcome.kept:,} |",
        f"| declined | {len(outcome.declined):,} |",
        f"| dropped | {len(outcome.dropped):,} |",
        "",
    ]
    if outcome.by_rule:
        lines += ["## Dropped by rule", "", "| rule | terms |", "| --- | ---: |"]
        lines += [f"| `{rule}` | {count:,} |" for rule, count in outcome.by_rule.items()]
        lines.append("")
    if outcome.declined:
        lines += [
            "## Declined",
            "",
            "Terms the model would not guess at. These are the ones to write by hand.",
            "",
        ]
        lines += [f"- {en}" for en in sorted(outcome.declined)]
        lines.append("")
    return "\n".join(lines)


async def ask(
    client: Completions,
    routes: Sequence[Route],
    made: Sequence[Batch],
    *,
    on_reply: Callable[[Reply], None] | None = None,
) -> list[Reply]:
    """Send every batch to the fleet and read what comes back.

    Curation is a hundred-odd calls rather than the tens of thousands a
    translation pass makes, so it dispatches straight from a queue of batches
    instead of going through the durable work queue.

    A batch whose route will not answer is offered to the other routes before
    it is given up on. Only when every route has refused it does it come back
    as a reply with every candidate dropped under ``call``, which reads the same
    in the report as any other failure and leaves those terms to the next run.
    """
    if not routes:
        raise ValueError("no routes to curate with")

    pending: asyncio.Queue[Batch] = asyncio.Queue()
    for one in made:
        pending.put_nowait(one)
    replies: list[Reply] = []

    async def consume(route: Route) -> None:
        # This route first, the rest in rank order behind it, so a batch only
        # moves after the host that was chosen for it has actually refused it.
        order = (route, *(other for other in routes if other.name != route.name))
        while True:
            try:
                batch = pending.get_nowait()
            except asyncio.QueueEmpty:
                return
            reply = await _one(client, order, batch)
            replies.append(reply)
            if on_reply is not None:
                on_reply(reply)

    async with asyncio.TaskGroup() as group:
        for route in routes:
            for _ in range(route.concurrency):
                group.create_task(consume(route))
    return sorted(replies, key=lambda reply: reply.index)


async def _one(client: Completions, routes: Sequence[Route], batch: Batch) -> Reply:
    """One batch, moved to another host when the first will not answer.

    The client's own retries stay on one route on purpose, because a host that
    is merely loaded does answer when asked again. This is the other failure: a
    host that has stopped answering at all, where all three attempts land in the
    same hole. The first full run over the real fleet lost 80 terms that way,
    two whole batches, while two other hosts sat idle with nothing to do.

    Only ``unanswered`` replies move. A reply that came back with rows dropped
    under a ``G`` rule is the model's answer and is kept as it is.
    """
    reply = Reply.failed(batch, "no route tried")
    for route in routes:
        reply = await _call(client, route, batch)
        if not reply.unanswered:
            return reply
        log.warning(
            "batch unanswered, moving it to another route",
            extra={"batch": batch.id, "route": route.name},
        )
    return reply


async def _call(client: Completions, route: Route, batch: Batch) -> Reply:
    """One call, with a failure turned into a reply rather than an exception."""
    try:
        answer = await client.complete(route, prompt(batch))
    except Exception as error:  # one bad call is not a bad run
        return Reply.failed(batch, f"{type(error).__name__}: {error}")
    if answer.empty:
        return Reply.failed(batch, f"{route.name} returned nothing")
    return read(batch, answer.text)


def collect(replies: Iterable[Reply]) -> Outcome:
    """Add up a run's replies, keeping the first answer for a repeated term."""
    accepted: dict[str, Term] = {}
    declined: list[str] = []
    dropped: list[Rejection] = []
    batches = 0
    for reply in replies:
        batches += 1
        for term in reply.accepted:
            accepted.setdefault(term.en, term)
        declined.extend(reply.declined)
        dropped.extend(reply.dropped)
    return Outcome(
        accepted=tuple(accepted.values()),
        declined=tuple(declined),
        dropped=tuple(dropped),
        batches=batches,
    )
