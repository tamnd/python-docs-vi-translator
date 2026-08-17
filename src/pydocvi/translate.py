"""From an answer to translations, and from a refusal to the next attempt.

Nothing here calls a model or touches a file. One function reads an answer into
accepted entries and refusals, and another says what to send next. The calling
loop lives in the worker, which already knows about routes, leases and
cancellation and should not also know what a placeholder is.

The rule this module exists to enforce is that failure is per entry. A batch
where entry 17 lost a marker gives back 39 translations and one refusal. At a
minute and a half a call, throwing away 39 good entries because of one bad one
produces no better corpus and a much longer run. The three exceptions come from
the invariants themselves: alignment is a property of the whole answer, and
narration means the model was not doing the task at all.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from pydocvi import invariants, parse
from pydocvi.batch import Batch, Item, batch_id
from pydocvi.catalog import SegmentId
from pydocvi.segment import restore

log = logging.getLogger(__name__)


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
        return Outcome(batch=batch.id, rejected=str(violation))

    accepted: list[Accepted] = []
    refused: list[Refused] = []
    for index, item in enumerate(batch.items, 1):
        translated = answer.entries[index]
        broken = invariants.check_entry(item.msgid, translated, item.protected.spans, index=index)
        fatal = next((one for one in broken if one.rejects_batch), None)
        if fatal is not None:
            return Outcome(batch=batch.id, rejected=str(fatal))
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
    """The batches to send on the next rung, in order.

    Rung 2 is the failed entries together, which is one call instead of thirty
    and keeps the neighbouring-paragraph context that made the first batch worth
    packing that way. Rung 3 is one entry per call, because an entry that has
    now failed twice is not failing for a reason its neighbours share, and the
    only thing left to give it is the whole call to itself.
    """
    items = _failed(batch, refused)
    if not items or attempt >= Attempt.DEAD:
        return []
    if attempt is Attempt.ALONE:
        return [_batch(batch.path, [item]) for item in items]
    return [_batch(batch.path, items)]


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
