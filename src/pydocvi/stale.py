"""What a change invalidated.

Three things can make a stored translation wrong, and they are worth separating
because they cost different amounts to fix.

Upstream changed the English
    The segment id is a hash of the string, so the old translation is attached
    to a string that no longer exists. Nothing to detect: the new string simply
    has no entry in the memory. The old one is orphaned and kept.

The glossary changed
    A term was added, removed or rendered differently. Every stored translation
    of an English string containing that term is now against a terminology
    decision that has been overturned. This is the case worth being precise
    about: a version bump that changes one term in a corpus of 87 008 entries
    typically affects a few hundred, and re-queueing a few hundred is what makes
    terminology revisable at all. Re-queueing the corpus is what makes people
    stop revising it.

The prompt changed
    Everything the current prompt did not produce is arguably stale. In practice
    this is only acted on per tier, after a review pass found something the
    prompt should have prevented, because acting on it in general means
    retranslating everything every time a comma moves in an example.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydocvi.catalog import SegmentId
from pydocvi.memory import Memory, Segment

type TermMatcher = Callable[[str], frozenset[str]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Stale:
    """Segments invalidated by one cause, with the reason attached."""

    cause: str
    ids: tuple[SegmentId, ...]
    detail: str = ""

    def __len__(self) -> int:
        return len(self.ids)


def by_glossary(
    memory: Memory,
    changed_terms: Iterable[str],
    matcher: TermMatcher,
    *,
    protect_human: bool = True,
) -> Stale:
    """Segments whose English contains a term whose rendering changed.

    ``matcher`` is the glossary matcher, passed in rather than imported, and
    that is deliberate. There is exactly one matcher implementation in this
    project and three callers: the prompt builder, the ``G02`` audit rule and
    this function. If this one matched differently from the prompt builder, a
    version bump would re-queue strings the model was never told about and leave
    alone the ones it was.
    """
    changed = frozenset(changed_terms)
    if not changed:
        return Stale(cause="glossary", ids=(), detail="no terms changed")
    ids = tuple(
        sorted(
            segment.id
            for segment in memory
            if _affected(segment, changed, matcher, protect_human=protect_human)
        )
    )
    return Stale(
        cause="glossary",
        ids=ids,
        detail=f"{len(changed)} term(s) changed: {', '.join(sorted(changed))}",
    )


def _affected(
    segment: Segment,
    changed: frozenset[str],
    matcher: TermMatcher,
    *,
    protect_human: bool,
) -> bool:
    if protect_human and segment.source == "human":
        return False
    return bool(matcher(segment.msgid) & changed)


def by_prompt(
    memory: Memory, current_prompt: str, *, sources: frozenset[str] | None = None
) -> Stale:
    """Machine segments produced by a prompt other than the current one.

    Human work is never stale for this reason, which is why the default source
    set excludes it.
    """
    wanted = sources or frozenset({"machine"})
    ids = tuple(
        sorted(
            segment.id
            for segment in memory
            if segment.source in wanted and segment.prompt != current_prompt
        )
    )
    return Stale(
        cause="prompt",
        ids=ids,
        detail=f"current prompt {current_prompt}",
    )


def by_upstream(memory: Memory, upstream_ids: Iterable[SegmentId]) -> Stale:
    """Segments upstream no longer has a string for.

    Reported rather than deleted. A string that came back after a revert should
    find its translation waiting.
    """
    known = frozenset(upstream_ids)
    ids = tuple(sorted(segment.id for segment in memory if segment.id not in known))
    return Stale(cause="upstream", ids=ids, detail="orphaned, kept rather than deleted")
