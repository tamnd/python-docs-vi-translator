"""Cutting the corpus into calls.

A batch is a contiguous run of entries from one file, cut at entry boundaries
and filled until any of three caps is reached. Batching is a pure function of
the file, the caps and the classifier, so the same corpus always produces the
same batches with the same identifiers. That is what makes a killed run
resumable and what makes ``--check`` mean anything.
"""

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydocvi.catalog import Catalog, Entry, SegmentId
from pydocvi.classify import Kind, classify
from pydocvi.segment import Protected, protect

#: Characters of ``msgid`` in one call.
#:
#: The browser composer is overrun somewhere above this and the call comes back
#: as "No ChatGPT response found", which is indistinguishable from a network
#: failure and costs the whole batch.
CHARACTER_CAP = 6_000

#: Entries in one call. Past about forty items a numbered list stops being a
#: list the model transcribes and starts being one it summarises, merges or
#: skips the middle of.
ENTRY_CAP = 40

#: Protected spans in one call. This is the cap that was learned the hard way: a
#: 3 503-character body carrying 45 opaque spans was refused four times running,
#: every time for the spans and never for the prose. Copying dozens of markers
#: byte for byte while translating the sentences around them is the thing that
#: fails, and it fails on span count rather than on length.
SPAN_CAP = 60

#: Length of a batch identifier in hex characters. Content-addressed, so a batch
#: that comes back the same after a re-run is recognisably the same batch.
ID_LENGTH = 12

_BATCH_HASH_PREFIX = b"pydocvi.batch.1"


@dataclass(frozen=True, slots=True, kw_only=True)
class Item:
    """One entry on its way to a model, markup already replaced."""

    segment: SegmentId
    msgid: str
    protected: Protected

    @property
    def characters(self) -> int:
        return len(self.msgid)

    @property
    def spans(self) -> int:
        return self.protected.count

    @property
    def oversized(self) -> bool:
        """Whether this entry breaks a cap on its own.

        Six entries in the corpus do, the largest at 12 707 characters. A batch
        is never cut below an entry, so those go over alone. That is the right
        trade: a refusal then costs one entry rather than thirty-nine innocent
        ones sharing the call.
        """
        return self.characters > CHARACTER_CAP or self.spans > SPAN_CAP


@dataclass(frozen=True, slots=True, kw_only=True)
class Batch:
    """One call's worth of work, from one file."""

    id: str
    path: str
    items: tuple[Item, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def characters(self) -> int:
        return sum(item.characters for item in self.items)

    @property
    def spans(self) -> int:
        return sum(item.spans for item in self.items)

    @property
    def oversized(self) -> bool:
        return len(self.items) == 1 and self.items[0].oversized


def batch_id(path: str, segments: Sequence[SegmentId]) -> str:
    """A batch's identity: its file and the exact segments in it.

    Hashed rather than numbered, because a number would shift for every batch
    downstream of an upstream insertion and every trace filed under the old
    number would start describing a different batch.
    """
    payload = b"\0".join(
        [_BATCH_HASH_PREFIX, path.encode("utf-8"), *(s.encode() for s in segments)]
    )
    return hashlib.sha256(payload).hexdigest()[:ID_LENGTH]


def translatable(catalog: Catalog) -> list[Entry]:
    """The entries in one catalog a model should see."""
    return [entry for entry in catalog if classify(entry.msgid) is Kind.PROSE]


def items(entries: Iterable[Entry]) -> list[Item]:
    return [
        Item(segment=entry.id, msgid=entry.msgid, protected=protect(entry.msgid))
        for entry in entries
    ]


def pack(path: str, entries: Iterable[Item]) -> Iterator[Batch]:
    """Fill batches from one file's entries, in order.

    Order is upstream's order and never anything cleverer. Neighbouring entries
    in a ``.po`` are neighbouring paragraphs on the page, so a batch is a piece
    of one explanation rather than forty unrelated sentences, and the model gets
    that context for free.
    """
    current: list[Item] = []
    characters = spans = 0

    for item in entries:
        over_caps = (
            len(current) >= ENTRY_CAP
            or characters + item.characters > CHARACTER_CAP
            or spans + item.spans > SPAN_CAP
        )
        if current and over_caps:
            yield _batch(path, current)
            current, characters, spans = [], 0, 0
        current.append(item)
        characters += item.characters
        spans += item.spans

    if current:
        yield _batch(path, current)


def _batch(path: str, current: list[Item]) -> Batch:
    return Batch(
        id=batch_id(path, [item.segment for item in current]),
        path=path,
        items=tuple(current),
    )


def build(catalogs: Iterable[Catalog], *, root: Path | None = None) -> list[Batch]:
    """Every batch for a corpus, one file at a time.

    Never across files. A batch carries a context line into the prompt naming
    the file it came from, and that one sentence is worth more than the packing
    efficiency it costs. A mixed batch would have to drop it or lie. Duplicate
    strings across files are handled afterwards by the memory, which is the
    right place for them, rather than beforehand by clever packing.
    """
    out: list[Batch] = []
    for catalog in catalogs:
        path = str(catalog.path.relative_to(root)) if root else catalog.path.name
        out.extend(pack(path, items(translatable(catalog))))
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class Stats:
    """What ``batch --stats`` prints."""

    batches: int
    entries: int
    characters: int
    spans: int
    oversized: int
    files: int

    @property
    def entries_per_batch(self) -> float:
        return self.entries / self.batches if self.batches else 0.0

    @property
    def characters_per_batch(self) -> float:
        return self.characters / self.batches if self.batches else 0.0


def stats(batches: Sequence[Batch]) -> Stats:
    return Stats(
        batches=len(batches),
        entries=sum(len(batch) for batch in batches),
        characters=sum(batch.characters for batch in batches),
        spans=sum(batch.spans for batch in batches),
        oversized=sum(1 for batch in batches if batch.oversized),
        files=len({batch.path for batch in batches}),
    )


def by_file(batches: Iterable[Batch]) -> dict[str, int]:
    """Batches per file, which is how the tier order was decided."""
    counts: dict[str, int] = {}
    for batch in batches:
        counts[batch.path] = counts.get(batch.path, 0) + 1
    return counts


#: The six tiers, in the order they are translated.
#:
#: By review value per call, not by size and not alphabetically. Tier 1 is the
#: PEP 545 language-switcher set and is small enough that a person can read all
#: of it, which is the point: whatever is systematically wrong shows up there
#: for the price of sixty calls instead of in tier 5 for the price of thirteen
#: hundred. Tier 6 is last because ``changelog.po`` is a sixth of the corpus in
#: one generated file nobody reads in translation.
#:
#: A directory name matches the directory and a file name matches the file, and
#: nothing here is a glob. A pattern language would let a tier quietly widen the
#: day somebody adds a directory upstream, and every tier boundary in this table
#: is a decision somebody made rather than a shape a path happens to have.
TIERS: Mapping[int, tuple[str, ...]] = {
    1: ("bugs.po", "tutorial", "library/functions.po"),
    2: ("faq", "installing", "distributing", "deprecations"),
    3: ("reference", "howto", "using", "extending"),
    4: ("c-api",),
    5: ("library",),
    6: ("whatsnew", "changelog.po"),
}

#: Tier 2 also takes every file at the top of the tree. Named as a rule rather
#: than as a list, because the root pages are the ones upstream adds without
#: telling anybody, and a list would go stale silently and leave them
#: untranslated with no error anywhere.
ROOT_TIER = 2


class TierError(ValueError):
    """A tier that is not one of the six."""


def tier_of(path: str) -> int:
    """Which tier a corpus path belongs to.

    Longest match wins, which is the whole reason ``library/functions.po`` can
    be in tier 1 while the rest of ``library/`` waits for tier 5. Without that
    rule the table would be ambiguous and the ambiguity would be resolved by
    dictionary order, which is not a decision anybody made.
    """
    best, found = 0, ROOT_TIER if "/" not in path else 0
    for tier, patterns in TIERS.items():
        for pattern in patterns:
            if (path == pattern or path.startswith(f"{pattern}/")) and len(pattern) > best:
                best, found = len(pattern), tier
    return found


def of_tier(batches: Iterable[Batch], tier: int) -> list[Batch]:
    """The batches belonging to one tier, in corpus order."""
    if tier not in TIERS:
        raise TierError(f"tier {tier} is not one of {', '.join(str(n) for n in TIERS)}")
    return [batch for batch in batches if tier_of(batch.path) == tier]
