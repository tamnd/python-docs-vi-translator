"""The translation memory.

One store, keyed by ``(msgctxt, msgid)`` through the segment id, holding every
translation this project has ever accepted along with where it came from. It is
the only durable artefact of a translation run. The catalogs in the content repo
are a projection of it, produced by ``apply``, and can be thrown away and
rebuilt.

Keying on the string rather than on a file and a line is what lets one memory
serve several upstream branches. The 3.14 catalogs are the same 85 192 strings
in a different arrangement, so they are produced from this store rather than
translated a second time.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Self

from pydocvi.catalog import Entry, SegmentId, segment_id

type Source = Literal["human", "machine", "passthrough", "legacy"]

#: Which source wins when two of them offer a translation for the same segment.
#: A person's work is never overwritten by a machine, and the output of the
#: pipeline this project replaces never overwrites anything.
PRECEDENCE: Mapping[Source, int] = {
    "human": 40,
    "machine": 30,
    "passthrough": 20,
    "legacy": 10,
}

SOURCES: tuple[Source, ...] = ("human", "machine", "passthrough", "legacy")


@dataclass(frozen=True, slots=True, kw_only=True)
class Segment:
    """One remembered translation and its provenance.

    The provenance fields are not bookkeeping. When a glossary bump or a prompt
    change invalidates work, they are how ``stale`` decides what to re-queue,
    and re-queueing the affected few hundred entries rather than the corpus is
    the difference between revisable terminology and terminology nobody dares
    touch.
    """

    id: SegmentId
    msgid: str
    msgstr: str
    source: Source
    msgctxt: str | None = None
    model: str | None = None
    prompt: str | None = None
    glossary: int | None = None
    batch: str | None = None
    run: str | None = None

    @classmethod
    def from_entry(cls, entry: Entry, *, source: Source, **provenance: object) -> Self:
        return cls(
            id=entry.id,
            msgid=entry.msgid,
            msgctxt=entry.msgctxt,
            msgstr=entry.msgstr,
            source=source,
            **provenance,  # type: ignore[arg-type]
        )

    @property
    def rank(self) -> int:
        return PRECEDENCE[self.source]

    def wins_over(self, other: Segment | None) -> bool:
        """Whether this segment should replace ``other``.

        Ties go to the incumbent. Re-running a tier should not churn the store
        when the model happened to produce the same thing twice, and should not
        silently swap one machine translation for an equally good one either.
        """
        if other is None:
            return True
        return self.rank > other.rank


class Memory:
    """A JSON-backed store of segments.

    Loaded whole. 87 008 segments is a few tens of megabytes, the machine
    running this has more than that, and a database would buy nothing except a
    schema migration the first time a provenance field is added.
    """

    def __init__(self, segments: Iterable[Segment] = (), *, path: Path | None = None) -> None:
        self._segments: dict[SegmentId, Segment] = {s.id: s for s in segments}
        self.path = path

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self) -> Iterator[Segment]:
        return iter(self._segments.values())

    def __contains__(self, key: SegmentId) -> bool:
        return key in self._segments

    def get(self, key: SegmentId) -> Segment | None:
        return self._segments.get(key)

    def lookup(self, msgid: str, msgctxt: str | None = None) -> Segment | None:
        return self._segments.get(segment_id(msgid, msgctxt))

    def add(self, segment: Segment) -> bool:
        """Store a segment if precedence allows. Returns whether it was stored."""
        if not segment.wins_over(self._segments.get(segment.id)):
            return False
        self._segments[segment.id] = segment
        return True

    def extend(self, segments: Iterable[Segment]) -> int:
        return sum(1 for segment in segments if self.add(segment))

    def remove(self, key: SegmentId) -> bool:
        return self._segments.pop(key, None) is not None

    def touch(self, key: SegmentId, **updates: object) -> None:
        """Replace fields on a stored segment."""
        current = self._segments[key]
        self._segments[key] = replace(current, **updates)  # type: ignore[arg-type]

    def counts(self) -> dict[Source, int]:
        counts: dict[Source, int] = dict.fromkeys(SOURCES, 0)
        for segment in self._segments.values():
            counts[segment.source] += 1
        return counts

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a store, or return an empty one if it does not exist yet."""
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            (Segment(**row) for row in payload["segments"]),
            path=path,
        )

    def save(self, path: Path | None = None) -> Path:
        """Write the store atomically, sorted, one segment per line group.

        Sorted by segment id so that two runs that reached the same state
        produce the same bytes. An unsorted dump would show every run as a whole
        file changed and make the store unreviewable.
        """
        target = path or self.path
        if target is None:
            raise ValueError("no path to save to")
        payload = {
            "version": 1,
            "segments": [
                {k: v for k, v in asdict(segment).items() if v is not None}
                for segment in sorted(self._segments.values(), key=lambda s: s.id)
            ],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.pydocvi-tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        return target
