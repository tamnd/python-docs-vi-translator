"""Reading a numbered answer.

The model is asked for a numbered list, not JSON, and this module is the reason.
A browser-backed session wraps JSON in a fence, escapes the Vietnamese or emits
a trailing comma often enough to matter, and one parse failure then throws away
all forty entries. A numbered list degrades one entry at a time: line 17
unparseable means entry 17 is refused and 1 to 16 and 18 to 40 are kept. At 151
seconds a call, partial credit is the whole game.

The parser is strict about the marker and permissive about everything after it.
"""

import re
from dataclasses import dataclass, field

#: What starts an entry: a number at the start of a line, optionally followed by
#: a dot or a bracket, then whitespace. Everything up to the next marker is the
#: body, however strange it looks.
MARKER = re.compile(r"^(\d+)[.):\]]?[ \t]+", re.MULTILINE)

#: A fenced block or a horizontal rule at the top of an answer means the model
#: decided to present the work rather than do it. Detected here so that ``P07``
#: has something to read.
FENCE = re.compile(r"^\s*(```|~~~|---)")


@dataclass(frozen=True, slots=True, kw_only=True)
class Problem:
    """One thing wrong with an answer, named by index where there is one."""

    kind: str
    index: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        where = f"entry {self.index}" if self.index is not None else "answer"
        return f"{where}: {self.kind}{f' ({self.detail})' if self.detail else ''}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """What came back, split into what is usable and what is not."""

    entries: dict[int, str] = field(default_factory=dict)
    problems: tuple[Problem, ...] = ()
    fenced: bool = False

    @property
    def usable(self) -> bool:
        return bool(self.entries)

    def missing(self, count: int) -> tuple[int, ...]:
        return tuple(n for n in range(1, count + 1) if n not in self.entries)


def parse(text: str, count: int) -> Answer:
    """Split an answer into entries, refusing the parts that do not fit.

    ``count`` is how many entries the batch asked about. A missing index is a
    problem reported against its own index and costs only its own entry. Only
    the alignment of the whole answer, which is spec 06's ``P03``, is a reason
    to reject everything.

    A numbered line is a marker only where it continues the sequence, and
    anything else on a line that looks like one is text. 29 of the 73 413 prose
    entries in this corpus contain such a line: shell transcripts, dtrace
    output, and numbered lists inside the prose. Reading one as a marker used to
    truncate the entry it was inside, silently, keeping the part before it and
    dropping the rest, and the result passed every invariant that a short
    translation passes. A half-translated string that nothing complains about is
    the worst thing this file can produce.

    28 of those 29 now survive whole. The last is a table of Unicode codepoints
    in ``unicode.po`` whose own lines run 0, 1, 2, 3, 4, which is a numbered
    sequence no parser working a line at a time can tell from an answer. It
    fails ``P03``, is retried, dies, and is left in English, which is what the
    ladder is for.

    Nothing is lost by being strict. An answer whose indices are not exactly
    ``1..count`` fails ``P03`` and the batch is retried whole, so the only
    behaviour this changes is the one that was wrong.
    """
    problems: list[Problem] = []
    entries: dict[int, str] = {}
    fenced = bool(FENCE.match(text))

    markers = list(MARKER.finditer(text))
    if not markers:
        return Answer(problems=(Problem(kind="no numbered entries found"),), fenced=fenced)

    preamble = text[: markers[0].start()].strip()
    if preamble and not fenced:
        problems.append(Problem(kind="preamble before the first entry", detail=preamble[:80]))

    real, stray = _sequence(markers, count)
    if stray:
        problems.append(
            Problem(
                kind="numbered line(s) inside an entry, read as text",
                detail=" ".join(str(n) for n in stray),
            )
        )

    for position, marker in enumerate(real):
        end = real[position + 1].start() if position + 1 < len(real) else len(text)
        index = int(marker.group(1))
        body = _body(text[marker.end() : end])
        if not body:
            problems.append(Problem(kind="empty body", index=index))
            continue
        entries[index] = body

    problems.extend(
        Problem(kind="no answer for this entry", index=n)
        for n in range(1, count + 1)
        if n not in entries
    )
    return Answer(entries=entries, problems=tuple(problems), fenced=fenced)


def _sequence(markers: list[re.Match[str]], count: int) -> tuple[list[re.Match[str]], list[int]]:
    """The markers that continue ``1, 2, 3, ...``, and the numbers of the rest.

    A model that skips an entry loses everything after the gap here, which
    sounds worse than it is: a gap fails ``P03``, and a batch that fails ``P03``
    is retried whole however much of it was parsed.
    """
    real: list[re.Match[str]] = []
    stray: list[int] = []
    expected = 1
    for marker in markers:
        index = int(marker.group(1))
        if index == expected <= count:
            real.append(marker)
            expected += 1
        else:
            stray.append(index)
    return real, stray


def _body(raw: str) -> str:
    """Strip the newlines the format introduced and nothing else.

    Leading and trailing spaces inside a body are kept, because 53 entries in
    this corpus have significant edge whitespace in the ``msgid`` and a parser
    that trimmed it would make those entries fail ``P04`` for a reason the model
    had no part in.
    """
    return raw.strip("\n").rstrip() if raw.strip() else ""


def aligned(answer: Answer, count: int) -> bool:
    """Whether the indices are exactly ``1..count``.

    This is the one property of the whole answer rather than of an entry, so it
    is the one failure that rejects a batch instead of an entry.
    """
    return sorted(answer.entries) == list(range(1, count + 1))
