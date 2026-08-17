"""Where candidate terms come from.

Four sources, in descending order of trust, and the order is the whole design.
What the humans already rendered beats what CPython's own glossary page calls a
term, which beats what is common, which beats what the previous pipeline
rendered three different ways.

Nothing here decides anything. Mining produces
``manifests/glossary-candidates.yaml`` and a candidate is a question, not a row.
The answering happens in :mod:`pydocvi.curate` and then in front of a person.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from pydocvi.catalog import Catalog, Entry
from pydocvi.glossary import scalar
from pydocvi.memory import Memory
from pydocvi.segment import strip_markup

#: How many frequency candidates to keep. The corpus has tens of thousands of
#: distinct noun phrases and almost all of them are prose.
FREQUENCY_LIMIT = 800

#: The longest phrase frequency counting will propose. Past three words a
#: repeated phrase is a sentence fragment rather than a term.
MAX_WORDS = 3

#: The shortest. Counting single words over a documentation corpus measures how
#: common a word is in English, not whether it is a term, so a single word has
#: to come from a source a person curated instead. See ``from_frequency``.
MIN_WORDS = 2

#: How many times a phrase has to appear before it is worth asking about.
MIN_COUNT = 8

#: The longest a ``msgid`` can be and still be a term rather than a sentence.
TERM_CHARACTERS = 48

#: The fewest letters a word may have for a phrase to count as a term. One is a
#: label on an index page, not something to ask a model to render.
MIN_LETTERS = 2

#: Words that may not appear anywhere in a term. Every word wrongly in here is a
#: term the mine will never propose, and that failure is silent, so the bar for
#: adding one is high: nothing goes in that could be a row in the contract.
#:
#: The list was short on purpose and the first real run over the corpus showed
#: that short was wrong. The spec asks this source for noun phrases. What it
#: gave back was changelog prose, because a documentation corpus is mostly
#: English and English words outrank terminology on raw count. ``now`` came
#: back 6,601 times, above every real term in the corpus, and 70% of the 728
#: single-word candidates were ordinary dictionary words.
#:
#: So the words below are closed-class, plus verbs and adverbs that carry no
#: technical sense. A determiner or a preposition can never be a glossary row,
#: which is what makes them safe to list. Words that read like prose but are
#: also Python terms are deliberately absent: ``set``, ``type``, ``list``,
#: ``class``, ``object``, ``value``, ``name``, ``call``, ``return``, ``file``,
#: ``string``, ``code``, ``data``, ``error``, ``module``, ``next``, ``get`` and
#: ``help`` all stay out even though every one of them is common English.
_STOPWORDS = """
a about above after again against all almost along already also although always
among an and another any are around as at back be because become becomes been
before behind being below beside besides between beyond both but by came can
come comes could currently do does during each either else enough especially
even ever every except few first for from further give given gives go goes had
has have he her here him his how however i if in instead into is it its itself
just least less let like likely made make makes making many may me might more
most much must my near neither never no not now of often on once one ones only
or other others otherwise our out outside over own per perhaps please
previously rather really same see seen several she should show shown simply
since so some still such take taken takes than that the their theirs them
themselves then there therefore these they this those though through throughout
thus to together too toward towards try under unless until up upon us use used
using usually very via want was way we well were what whatever when whenever
where whether which while who whom whose why will with within without would yet
you your yourself
"""

STOPWORDS = frozenset(_STOPWORDS.split())

#: A word, for the purposes of counting phrases. Letters only, so version
#: numbers, hex constants and ``__init__`` never reach the counter.
#:
#: The apostrophes are in here because they were missing. Without them the
#: first real run over the corpus split every contraction in two and proposed
#: both halves: ``doesn`` and ``doesn t`` each 550 times, ``don`` 606, ``isn``
#: 260, ``python s`` 400. Both the straight and the typographic form are
#: allowed, because the corpus contains both.
_WORD = re.compile("[A-Za-z][A-Za-z'\\u2019-]*")

#: A phrase carrying one of these is a contraction or a possessive, which is
#: prose wearing a term's clothes.
_APOSTROPHE = re.compile("['\\u2019]")

#: Sentence-final punctuation. A ``msgid`` carrying one is prose.
_SENTENCE = re.compile(r"[.!?:;]\s*$")


class Source(IntEnum):
    """Where a candidate came from, ordered by how much it is worth.

    An ``IntEnum`` because the order is the point and merging two candidates
    means keeping the lower number. The values are trust, not sequence: 1 is the
    1 435 strings people have already translated and 4 is a machine
    contradicting itself.
    """

    HUMAN = 1
    TERM_PAGE = 2
    FREQUENCY = 3
    MACHINE = 4

    @property
    def label(self) -> str:
        return {
            Source.HUMAN: "human translation",
            Source.TERM_PAGE: "glossary.po",
            Source.FREQUENCY: "corpus frequency",
            Source.MACHINE: "machine disagreement",
        }[self]


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """One English phrase worth asking about, and why.

    ``definition`` is the sentence that goes into the curation prompt beside the
    term. A term without one is asked about bare, and a model asked to render
    "annotation" with no context will render the English word rather than the
    Python concept.

    ``seen`` is every rendering the sources observed. For a human candidate that
    is evidence and usually the answer. For a machine candidate it is the
    disagreement that made it a candidate at all.
    """

    en: str
    source: Source
    count: int = 0
    definition: str = ""
    seen: tuple[str, ...] = ()

    @property
    def words(self) -> int:
        return len(self.en.split())

    @property
    def contested(self) -> bool:
        """Whether the sources rendered this more than one way."""
        return len(set(self.seen)) > 1


def term_like(msgid: str) -> bool:
    """Whether a ``msgid`` is a term rather than a sentence about one.

    Headings, index entries and glossary keys are terms. Anything with
    sentence-final punctuation, more than three words or more than
    :data:`TERM_CHARACTERS` characters is prose, and prose aligned against its
    own translation is a sentence pair rather than a term pair.
    """
    prose = strip_markup(msgid).strip()
    if not prose or _SENTENCE.search(prose):
        return False
    if len(prose) > TERM_CHARACTERS or len(prose.split()) > MAX_WORDS:
        return False
    return substantial(prose)


def substantial(text: str) -> bool:
    """Whether a phrase has a word in it, as opposed to a letter.

    Found by running this over the real corpus. The alphabet headings on
    CPython's glossary index page are ``**A**`` through ``**Z**``, they are
    short, they carry no sentence punctuation and they survived every other
    filter, so 19 of the 102 highest-trust candidates were single letters. The
    frequency source produced another 21 the same way.

    ``MIN_LETTERS`` rather than a ratio of letters to punctuation, because
    ``# (hash)`` is a real glossary entry and a ratio would have to refuse it to
    refuse ``**A**``.
    """
    return any(len(word.group(0)) >= MIN_LETTERS for word in _WORD.finditer(text))


def from_human(memory: Memory) -> list[Candidate]:
    """Terms the 1 435 human translations already decided.

    The highest-value source and the smallest, which is why ``sync --human``
    runs before mining rather than after.

    Alignment is only attempted where it is free: a short entry with no sentence
    punctuation is one term, and its ``msgstr`` is that term's rendering. Longer
    entries are left alone. A word aligner over 1 435 sentence pairs would
    propose renderings nobody wrote, and a candidate list nobody trusts gets
    read once.

    The memory holds one rendering per ``msgid`` and ``msgctxt``, so a candidate
    from here is contested only when the same English was rendered two ways
    under two different contexts. That is rare and it is exactly the case worth
    seeing, because it is a person having decided the term needs a ``context``
    row without there being one.
    """
    seen: dict[str, list[str]] = {}
    for segment in memory:
        if segment.source != "human" or not segment.msgstr.strip():
            continue
        if term_like(segment.msgid):
            seen.setdefault(strip_markup(segment.msgid).strip(), []).append(segment.msgstr.strip())
    return [
        Candidate(en=en, source=Source.HUMAN, count=len(renderings), seen=tuple(renderings))
        for en, renderings in sorted(seen.items())
    ]


def from_term_page(catalog: Catalog) -> list[Candidate]:
    """Terms from ``glossary.po``, with the definition that follows each one.

    CPython's glossary page is 18 000 words defining exactly the terms this
    project needs. The page's structure survives into the catalog as an
    alternation: a short entry naming a term, then one or more longer entries
    defining it. That alternation is what is read here, and the first definition
    paragraph is kept as the context the curation prompt needs.
    """
    out: list[Candidate] = []
    entries = list(catalog)
    for index, entry in enumerate(entries):
        if not term_like(entry.msgid) or _is_code(entry.msgid):
            continue
        definition = _definition(entries[index + 1 :])
        if definition is None:
            continue
        out.append(
            Candidate(
                en=strip_markup(entry.msgid).strip(),
                source=Source.TERM_PAGE,
                definition=definition,
            )
        )
    return out


def _is_code(msgid: str) -> bool:
    """Whether the entry is nothing but markup.

    ``>>>`` and ``...`` are glossary keys on the page and neither is a term
    anybody translates.
    """
    return not strip_markup(msgid).strip(" .>")


def _definition(rest: Sequence[Entry]) -> str | None:
    """The first sentence of the paragraph following a term, or nothing."""
    for entry in rest:
        prose = " ".join(strip_markup(entry.msgid).split())
        if not prose:
            continue
        if term_like(entry.msgid):
            return None
        sentence, _, _ = prose.partition(". ")
        return sentence.strip().rstrip(".") + "."
    return None


def from_frequency(
    msgids: Iterable[str], *, limit: int = FREQUENCY_LIMIT, minimum: int = MIN_COUNT
) -> list[Candidate]:
    """The most frequent noun phrases in the corpus, markup excluded.

    Excluded rather than merely ignored. A phrase that only ever appears inside
    a double-backtick span is code, and code is what the protector already took
    out, so counting over the stripped text is the same operation the prompt
    builder does and needs no second definition of what a span is.

    Phrases are two or three words with no stopword anywhere in them. That is
    not a noun-phrase grammar and does not claim to be. It is the cheapest
    filter that keeps "context manager" and drops "of the".

    Anywhere rather than at the ends, which is what it used to be. Checking the
    ends only lets a stopword sit in the middle, and the middle is where the
    changelog puts it: "patch by victor", "contributed by serhiy" and "see for
    more" were all candidates on the first real run.

    Two words rather than one, which is the harder rule and the one that costs
    something. Measured over the corpus, 65% of the single-word candidates this
    produced were ordinary English dictionary words, and the highest-count
    candidate in the whole run was "now" at 6,601. No stopword list fixes that,
    because the list would have to be all of English.

    A glossary row is not a suggestion. It is quoted in every prompt and held to
    by ``G02`` across all 87,008 entries, so a wrong row fails translations that
    are correct, while a missing row only leaves a term unenforced. Those costs
    are not symmetric, and this rule buys precision with recall accordingly.

    It does lose real single-word terms: "encoding", "timeout", "traceback",
    "whitespace", "newline", "float". They come back by hand in review, which is
    a much smaller job than striking 482 prose rows. Single words that a person
    has already vouched for are unaffected, because they arrive from the human
    translations and from ``glossary.po`` rather than from here.

    A phrase carrying an apostrophe is dropped rather than asked about. Every
    one of them is a contraction or a possessive, and asking a model to render
    "python's" wastes a line and gets an answer that cannot be a glossary row.
    """
    counts: dict[str, int] = {}
    for msgid in msgids:
        for phrase, count in _phrases(strip_markup(msgid)).items():
            counts[phrase] = counts.get(phrase, 0) + count
    ranked = sorted(
        (
            (phrase, count)
            for phrase, count in counts.items()
            if count >= minimum
            and len(phrase.split()) >= MIN_WORDS
            and substantial(phrase)
            and not _APOSTROPHE.search(phrase)
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [
        Candidate(en=phrase, source=Source.FREQUENCY, count=count)
        for phrase, count in ranked[:limit]
    ]


def _phrases(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in _runs(text):
        for size in range(1, MAX_WORDS + 1):
            for start in range(len(run) - size + 1):
                window = run[start : start + size]
                if any(word in STOPWORDS for word in window):
                    continue
                phrase = " ".join(window)
                counts[phrase] = counts.get(phrase, 0) + 1
    return counts


def _runs(text: str) -> list[list[str]]:
    """Words grouped by the punctuation between them.

    Phrases never cross a comma or a full stop, because "the module, function"
    is two things and counting it as one produces candidates that read like
    somebody transcribed a list wrongly.
    """
    return [
        [word.group(0).lower() for word in _WORD.finditer(clause)]
        for clause in re.split(r"[^A-Za-z-]*[,.;:()\[\]]+[^A-Za-z-]*|\n", text)
        if clause
    ]


def from_machine(catalogs: Iterable[Catalog]) -> list[Candidate]:
    """Terms the previous pipeline rendered inconsistently.

    Where Google rendered the same English three different ways across three
    files, that phrase is a candidate *because* it was inconsistent.
    Disagreement is a signal, and it is the only signal in this module that
    points at a term the other three sources have no reason to notice.

    Agreement is not a signal, so a phrase rendered the same way everywhere is
    not proposed here. It may still arrive from frequency, which is the right
    outcome: it is a candidate on its merits rather than on a machine having
    been consistent.
    """
    seen: dict[str, list[str]] = {}
    for catalog in catalogs:
        for entry in catalog:
            if not entry.msgstr.strip() or not term_like(entry.msgid):
                continue
            renderings = seen.setdefault(strip_markup(entry.msgid).strip(), [])
            renderings.append(entry.msgstr.strip())
    return [
        Candidate(
            en=en, source=Source.MACHINE, count=len(renderings), seen=tuple(sorted(set(renderings)))
        )
        for en, renderings in sorted(seen.items())
        if len(set(renderings)) > 1
    ]


def merge(*groups: Iterable[Candidate]) -> list[Candidate]:
    """One list, each phrase once, attributed to its most trusted source.

    Counts are summed and renderings are unioned across sources, because a
    phrase that is both frequent and rendered two ways by the old pipeline is
    more interesting than either fact alone. The definition is kept from
    whichever source had one, since only ``glossary.po`` ever does.

    The result is sorted by trust and then alphabetically, never by count.
    Sorting a candidate list by count puts "the following example" above
    "context manager", and the list is read from the top.
    """
    merged: dict[str, Candidate] = {}
    for group in groups:
        for candidate in group:
            existing = merged.get(candidate.en)
            merged[candidate.en] = candidate if existing is None else _combine(existing, candidate)
    return sorted(merged.values(), key=lambda candidate: (candidate.source, candidate.en))


def _combine(left: Candidate, right: Candidate) -> Candidate:
    best, other = (left, right) if left.source <= right.source else (right, left)
    return replace(
        best,
        count=left.count + right.count,
        definition=best.definition or other.definition,
        seen=tuple(sorted(set(left.seen) | set(right.seen))),
    )


def dumps(candidates: Sequence[Candidate]) -> str:
    """Write ``manifests/glossary-candidates.yaml``.

    Hand-rolled for the same reason the glossary itself is: this file is read in
    a diff between one mining run and the next, and a key that moves makes that
    diff useless.
    """
    out = [
        "# Written by pydocvi glossary mine. Candidates are questions, not rows.",
        f"# {len(candidates)} candidate(s), most trusted source first.",
        "candidates:",
    ]
    for candidate in candidates:
        out.append(f"  - en: {scalar(candidate.en)}")
        out.append(f"    source: {candidate.source.name.lower()}")
        out.append(f"    count: {candidate.count}")
        if candidate.definition:
            out.append(f"    definition: {scalar(candidate.definition)}")
        if candidate.seen:
            out.append("    seen:")
            out.extend(f"      - {scalar(rendering)}" for rendering in candidate.seen)
    return "\n".join(out) + "\n"


class MineError(ValueError):
    """A candidates file this module cannot read."""


def loads(text: str) -> list[Candidate]:
    """Read back what ``dumps`` wrote.

    Read back rather than kept in memory because a person edits this file
    between mining and curating, and the edited file is the one that should be
    asked about.
    """
    yaml = YAML(typ="safe")
    try:
        payload = yaml.load(text)
    except YAMLError as error:
        raise MineError(f"not valid YAML: {error}") from error
    if not isinstance(payload, dict):
        raise MineError("expected a mapping at the top level")
    raw = payload.get("candidates")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MineError("candidates must be a list")
    return [_candidate(one, at) for at, one in enumerate(raw, start=1)]


def _candidate(raw: object, at: int) -> Candidate:
    if not isinstance(raw, dict):
        raise MineError(f"candidate {at} is not a mapping")
    unknown = set(raw) - {"en", "source", "count", "definition", "seen"}
    if unknown:
        raise MineError(f"candidate {at} has unknown fields: {', '.join(sorted(unknown))}")
    name = str(raw.get("source", "")).upper()
    if name not in Source.__members__:
        raise MineError(f"candidate {at} has unknown source {name.lower()!r}")
    seen = raw.get("seen") or []
    if not isinstance(seen, list):
        raise MineError(f"candidate {at} has a non-list seen")
    return Candidate(
        en=str(raw.get("en", "")),
        source=Source[name],
        count=int(raw.get("count", 0)),
        definition=str(raw.get("definition", "")),
        seen=tuple(str(one) for one in seen),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stats:
    """What ``glossary mine`` prints when it finishes."""

    total: int
    by_source: dict[str, int]
    defined: int
    contested: int


def stats(candidates: Sequence[Candidate]) -> Stats:
    by_source: dict[str, int] = {}
    for candidate in candidates:
        by_source[candidate.source.label] = by_source.get(candidate.source.label, 0) + 1
    return Stats(
        total=len(candidates),
        by_source=by_source,
        defined=sum(1 for candidate in candidates if candidate.definition),
        contested=sum(1 for candidate in candidates if candidate.contested),
    )
