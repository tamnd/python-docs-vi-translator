"""The term list, and the one matcher that reads it.

The glossary is the artefact of this project that cannot be repaired
afterwards. A mistranslated sentence is one entry to fix. A term rendered
wrong is every entry that used it, across 548 files, and nobody finds out
until a reader searches for the right Vietnamese word and gets nothing back.

Two files, two audiences. ``manifests/glossary.yaml`` is the rows and the
version, read by the tool. ``GLOSSARY.md`` is the style rules, the worked
examples and the same rows rendered as a table, read by reviewers. ``G05``
fails when the two disagree, so there is one source of truth with two
renderings rather than two lists that drift.

The matcher at the bottom of this module has one implementation and three
callers: the prompt builder, the ``G02`` audit rule and ``stale --glossary``.
That is a hard rule rather than a preference. If the audit matched differently
from the prompt builder, the audit would hold a translation to a term the model
was never shown, and the obvious fix for that failure is to edit the glossary,
which makes it worse.
"""

import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

from pydocvi.segment import strip_markup

#: What separates a word from the text around it, for matching purposes.
#:
#: Not ``\b``, because ``\b`` is defined against ``\w`` and would find "built"
#: inside "built-in" and "list" inside "list_of". A term is a whole word with
#: neither a letter, a digit, an underscore nor a hyphen touching either end.
_BEFORE = r"(?<![0-9A-Za-z_-])"
_AFTER = r"(?![0-9A-Za-z_-])"

#: The keyword a curated row uses to say the English word is the term.
KEEP = "KEEP"


@dataclass(frozen=True, slots=True, kw_only=True)
class Term:
    """One row.

    ``keep_en`` is the field that makes this glossary honest. Half of Python's
    vocabulary is not translated by Vietnamese programmers, and a glossary that
    insisted on a Vietnamese word for every one of them would produce a document
    no Vietnamese Python programmer would read. A ``keep_en`` row is a decision
    that the English word *is* the Vietnamese term, it goes into the prompt as
    such, and ``G02`` then checks the opposite thing: that the English survived
    rather than that it was replaced.

    ``identifier`` is the field for the words that are both. "float" in a
    sentence is "số thực" and the corpus says so twice; "float" on its own in the
    table of struct format codes is the name of a C type and translating it would
    break the table. One row has to be able to say both things, because the
    alternative is what the file did before: pick the reading that suits the
    louder check and be wrong about the other. ``keep_en`` answers the question
    for running prose, ``identifier`` answers it for an entry that is nothing but
    the term, and ``L02`` is the only check that asks the second one.

    ``context`` is a filter, not prose. It is a fragment matched against the
    entry's ``msgctxt`` and against the file path, so a row carrying one applies
    only where that fragment appears. Prose about a row belongs in ``note``,
    which nothing matches against. There should stay few contexts, because a
    term that needs one is a term two reviewers will render two ways.
    """

    en: str
    vi: str
    keep_en: bool = False
    identifier: bool = False
    context: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.en.strip():
            raise ValueError("a term with no English side is not a term")
        if not self.vi.strip():
            raise ValueError(f"{self.en!r} has no rendering")

    @property
    def rendering(self) -> str:
        """What the translation is expected to contain."""
        return self.en if self.keep_en else self.vi

    def applies(self, *, where: str | None = None, msgctxt: str | None = None) -> bool:
        """Whether this row is in force for one entry.

        A row with no context is always in force. A row with one is in force
        where the fragment appears in the path or in the ``msgctxt``, and
        nowhere else, which is the whole point of having the field.
        """
        if self.context is None:
            return True
        fragment = self.context.casefold()
        return any(fragment in haystack.casefold() for haystack in (where or "", msgctxt or ""))


@dataclass(frozen=True, slots=True, kw_only=True)
class Glossary:
    """A version and the rows it names.

    ``version`` is a single integer and it moves whenever any ``vi`` changes or
    any row is added or removed. It is stamped onto every segment the memory
    stores, which is what lets ``stale --glossary`` find the few hundred entries
    a bump affects instead of re-queueing all 87 008.
    """

    version: int
    terms: tuple[Term, ...] = ()

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def get(self, en: str) -> Term | None:
        return next((term for term in self.terms if term.en == en), None)

    @property
    def by_en(self) -> dict[str, Term]:
        return {term.en: term for term in self.terms}

    @property
    def translated(self) -> tuple[Term, ...]:
        return tuple(term for term in self.terms if not term.keep_en)

    @property
    def kept(self) -> tuple[Term, ...]:
        return tuple(term for term in self.terms if term.keep_en)

    @property
    def standalone(self) -> tuple[Term, ...]:
        """The rows an entry may equal and still be correct in English.

        A ``keep_en`` row is one of these without saying so: a term that stays
        English everywhere stays English standing alone too. The rows that need
        the flag are the ones translated in a sentence and named in a table.
        """
        return tuple(term for term in self.terms if term.keep_en or term.identifier)

    def with_terms(self, terms: Iterable[Term], *, version: int | None = None) -> Self:
        rows = tuple(terms)
        bumped = version if version is not None else self.version + bool(_changed(self.terms, rows))
        return replace(self, version=bumped, terms=rows)

    def matcher(self) -> Matcher:
        return Matcher(self)


def match_order(terms: Iterable[Term]) -> tuple[Term, ...]:
    """Longest English side first, then alphabetical.

    "context manager" contains "context" and "list comprehension" contains
    "list", so a matcher that took the first row that fitted would tell the
    model to render "context manager" as "ngu canh manager". Length first
    settles that; the alphabetical tiebreak only exists so the order is a
    function of the rows rather than of the order they were loaded in.
    """
    return tuple(sorted(terms, key=lambda term: (-len(term.en), term.en)))


class Matcher:
    """Which terms appear in a string.

    One implementation, three callers, and this is the hard rule of the module.
    The prompt builder uses :meth:`select` to decide which rows go into a
    batch's prompt. The ``G02`` audit rule uses :meth:`missing` to check that a
    term that appeared in the English got its rendering in the Vietnamese.
    ``stale --glossary`` calls the instance to find which entries a version bump
    affects.

    Matching is over the *unprotected* text, on whole words, longest term first,
    and case-insensitively for the first letter only. The last of those is the
    fussy one and it is deliberate: "Iterable" opening a sentence is the same
    term, "ITERABLE" in a heading is shouting, and "URL" is not "url".
    """

    def __init__(self, glossary: Glossary) -> None:
        self.glossary = glossary
        self._order = match_order(glossary.terms)
        self._everything = frozenset(range(len(self._order)))
        self._contextual = any(term.context is not None for term in self._order)
        self._patterns = {self._everything: _compile(list(enumerate(self._order)))}

    def __call__(
        self, text: str, *, where: str | None = None, msgctxt: str | None = None
    ) -> frozenset[str]:
        """The English side of every term in the string.

        The signature ``stale.by_glossary`` expects, with the context arguments
        optional so that a caller holding nothing but a ``msgid`` can still ask.
        """
        return frozenset(term.en for term in self.select(text, where=where, msgctxt=msgctxt))

    def select(
        self, text: str, *, where: str | None = None, msgctxt: str | None = None
    ) -> tuple[Term, ...]:
        """The rows in force for this string, in match order.

        Returned as rows rather than as names because the prompt builder needs
        the rendering and the note, and a caller that had to look each name back
        up would be a second place where context is applied.

        Context is applied before matching rather than after, and the difference
        is visible. "list comprehension" is a contextual row and "list" is not,
        so in a file where the contextual row is out of force, the string "a
        list comprehension" matches "list". Filtering afterwards would have let
        a row that is not in force shadow one that is, which is the one way a
        context can make the glossary quieter instead of more precise.
        """
        pattern = self._pattern(where, msgctxt)
        if pattern is None:
            return ()
        found: set[int] = set()
        for match in pattern.finditer(strip_markup(text)):
            assert match.lastgroup is not None
            found.add(int(match.lastgroup.removeprefix("t")))
        return tuple(self._order[index] for index in sorted(found))

    def _pattern(self, where: str | None, msgctxt: str | None) -> re.Pattern[str] | None:
        """The alternation for the rows in force, compiled once per set of them.

        Cached on the set of rows rather than on the path, because a glossary
        with three contextual rows has at most eight distinct sets however many
        files it is run over, and the corpus has 548 files.
        """
        if not self._contextual:
            return self._patterns[self._everything]
        in_force = frozenset(
            index
            for index, term in enumerate(self._order)
            if term.applies(where=where, msgctxt=msgctxt)
        )
        if in_force not in self._patterns:
            self._patterns[in_force] = _compile(
                [(index, self._order[index]) for index in sorted(in_force)]
            )
        return self._patterns[in_force]

    def missing(
        self, msgid: str, msgstr: str, *, where: str | None = None, msgctxt: str | None = None
    ) -> tuple[Term, ...]:
        """Terms the English used whose rendering never arrived in the Vietnamese.

        This is ``G02``. A ``keep_en`` row passes when the English word survived
        and a translated row passes when the Vietnamese phrase is there, which
        is the same check asking opposite questions of the two kinds of row.

        Presence, not agreement. Vietnamese word order moves modifiers around
        and the rendering can be inflected by the words beside it, so anything
        stricter than "the phrase is in there somewhere" would fail on correct
        translations, and a rule that fails on correct work stops being read.
        """
        found = self.select(msgid, where=where, msgctxt=msgctxt)
        prose = _fold(strip_markup(msgstr))
        return tuple(term for term in found if _fold(term.rendering) not in prose)


def _compile(rows: Sequence[tuple[int, Term]]) -> re.Pattern[str] | None:
    """One alternation over the given terms, in match order.

    Each branch is named after the term's position in the full match order
    rather than its position in this pattern, so a pattern built for a subset
    reports the same numbers as the pattern built for everything.

    One pattern rather than one per term because ``finditer`` over a single
    alternation is non-overlapping and resolves a tie at the same position by
    taking the earlier branch. Match order is longest first, so "context
    manager" is the earlier branch and wins the position that "context" would
    otherwise have taken. A loop of separate patterns would have to reimplement
    that, and would get it wrong in the direction that is hard to see.
    """
    if not rows:
        return None
    branches = [f"(?P<t{index}>{_branch(term.en)})" for index, term in rows]
    return re.compile(f"{_BEFORE}(?:{'|'.join(branches)}){_AFTER}")


def _branch(en: str) -> str:
    """One term, with only its first letter allowed to differ in case."""
    head, tail = en[0], en[1:]
    lead = (
        f"[{re.escape(head.lower())}{re.escape(head.upper())}]"
        if head.isalpha()
        else re.escape(head)
    )
    return lead + re.escape(tail)


def _fold(text: str) -> str:
    """Case-folded and composed, for the one comparison that is about presence.

    Normalised because a rendering typed on a Telex keyboard and the same
    rendering pasted out of a browser differ in normal form and not in meaning,
    and ``G02`` failing on that would be a bug report nobody could reproduce.
    """
    return unicodedata.normalize("NFC", text).casefold()


@dataclass(frozen=True, slots=True, kw_only=True)
class Diff:
    """What changed between two versions.

    ``changed`` carries both renderings because the question a reviewer asks of
    a bump is never "which terms moved" on its own, it is "what did this one
    become".
    """

    old: int
    new: int
    added: tuple[Term, ...] = ()
    removed: tuple[Term, ...] = ()
    changed: tuple[tuple[Term, Term], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def __len__(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)

    @property
    def terms(self) -> frozenset[str]:
        """Every English side a run has to reconsider.

        Additions are in here as well as changes, and that is not an oversight.
        A term added at version 8 was translated at version 7 by a model that
        was never told about it, so those entries are exactly as stale as the
        ones whose rendering was overturned.
        """
        return frozenset(
            [
                *(term.en for term in self.added),
                *(term.en for term in self.removed),
                *(after.en for _, after in self.changed),
            ]
        )


def diff(old: Glossary, new: Glossary) -> Diff:
    """Compare two versions row by row."""
    before, after = old.by_en, new.by_en
    changed = tuple(
        (before[en], after[en])
        for en in sorted(before.keys() & after.keys())
        if _differs(before[en], after[en])
    )
    return Diff(
        old=old.version,
        new=new.version,
        added=tuple(after[en] for en in sorted(after.keys() - before.keys())),
        removed=tuple(before[en] for en in sorted(before.keys() - after.keys())),
        changed=changed,
    )


def _differs(before: Term, after: Term) -> bool:
    """Whether a row changed in a way a translation could notice.

    ``note`` is not in here. A note is written for the person reading the file
    and changing one is not a reason to re-queue a few hundred entries.
    """
    return (before.vi, before.keep_en, before.identifier, before.context) != (
        after.vi,
        after.keep_en,
        after.identifier,
        after.context,
    )


def _changed(before: Sequence[Term], after: Sequence[Term]) -> bool:
    return bool(
        diff(Glossary(version=0, terms=tuple(before)), Glossary(version=0, terms=tuple(after)))
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Rejection:
    """One rule a row broke, named by the rule that caught it."""

    rule: str
    en: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.en!r} {self.detail}"


def check(glossary: Glossary) -> list[Rejection]:
    """``G-e`` and ``G-f``, the two rules that are about a list rather than a row.

    The per-row rules ``G-a`` through ``G-d`` live in :mod:`pydocvi.curate`,
    because they are about an answer that came back from a model and there is
    nothing to check once a row is in the file.
    """
    return [*_collisions(glossary), *_shadowing(glossary)]


def _bare(word: str) -> str:
    """A word with the joins taken out, so "built-in" and "builtin" are one word."""
    return _fold(word).replace("-", "").replace("_", "").replace(" ", "")


def _inflected(one: str, other: str) -> bool:
    """Whether two English terms are one term wearing different endings.

    Vietnamese does not mark plural, so "file object" and "file objects" have
    the same rendering and there is no wording that would give them different
    ones. The matcher works on whole words, so "file object" does not match
    inside "file objects" and both rows have to be in the file for both forms to
    be enforced. A rule that forbade the pair would be a rule the glossary
    cannot satisfy, so this is what it excuses.

    It compares word by word first, because the ending that differs is often not
    at the end of the term: "backwards compatibility" against "backward
    compatibility".
    """
    if _bare(one) == _bare(other):
        return True
    left, right = one.split(), other.split()
    if len(left) != len(right):
        return False
    return all(_ending(this, that) for this, that in zip(left, right, strict=True))


def _ending(one: str, other: str) -> bool:
    """Whether two words differ by nothing but an English plural."""
    one, other = _bare(one), _bare(other)
    if one == other:
        return True
    shorter, longer = sorted((one, other), key=len)
    if longer in (shorter + "s", shorter + "es"):
        return True
    return shorter.endswith("y") and longer == shorter[:-1] + "ies"


def _collisions(glossary: Glossary) -> list[Rejection]:
    """``G-e``: two English terms may not share a rendering uncontextualised.

    A collision is how a reader searching for one thing finds another, and it is
    invisible in review because each row is defensible on its own. Terms that
    are the same word in singular and plural are not a collision, because a
    reader who finds both has found one thing. See :func:`_inflected`.
    """
    by_vi: dict[str, list[Term]] = {}
    for term in glossary:
        by_vi.setdefault(_fold(term.rendering), []).append(term)
    return [
        Rejection(
            rule="G-e",
            en=term.en,
            detail=f"renders as {term.rendering!r}, so does "
            f"{', '.join(repr(other.en) for other in clashing)}",
        )
        for group in by_vi.values()
        if len(group) > 1
        for term in group
        if any(other.context is None for other in group)
        if (clashing := [other for other in group if not _inflected(term.en, other.en)])
    ]


def _shadowing(glossary: Glossary) -> list[Rejection]:
    """``G-f``: a term containing another must be listed above it.

    The matcher sorts longest first regardless, so nothing translates wrongly
    when this fails. What fails is the file: a reader who finds "context" above
    "context manager" reads the list in the order it is written and concludes
    that the shorter one wins. Terminology is decided by people reading this
    file, so a file that reads wrongly is a real defect.
    """
    positions = {term.en: index for index, term in enumerate(glossary.terms)}
    return [
        Rejection(
            rule="G-f",
            en=short.en,
            detail=f"is contained in {long.en!r} and is listed above it",
        )
        for short in glossary
        for long in glossary
        if short.en != long.en
        and _contains(long.en, short.en)
        and positions[short.en] < positions[long.en]
    ]


def _contains(long: str, short: str) -> bool:
    """Whether one term contains another as whole words."""
    return re.search(f"{_BEFORE}{re.escape(short)}{_AFTER}", long) is not None


class GlossaryError(ValueError):
    """A glossary file that cannot be read as one."""


def loads(text: str) -> Glossary:
    """Read ``glossary.yaml``.

    Through a YAML reader rather than by hand, because this file is edited by
    people and a hand-rolled reader would accept exactly the subset the writer
    emits and reject the equivalent forms a person typed.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    try:
        raw = YAML(typ="safe").load(text)
    except YAMLError as error:
        raise GlossaryError(f"not readable as YAML: {error}") from error
    if not isinstance(raw, dict):
        raise GlossaryError("expected a mapping with 'version' and 'terms'")
    version = raw.get("version")
    if not isinstance(version, int):
        raise GlossaryError("'version' must be an integer")
    rows = raw.get("terms") or []
    if not isinstance(rows, list):
        raise GlossaryError("'terms' must be a list")
    return Glossary(version=version, terms=tuple(_row(row, at) for at, row in enumerate(rows, 1)))


def _row(raw: object, at: int) -> Term:
    if not isinstance(raw, dict):
        raise GlossaryError(f"term {at} is not a mapping")
    unknown = set(raw) - {"en", "vi", "keep_en", "identifier", "context", "note"}
    if unknown:
        raise GlossaryError(f"term {at} has unknown field(s): {', '.join(sorted(unknown))}")
    try:
        return Term(
            en=str(raw["en"]),
            vi=str(raw["vi"]),
            keep_en=bool(raw.get("keep_en", False)),
            identifier=bool(raw.get("identifier", False)),
            context=str(raw["context"]) if raw.get("context") else None,
            note=str(raw.get("note", "")),
        )
    except KeyError as error:
        raise GlossaryError(f"term {at} is missing {error}") from error
    except ValueError as error:
        raise GlossaryError(f"term {at}: {error}") from error


def dumps(glossary: Glossary) -> str:
    """Write ``glossary.yaml``.

    Hand-rolled, like the upstream pin, and for the same reason. This file is
    reviewed in a diff far more often than it is parsed, so key order has to be
    fixed, absent fields have to stay absent rather than appear as ``null``, and
    a row that did not change has to produce a line that did not change. A
    round-trip test holds this against :func:`loads`.
    """
    out = [
        "# Written by pydocvi glossary. Rows are in match order, longest first.",
        "# GLOSSARY.md is generated from this file and G05 fails when they disagree.",
        f"version: {glossary.version}",
        "terms:",
    ]
    for term in glossary:
        out.append(f"  - en: {scalar(term.en)}")
        out.append(f"    vi: {scalar(term.vi)}")
        if term.keep_en:
            out.append("    keep_en: true")
        if term.identifier:
            out.append("    identifier: true")
        if term.context is not None:
            out.append(f"    context: {scalar(term.context)}")
        if term.note:
            out.append(f"    note: {scalar(term.note)}")
    return "\n".join(out) + "\n"


def scalar(value: str) -> str:
    """Every string double quoted, with the two characters that matter escaped.

    Always quoted rather than only when needed. "yes", "no", "on" and "null" are
    booleans to a YAML reader, "3.15" is a float, and a glossary is precisely
    the kind of file that ends up containing all four.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load(path: Path) -> Glossary:
    try:
        return loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GlossaryError(f"cannot read {path}: {error}") from error


def save(glossary: Glossary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(glossary), encoding="utf-8")
    return path


#: Where the generated table starts and ends inside ``GLOSSARY.md``.
#:
#: Markers rather than "replace the last table", because the prose around them
#: contains tables of its own and a reviewer moving a section should not silently
#: change which one the tool overwrites.
TABLE_OPEN = "<!-- generated: terms -->"
TABLE_CLOSE = "<!-- /generated: terms -->"


def table(glossary: Glossary) -> str:
    """The term table as it appears in ``GLOSSARY.md``.

    Two columns and a note, in match order, because match order is the order the
    rows resolve in and a reviewer arguing about "list" against "list
    comprehension" needs to see which one the tool reaches first.
    """
    lines = [
        f"Version {glossary.version}. {len(glossary)} terms, "
        f"{len(glossary.kept)} of them kept in English.",
        "",
        "| English | Vietnamese | Notes |",
        "| --- | --- | --- |",
    ]
    for term in match_order(glossary.terms):
        vi = f"`{term.en}` (kept)" if term.keep_en else term.vi
        note = " ".join(
            part for part in (_context_note(term), _standalone_note(term), term.note) if part
        )
        lines.append(f"| {term.en} | {vi} | {_cell(note)} |")
    return "\n".join(lines)


def _standalone_note(term: Term) -> str:
    """Said in the table because a reviewer reading one row cannot infer it."""
    if not term.identifier or term.keep_en:
        return ""
    return f"An entry that is only `{term.en}` names the thing and stays English."


def _context_note(term: Term) -> str:
    return f"Only where the path or msgctxt contains `{term.context}`." if term.context else ""


def _cell(value: str) -> str:
    """A pipe inside a cell ends the cell, so it has to be spelled."""
    return value.replace("|", "\\|")


def render(markdown: str, glossary: Glossary) -> str:
    """Put the generated table back into ``GLOSSARY.md``, prose untouched.

    The prose is the half of this file no machine can write. Which second person
    pronoun to use and whether headings take noun form are decisions that touch
    more entries than every terminology row combined, and they live between
    these markers' neighbours rather than inside them.
    """
    start = markdown.find(TABLE_OPEN)
    end = markdown.find(TABLE_CLOSE)
    if start < 0 or end < 0 or end < start:
        raise GlossaryError(f"{TABLE_OPEN} and {TABLE_CLOSE} must both be present, in that order")
    head = markdown[: start + len(TABLE_OPEN)]
    tail = markdown[end:]
    return f"{head}\n\n{table(glossary)}\n\n{tail}"


def agrees(markdown: str, glossary: Glossary) -> list[Rejection]:
    """``G05``: the Markdown table and the YAML say the same thing.

    Compared as text rather than by parsing the table back into rows. The file
    is generated, so the only question worth asking is whether it was
    regenerated, and a parser would let a hand edit survive by being close
    enough.
    """
    try:
        wanted = render(markdown, glossary)
    except GlossaryError as error:
        return [Rejection(rule="G05", en="GLOSSARY.md", detail=str(error))]
    if wanted != markdown:
        return [
            Rejection(
                rule="G05",
                en="GLOSSARY.md",
                detail=f"is not what version {glossary.version} renders to, run glossary check --fix",
            )
        ]
    return []


@dataclass(frozen=True, slots=True, kw_only=True)
class Stats:
    """What ``glossary show`` prints."""

    version: int
    terms: int
    kept: int
    contextual: int
    noted: int
    problems: int = 0
    longest: str = ""
    by_words: dict[int, int] = field(default_factory=dict)


def stats(glossary: Glossary) -> Stats:
    order = match_order(glossary.terms)
    words: dict[int, int] = {}
    for term in glossary:
        count = len(term.en.split())
        words[count] = words.get(count, 0) + 1
    return Stats(
        version=glossary.version,
        terms=len(glossary),
        kept=len(glossary.kept),
        contextual=sum(1 for term in glossary if term.context is not None),
        noted=sum(1 for term in glossary if term.note),
        problems=len(check(glossary)),
        longest=order[0].en if order else "",
        by_words=dict(sorted(words.items())),
    )
