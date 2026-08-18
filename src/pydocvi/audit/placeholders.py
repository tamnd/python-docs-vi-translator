"""``P01`` to ``P08``: every translated entry against its ``msgid``.

These are the invariants of the translation stage run again over the committed
corpus rather than over one answer. A batch that passed at translation time and
a corpus that fails here means something wrote a catalog outside the pipeline,
which is worth knowing and is otherwise invisible.

**The numbering is not the same as :mod:`pydocvi.invariants`, and the overlap is
the trap.** Both groups are called ``P``, both run from 1, and they do not line
up: audit ``P05`` is format specifiers, which is ``P09`` at translation time,
while translation-time ``P05`` is "empty or identical to the source", which is
audit ``L02``. The two lists were numbered independently in the specs and
renumbering either would silently invalidate every issue comment and report
that quotes the old id, so they are mapped here instead, once, in one table:

===========  ==============================  ==========================
Audit        What it checks                  Translation-time rule
===========  ==============================  ==========================
``P01``      protected spans, same count     ``P01``
``P02``      no ``⟦n⟧`` survives             ``P02``
``P03``      role targets unchanged          none, new here
``P04``      surrounding whitespace          ``P04``
``P05``      format specifiers               ``P09``
``P06``      link targets unchanged          none, new here
``P07``      code entries copied exactly     none, new here
``P08``      no fence, no horizontal rule    ``P07``
===========  ==============================  ==========================

The three that are new are all things that cannot be checked from one answer in
isolation and can be checked from a committed catalog.
"""

import re
from collections.abc import Iterator

from pydocvi import classify, invariants
from pydocvi.audit.model import Corpus, Finding, Group, Registry
from pydocvi.segment import FORMATS, PLACEHOLDER_LOOSE, spans_of

registry = Registry()
check = registry.check

#: A reST role and the target inside its backticks. The target is the part that
#: has to survive: ``:func:`len`` translated to ``:func:`chiều dài`` is a link
#: to nothing, and 53.7 % of entries in this corpus carry at least one.
_ROLE_TARGET = re.compile(r":([a-zA-Z0-9_+:.-]+):`([^`]*)`")

#: A link with an explicit target, ``` `text <url>`_ ```. The text is prose and
#: should be translated. The URL is not and must not be.
_LINK_TARGET = re.compile(r"`[^`<>]+?\s*<([^>]*)>`__?")

#: A fence or a horizontal rule at the start of an answer.
_FENCE = re.compile(r"^\s*(?:```|~~~|---)")


@check("P01", Group.PLACEHOLDERS, hard=True, title="protected spans survive")
def p01_spans_survive(corpus: Corpus) -> Iterator[Finding]:
    """Every protected span in the ``msgid`` appears in the ``msgstr``, same count.

    Order is not checked and never will be. Vietnamese puts modifiers on the
    other side of what they modify, so a translation that moves a role span is
    doing its job.

    Counted by occurrence in the string rather than by re-running the protector
    over the Vietnamese. Re-protecting would ask a different question: whether
    the translation contains the same *constructs*, which it can fail for
    reasons that are not defects, because what a span matches depends on the
    characters either side of it and those are exactly what translating
    changes. The count is the property :func:`segment.restore` enforces at
    translation time, so this is that rule and not a second one near it.
    """
    for one, entry in corpus.translated():
        for span in dict.fromkeys(spans_of(entry.msgid)):
            wanted = entry.msgid.count(span)
            got = entry.msgstr.count(span)
            if wanted != got:
                yield Finding(
                    check="P01",
                    path=corpus.relative(one.path),
                    line=entry.line,
                    detail=f"{span!r} appears {got} time(s), the source has {wanted}",
                    english=entry.msgid,
                    got=entry.msgstr,
                    segment=entry.id,
                )


@check("P02", Group.PLACEHOLDERS, hard=True, title="no placeholder marker survives")
def p02_no_marker_survives(corpus: Corpus) -> Iterator[Finding]:
    """No ``⟦n⟧`` reaches a committed ``msgstr``.

    A marker in the corpus is a restoration that did not happen, and it renders
    to the reader as two characters nobody can type looking for what they mean.
    """
    for one, entry in corpus.translated():
        if found := PLACEHOLDER_LOOSE.findall(entry.msgstr):
            yield Finding(
                check="P02",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"unrestored placeholder(s): {' '.join(found)}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("P03", Group.PLACEHOLDERS, hard=True, title="role targets unchanged")
def p03_role_targets_unchanged(corpus: Corpus) -> Iterator[Finding]:
    """Every ``:role:`target`` in the ``msgstr`` has its target unchanged.

    A translated target is a broken cross-reference. It renders as a link to
    nothing, Sphinx reports it in a build nobody in this project runs, and a
    reader following it lands nowhere.

    This is the check the old Google pipeline structurally cannot pass, because
    there is no way to tell Google Translate that ``:func:`` is not prose, and
    it is one of the two the M6 comparison turns on.

    Mostly redundant with ``P01``, and deliberately so. A role is a protected
    span, so anything this pipeline writes has already been checked twice over.
    What is left is the case the audit exists for: entries that reached the
    corpus without going through the protector, which is every human
    translation inherited from Transifex and anything written by hand
    afterwards. Those have no span record at all, and ``P01`` cannot see them.
    """
    for one, entry in corpus.translated():
        wanted = sorted(_ROLE_TARGET.findall(entry.msgid))
        got = sorted(_ROLE_TARGET.findall(entry.msgstr))
        if wanted == got:
            continue
        for role, target in _pairs_missing(wanted, got):
            yield Finding(
                check="P03",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f":{role}:`{target}` is not in the translation",
                english=f":{role}:`{target}`",
                got=_nearest(role, got),
                segment=entry.id,
            )


@check("P04", Group.PLACEHOLDERS, hard=True, title="surrounding whitespace matches")
def p04_whitespace_matches(corpus: Corpus) -> Iterator[Finding]:
    """Leading and trailing whitespace, and the trailing newline, match.

    gettext concatenates these strings into rendered pages, so a lost trailing
    space is two words run together somewhere a long way from here.
    """
    for one, entry in corpus.translated():
        source = invariants.edges(entry.msgid)
        translation = invariants.edges(entry.msgstr)
        for index, edge in enumerate(("leading", "trailing")):
            if source[index] != translation[index]:
                yield Finding(
                    check="P04",
                    path=corpus.relative(one.path),
                    line=entry.line,
                    detail=f"{edge} whitespace {source[index]!r} became {translation[index]!r}",
                    english=entry.msgid,
                    got=entry.msgstr,
                    segment=entry.id,
                )


@check("P05", Group.PLACEHOLDERS, hard=True, title="format specifiers match")
def p05_format_specifiers_match(corpus: Corpus) -> Iterator[Finding]:
    """Format specifiers: same set, same count.

    A dropped ``%s`` is a ``TypeError`` at runtime in whatever program copied
    the string, which makes this the one rule in the group whose failures escape
    the documentation entirely.
    """
    for one, entry in corpus.translated():
        wanted = sorted(FORMATS.findall(entry.msgid))
        got = sorted(FORMATS.findall(entry.msgstr))
        if wanted != got:
            yield Finding(
                check="P05",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"source has {wanted}, translation has {got}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("P06", Group.PLACEHOLDERS, hard=True, title="link targets unchanged")
def p06_link_targets_unchanged(corpus: Corpus) -> Iterator[Finding]:
    """The URL inside ``` `text <url>`_ ``` is unchanged.

    The visible text is prose and should be translated. The URL is not prose and
    a translated one is a dead link, which is the same failure as ``P03`` in a
    different piece of syntax, and it is here for the same reason: an inherited
    human entry never met the protector, so ``P01`` has nothing to check it
    against.
    """
    for one, entry in corpus.translated():
        wanted = sorted(_LINK_TARGET.findall(entry.msgid))
        got = sorted(_LINK_TARGET.findall(entry.msgstr))
        if wanted != got:
            yield Finding(
                check="P06",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"link target(s) changed: {_difference(wanted, got)}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("P07", Group.PLACEHOLDERS, hard=True, title="code entries copied exactly")
def p07_code_is_byte_identical(corpus: Corpus) -> Iterator[Finding]:
    """A doctest or a literal block is byte-identical to its ``msgid``.

    These are never translated, they are copied, so any difference at all is a
    corruption rather than a translation choice. One changed character in
    ``>>> sorted(d.keys())`` is a broken example that a reader will type and
    then have to debug.

    This is also the check that would have caught the classifier calling code
    prose, from the other end: the entries a model was asked to translate and
    correctly refused to change would have arrived here identical, and the ones
    it did change would have arrived here as findings.
    """
    for one, entry in corpus.translated():
        kind = classify.classify(entry.msgid)
        if kind not in {classify.Kind.DOCTEST, classify.Kind.LITERAL_BLOCK}:
            continue
        if entry.msgstr != entry.msgid:
            yield Finding(
                check="P07",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"{kind} entry differs from its source",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("P08", Group.PLACEHOLDERS, hard=True, title="no fence or horizontal rule")
def p08_no_fence(corpus: Corpus) -> Iterator[Finding]:
    """No ``msgstr`` opens a fence or a ``---`` that its ``msgid`` did not.

    Both are a model formatting its answer rather than answering, and both are
    valid reST that renders as something the English does not.

    Unless the English opened the same way, which is the clause this check was
    missing. Every other rule in this module reads the pair, and this one read
    the translation alone, so a ``msgid`` of ``---`` copied through as ``---``
    was reported as a model drawing a rule under its answer. Two entries in the
    corpus are that: a literal ``---`` in ``c-api/call.po`` and the inheritance
    diagram in ``howto/mro.po``, which opens with a line of dashes because it is
    a picture of a class hierarchy.
    """
    for one, entry in corpus.translated():
        if _FENCE.match(entry.msgstr) and not _FENCE.match(entry.msgid):
            yield Finding(
                check="P08",
                path=corpus.relative(one.path),
                line=entry.line,
                detail="opens with a fence or a horizontal rule",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


def _difference(wanted: list[str], got: list[str]) -> str:
    """What is in the source and not the translation, and the reverse."""
    missing = _remaining(wanted, got)
    extra = _remaining(got, wanted)
    parts = []
    if missing:
        parts.append(f"missing {' '.join(missing)}")
    if extra:
        parts.append(f"unexpected {' '.join(extra)}")
    return ", ".join(parts) or "counts differ"


def _remaining(left: list[str], right: list[str]) -> list[str]:
    """Everything in ``left`` that ``right`` does not also account for.

    By multiplicity rather than by set, because a span that appears twice in the
    English and once in the Vietnamese is a real failure that a set difference
    reports as nothing at all.
    """
    pool = list(right)
    out = []
    for one in left:
        if one in pool:
            pool.remove(one)
        else:
            out.append(one)
    return out


def _pairs_missing(
    wanted: list[tuple[str, str]], got: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    pool = list(got)
    out = []
    for one in wanted:
        if one in pool:
            pool.remove(one)
        else:
            out.append(one)
    return out


def _nearest(role: str, got: list[tuple[str, str]]) -> str:
    """What the translation has under the same role, for the report.

    Naming what did come back rather than only what did not, because
    ``:func:`asyncio.chạy` ← :func:`asyncio.run`` is a fix and "a target is
    missing" is a search.
    """
    same = [target for name, target in got if name == role]
    return f":{role}:`{same[0]}`" if same else ""
