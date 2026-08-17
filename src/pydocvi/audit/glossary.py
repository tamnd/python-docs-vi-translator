"""``G01`` to ``G06``: the terminology, and whether the corpus kept to it.

Soft until M7, and the reason is honest rather than tactical. The glossary is
1 000 rows curated by a model and reviewed by nobody yet, so a corpus failing
``G02`` today is as likely to be evidence against the row as against the
translation. M7 is where a person reads the table, and hardening these before
that would be enforcing a document nobody has agreed to.

``G01`` and ``G05`` are wrappers. The rules themselves live in
:mod:`pydocvi.glossary`, because ``glossary check`` runs them at edit time where
a bad row can still be fixed cheaply, and a second implementation here would be
a second answer to the same question.
"""

from collections.abc import Iterator

from pydocvi import glossary as terminology
from pydocvi.audit.model import Corpus, Finding, Group, Registry
from pydocvi.catalog import Catalog, Entry

registry = Registry()
check = registry.check


@check("G01", Group.GLOSSARY, hard=False, title="the glossary is self-consistent")
def g01_glossary_is_consistent(corpus: Corpus) -> Iterator[Finding]:
    """No duplicate English, no rendering collision, no row shadowing another.

    Run here as well as at edit time because a glossary can also become
    inconsistent without anyone editing it, by a merge that takes both sides of
    a conflicting row.
    """
    if corpus.glossary is None:
        return
    for rejection in terminology.check(corpus.glossary):
        yield Finding(
            check="G01",
            path="glossary.yaml",
            detail=f"{rejection.rule}: {rejection.detail}",
            english=rejection.en,
        )


@check("G02", Group.GLOSSARY, hard=False, title="glossary renderings are used")
def g02_renderings_are_used(corpus: Corpus) -> Iterator[Finding]:
    """A term in the English means its rendering is in the Vietnamese.

    Presence, not agreement, because Vietnamese inflects a phrase by the words
    beside it and moves modifiers around, so anything stricter fails on correct
    translations.
    """
    for one, entry, term in _missing(corpus):
        if term.keep_en:
            continue
        yield Finding(
            check="G02",
            path=corpus.relative(one.path),
            line=entry.line,
            detail=f"{term.en!r} should render as {term.vi!r}",
            english=entry.msgid,
            got=entry.msgstr,
            segment=entry.id,
        )


@check("G03", Group.GLOSSARY, hard=False, title="kept terms stayed in English")
def g03_kept_terms_stay_english(corpus: Corpus) -> Iterator[Finding]:
    """A ``keep_en`` term is still in English in the translation.

    ``G02`` read the other way round, and split from it because the two failures
    call for opposite fixes. A missing rendering is a translation that ignored
    the glossary; a translated ``keep_en`` term is a translation that invented a
    Vietnamese word for something Vietnamese programmers say in English.
    """
    for one, entry, term in _missing(corpus):
        if not term.keep_en:
            continue
        yield Finding(
            check="G03",
            path=corpus.relative(one.path),
            line=entry.line,
            detail=f"{term.en!r} is kept in English and did not survive",
            english=entry.msgid,
            got=entry.msgstr,
            segment=entry.id,
        )


@check("G04", Group.GLOSSARY, hard=False, title="no untranslated glossary term")
def g04_no_english_term_survives(corpus: Corpus) -> Iterator[Finding]:
    """No English term that has a Vietnamese rendering is left standing.

    The complement of ``G02``: that one catches a translation with the term
    missing, this one catches a translation with the term still in English.
    Both can be true of the same entry, and an entry that fails only this one is
    the more interesting case, because it means the sentence was translated
    around a word that was left alone.

    Matched over the stripped text, so a term inside a role or a literal is not
    counted. ``:func:`open`` is not the word "open".
    """
    if corpus.glossary is None:
        return
    matcher = corpus.glossary.matcher()
    for one, entry in corpus.translated():
        where = corpus.relative(one.path)
        left = matcher.select(entry.msgstr, where=where, msgctxt=entry.msgctxt)
        for term in left:
            if term.keep_en:
                continue
            yield Finding(
                check="G04",
                path=where,
                line=entry.line,
                detail=f"{term.en!r} is still in English, and renders as {term.vi!r}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("G05", Group.GLOSSARY, hard=False, title="GLOSSARY.md matches the YAML")
def g05_markdown_agrees(corpus: Corpus) -> Iterator[Finding]:
    """The generated table in ``GLOSSARY.md`` is what the YAML renders to.

    Compared as text rather than parsed back, because the file is generated and
    the only question worth asking is whether it was regenerated. A parser would
    let a hand edit survive by being close enough.
    """
    if corpus.glossary is None or corpus.markdown is None:
        return
    for rejection in terminology.agrees(corpus.markdown, corpus.glossary):
        yield Finding(check="G05", path="GLOSSARY.md", detail=rejection.detail)


@check("G06", Group.GLOSSARY, hard=False, title="no entry is on a stale glossary")
def g06_glossary_version_is_current(corpus: Corpus) -> Iterator[Finding]:
    """Every machine translation was made against the current glossary version.

    Counted rather than fixed. A version bump touches a few hundred entries out
    of 87 008, and ``stale --glossary`` is the command that re-queues them; this
    check exists so that the number is visible rather than discovered later by
    somebody wondering why a term reads two ways in two files.

    Human and legacy entries are skipped. A person's translation was not made
    against a glossary version and marking it stale would be claiming the
    glossary outranks them, which it does not.
    """
    if corpus.glossary is None or corpus.memory is None:
        return
    current = corpus.glossary.version
    for one, entry in corpus.translated():
        known = corpus.memory.lookup(entry.msgid, entry.msgctxt)
        if known is None or known.source != "machine":
            continue
        if known.glossary != current:
            yield Finding(
                check="G06",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"translated against glossary version {known.glossary}, current is {current}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


def _missing(corpus: Corpus) -> Iterator[tuple[Catalog, Entry, terminology.Term]]:
    """Every term that appeared in a ``msgid`` and not in its ``msgstr``.

    Shared by ``G02`` and ``G03``, which are one question asked of the two kinds
    of row and would otherwise walk 87 008 entries twice to ask it.
    """
    if corpus.glossary is None:
        return
    matcher = corpus.glossary.matcher()
    for one, entry in corpus.translated():
        where = corpus.relative(one.path)
        for term in matcher.missing(entry.msgid, entry.msgstr, where=where, msgctxt=entry.msgctxt):
            yield one, entry, term
