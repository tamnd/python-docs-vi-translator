"""``S01`` to ``S08``: the catalogs against the upstream pin.

All hard. A structural failure explains most of what would follow it, which is
why this group prints first and why a run that fails here is worth reading
before anything else in the report.

``S02`` is the one that matters most. An edited ``msgid`` is the single most
damaging thing anyone can do to a catalog: gettext keys on it, so changing one
character silently detaches the translation from the string it belongs to, and
nothing downstream can tell that from an entry nobody has got to yet.
"""

from collections.abc import Iterator

from pydocvi import apply, sync
from pydocvi.audit.model import Corpus, Finding, Group, Registry
from pydocvi.catalog import Entry

registry = Registry()
check = registry.check


@check("S01", Group.STRUCTURE, hard=True, title="the pin's counts match a recount")
def s01_counts_match_the_pin(corpus: Corpus) -> Iterator[Finding]:
    """Every count in ``manifests/upstream.yaml`` matches a recount.

    The pin is what every other number in the project is quoted against, so a
    pin that has drifted turns every report into a confident statement about a
    corpus that is not there any more.
    """
    if corpus.pin is None:
        return
    pin = sync.Pin.read(corpus.pin)
    if pin is None:
        yield Finding(
            check="S01",
            path="manifests/upstream.yaml",
            detail="no upstream pin on record, so nothing can be checked against it",
        )
        return
    if not corpus.upstream:
        return
    files, entries, words, characters, translated = sync.measure(corpus.upstream.values())
    for name, recorded, measured in (
        ("files", pin.files, files),
        ("entries", pin.entries, entries),
        ("words", pin.words, words),
        ("characters", pin.characters, characters),
        ("translated", pin.translated, translated),
    ):
        if recorded != measured:
            yield Finding(
                check="S01",
                path="manifests/upstream.yaml",
                detail=f"pin says {recorded:,} {name}, a recount finds {measured:,}",
            )


@check("S02", Group.STRUCTURE, hard=True, title="no msgid was edited")
def s02_every_msgid_exists_upstream(corpus: Corpus) -> Iterator[Finding]:
    """Every ``msgid`` in the content repo exists in the pin, byte-identical.

    This is the one that catches a ``msgid`` edit. Comparing by segment id
    rather than by position, because an id is a hash of the ``msgid`` and the
    ``msgctxt``, so an entry whose English was touched by one character does
    not merely differ here, it stops existing.
    """
    for one, entry in corpus.translated():
        relative = corpus.relative(one.path)
        source = corpus.upstream.get(relative)
        if source is None:
            yield Finding(
                check="S02",
                path=relative,
                line=entry.line,
                detail="no such file in the upstream pin",
                segment=entry.id,
            )
            continue
        if entry.id not in source.by_id():
            yield Finding(
                check="S02",
                path=relative,
                line=entry.line,
                detail="msgid is not in the upstream pin, so it was edited here",
                english=entry.msgid,
                segment=entry.id,
            )


@check("S03", Group.STRUCTURE, hard=True, title="entry count, order and msgctxt match upstream")
def s03_order_matches_upstream(corpus: Corpus) -> Iterator[Finding]:
    """The catalogs are the upstream catalogs, in the upstream order.

    Order is checked and not just membership, because a reviewer reads a
    catalog top to bottom against the English file beside it, and a corpus that
    has quietly been sorted costs them that.
    """
    for one in corpus.catalogs:
        relative = corpus.relative(one.path)
        source = corpus.upstream.get(relative)
        if source is None:
            yield Finding(check="S03", path=relative, detail="no such file in the upstream pin")
            continue
        if len(one) != len(source):
            yield Finding(
                check="S03",
                path=relative,
                detail=f"{len(one):,} entries here against {len(source):,} upstream",
            )
            continue
        for entry, expected in zip(one, source, strict=True):
            if entry.id != expected.id:
                yield Finding(
                    check="S03",
                    path=relative,
                    line=entry.line,
                    detail="entry is not the upstream entry at this position",
                    english=expected.msgid,
                    got=entry.msgid,
                    segment=entry.id,
                )
                break


@check("S04", Group.STRUCTURE, hard=True, title="no unmarked machine translation")
def s04_translated_entries_are_marked(corpus: Corpus) -> Iterator[Finding]:
    """No entry carries a ``msgstr`` without either ``fuzzy`` or a human record.

    Everything this tool writes is fuzzy, because Sphinx renders the English for
    a fuzzy string and the worst a reader meets is the English they would have
    met anyway. An unmarked machine string is the opposite: it reads to a
    reviewer, and to the renderer, as somebody's considered work.
    """
    memory = corpus.memory
    for one, entry in corpus.translated():
        if entry.fuzzy:
            continue
        known = memory.lookup(entry.msgid) if memory is not None else None
        if known is not None and known.source == "human":
            continue
        yield Finding(
            check="S04",
            path=corpus.relative(one.path),
            line=entry.line,
            detail="translated, not fuzzy, and not human in the memory",
            english=entry.msgid,
            got=entry.msgstr,
            segment=entry.id,
        )


@check("S05", Group.STRUCTURE, hard=True, title="format flags preserved")
def s05_format_flags_are_preserved(corpus: Corpus) -> Iterator[Finding]:
    """Every ``python-format`` and ``python-brace-format`` flag survives.

    The flag is what tells gettext to validate the specifiers, so dropping it
    turns a checked string into an unchecked one and the failure surfaces at
    runtime in somebody's traceback rather than here.
    """
    for one, entry, source in corpus.paired():
        if source is None:
            continue
        wanted = {flag for flag in source.flags if flag.endswith("-format")}
        held = {flag for flag in entry.flags if flag.endswith("-format")}
        if missing := wanted - held:
            yield Finding(
                check="S05",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"upstream flag dropped: {', '.join(sorted(missing))}",
                english=entry.msgid,
                segment=entry.id,
            )


@check("S06", Group.STRUCTURE, hard=True, title="obsolete entries preserved")
def s06_obsolete_entries_survive(corpus: Corpus) -> Iterator[Finding]:
    """Obsolete entries are kept rather than dropped.

    An obsolete entry is a translation upstream stopped using, and it is the
    cheapest thing in the corpus: if the string comes back, and they do come
    back, the work is already done. Dropping them is a silent one-way loss.
    """
    for one in corpus.catalogs:
        relative = corpus.relative(one.path)
        source = corpus.upstream.get(relative)
        if source is None:
            continue
        held = {entry.id for entry in one}
        for entry in source:
            if _obsolete(entry) and entry.id not in held:
                yield Finding(
                    check="S06",
                    path=relative,
                    line=entry.line,
                    detail="obsolete entry present upstream and dropped here",
                    english=entry.msgid,
                    segment=entry.id,
                )


@check("S07", Group.STRUCTURE, hard=True, title="no msgid_plural")
def s07_no_plurals(corpus: Corpus) -> Iterator[Finding]:
    """No ``msgid_plural`` appears anywhere.

    There is none in the corpus today. If CPython adds one, this fails loudly,
    which is the point: Vietnamese has one plural form and the correct
    ``nplurals`` header is a decision for a person, not something a pipeline
    should quietly guess at the first time it meets the case.
    """
    for one in corpus.catalogs:
        for entry in one:
            if any(comment.startswith("msgid_plural") for comment in entry.comments) or any(
                text.startswith("msgid_plural") for text in entry.raw or ()
            ):
                yield Finding(
                    check="S07",
                    path=corpus.relative(one.path),
                    line=entry.line,
                    detail="msgid_plural needs an nplurals decision from a person",
                    english=entry.msgid,
                    segment=entry.id,
                )


@check("S08", Group.STRUCTURE, hard=True, title="apply --check is byte-identical")
def s08_apply_is_byte_identical(corpus: Corpus) -> Iterator[Finding]:
    """The committed catalogs are exactly what the memory and upstream produce.

    Determinism is the whole claim of this pipeline: the corpus is a function of
    the pin, the memory and the glossary, and anyone can rerun it and get the
    same bytes. A file that differs here was written by something outside the
    pipeline, and the difference is the only evidence that will ever exist of
    whatever that was.
    """
    if corpus.memory is None or corpus.upstream_root is None or corpus.stamp is None:
        return
    sources = [corpus.upstream_root / relative for relative in sorted(corpus.upstream)]
    if not sources:
        return
    result = apply.check(
        sources,
        corpus.memory,
        root=corpus.upstream_root,
        into=corpus.root,
        stamp=corpus.stamp,
    )
    for one in result.plans:
        if one.changed:
            yield Finding(
                check="S08",
                path=corpus.relative(one.path),
                detail="committed bytes differ from what apply would write",
            )


def _obsolete(entry: Entry) -> bool:
    """Whether an entry is one gettext has marked as no longer used."""
    return "obsolete" in entry.flags or any(one.startswith("#~") for one in entry.comments)
