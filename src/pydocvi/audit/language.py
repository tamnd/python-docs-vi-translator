"""``L01`` to ``L08``: whether the Vietnamese is Vietnamese, and whether it reads.

Four hard and four soft. The soft ones are soft on purpose: ``L05``, ``L06`` and
``L08`` each have real exceptions, a heading that is genuinely an instruction or
a sentence Vietnamese splits in two, and a hard rule that a correct translation
breaks is not a rule but a source of noise that ends in the whole audit being
switched off. ``L07`` is soft because a cut-down model's output is worth redoing
and is not, on its own, wrong.

Spec 03 §5 calls the function-word check ``L07`` and spec 02 §4 calls the
provenance check ``L08``. The audit spec's own table numbers them ``L04`` and
``L07``, and that table is the one the reports and the issue comments quote, so
it is the one implemented here.
"""

import re
import unicodedata
from collections.abc import Iterator, Sequence

from pydocvi import apply, classify, invariants, textguard
from pydocvi.audit.model import Corpus, Finding, Group, Registry
from pydocvi.segment import strip_markup

registry = Registry()
check = registry.check

#: The kinds that are copied rather than translated. Their ``msgstr`` is the
#: ``msgid``, so the rules about being written in Vietnamese do not apply.
PASSTHROUGH = frozenset(
    {
        classify.Kind.NOOP,
        classify.Kind.DOCTEST,
        classify.Kind.LITERAL_BLOCK,
        classify.Kind.VERSION_MARKER,
    }
)

#: Second-person pronouns ``GLOSSARY.md`` rules out. Vietnamese pronouns encode
#: the age and the relationship between the speakers, and documentation has no
#: business assuming either about its reader.
FORBIDDEN_PRONOUNS = ("anh", "chị", "em")

#: The neutral second-person pronoun, allowed where the English addresses the
#: reader and to be dropped everywhere else.
READER = "bạn"

#: Ordinary compounds that happen to be spelled out of pronouns. "anh em" is
#: "siblings", and the corpus uses it for "sibling class" six times in
#: ``library/functions.po`` alone, where the ``super`` documentation explains
#: the method resolution order. Removed before the pronoun scan rather than
#: exempted afterwards, because the words are not pronouns in that position at
#: all and reporting them would be reporting the wrong thing about a correct
#: sentence.
COMPOUNDS = ("anh em",)

#: Verbs that open an instruction to the reader. English addresses a reader
#: without a pronoun more often than with one, and documentation does it in a
#: small and repetitive vocabulary, so a first-word test against this list is
#: most of what a parser would find at none of the cost. Deliberately without
#: "return", "raise" or "yield": those open a description of what a function
#: does, not an instruction to whoever is reading.
READER_IMPERATIVES = frozenset(
    {
        "avoid",
        "beware",
        "call",
        "check",
        "compare",
        "consider",
        "do",
        "ensure",
        "install",
        "make",
        "note",
        "notice",
        "please",
        "read",
        "refer",
        "remember",
        "run",
        "see",
        "try",
        "use",
    }
)

#: Vietnamese for "let us", which opens an exhortation. A heading is a noun
#: phrase naming a section, not an invitation to do something. Matched with its
#: tone mark, because without one it is ``hay``, which means "or".
HORTATIVE = "hãy"

#: Substrings that name a smaller variant of a model. Matched rather than an
#: allowlist of good models, because the failure this catches is a route
#: quietly serving something cheaper, and an allowlist would have to be edited
#: every time a legitimate model is added or the check starts lying.
CUT_DOWN = ("mini", "nano", "lite", "small", "turbo", "flash", "haiku", "instant")

#: Above this many characters a heading is a sentence and not a heading. Chosen
#: off the corpus rather than by taste: the longest section title in the
#: Python documentation is comfortably under it.
HEADING = 60

#: How far the sentence counts may differ before it is worth reporting.
SENTENCE_SLACK = 1

#: How many capitalised words past the first make a heading title-cased. Two
#: rather than one, because Vietnamese capitalises a proper noun mid-heading
#: exactly as English does, and one is not a pattern.
TITLE_CASE_WORDS = 2

#: An English word. Kept to the unaccented letters on purpose, because the two
#: places it is used read a ``msgid``, and a ``msgid`` is English.
_WORD = re.compile(r"[a-z]+")

#: A Vietnamese word. Every letter that is not a digit or an underscore, so the
#: accented vowels are inside the word rather than splitting it in two.
_VIETNAMESE_WORD = re.compile(r"[^\W\d_]+")

_SENTENCE = re.compile(r"[.!?](?:\s|$)")


@check("L01", Group.LANGUAGE, hard=True, title="the translation is in Vietnamese")
def l01_is_vietnamese(corpus: Corpus) -> Iterator[Finding]:
    """A long ``msgstr`` carries at least one Vietnamese diacritic.

    The weakest rule in the file and knowingly so. A short string with no
    diacritics is ordinary, because "API" and "Unicode" are what a Vietnamese
    reader expects to see, so only a long diacritic-free string is suspicious.
    It catches the failure it exists for, which is a batch that came back in
    English and was committed.
    """
    for one, entry in corpus.translated():
        if classify.classify(entry.msgid) in PASSTHROUGH:
            continue
        stripped = strip_markup(entry.msgstr).strip()
        if len(stripped) <= invariants.SHORT or invariants.vietnamese(stripped):
            continue
        yield Finding(
            check="L01",
            path=corpus.relative(one.path),
            line=entry.line,
            detail=f"{len(stripped)} characters of prose with no Vietnamese diacritic",
            english=entry.msgid,
            got=entry.msgstr,
            segment=entry.id,
        )


@check("L02", Group.LANGUAGE, hard=True, title="no entry is the English verbatim")
def l02_not_the_source(corpus: Corpus) -> Iterator[Finding]:
    """A prose entry's ``msgstr`` is not its ``msgid`` handed back.

    Passthrough entries are exempt, because being identical to the source is
    what they are for. Everything else that is identical is either a model
    refusing the work or a classifier that should have called it passthrough,
    and both are worth a line in the report.

    An entry whose whole ``msgid`` is a term the glossary marks as standing
    alone is exempt too, and that exemption is the reason the glossary carries
    the flag. Narrowing the identifier rule put 6 558 single-word entries into
    :meth:`Corpus.prose` for the first time and took this check from 10 findings
    to 144. A third of them were entries reading ``sys``, ``builtins``,
    ``import``, ``exec`` and ``NaN``, all of which are index entries naming a
    module or a statement, and all of which a reviewer left in English because
    that is what a Vietnamese programmer calls them.

    Nothing in the string can tell those from ``module``, ``object`` and
    ``type``, which are the other 89 and are ordinary English words used as
    index categories. ``sys`` and ``Notes`` are the same shape, and that is the
    discrimination the classifier was narrowed for being unable to make. So it
    is made once, by hand, in the glossary, where it is a written decision
    rather than an exception buried here.

    Read from :attr:`Glossary.standalone` and not from ``keep_en``, because the
    two questions came apart. ``float`` is ``số thực`` in a sentence and the
    name of a C type in the table of ``struct`` format codes, and while this
    check read ``keep_en`` a row could only answer one of those. 69 of the 94
    findings ``G03`` was reporting were correct translations of ``type`` and
    ``list``, held there by rows that said "keep this in English" when what they
    meant was "leave the table cell alone".

    Matched on the whole ``msgid`` and not on a substring. A term inside a
    sentence says nothing about whether the sentence was translated, and this
    check is about the entry. Nor is the match folded or de-inflected: ``Lists``
    is a section heading three times in the corpus, followed each time by prose
    beginning "Lists are mutable sequences", and a heading is translated.
    """
    kept = (
        {term.en for term in corpus.glossary.standalone} if corpus.glossary is not None else set()
    )
    for one, entry in corpus.translated():
        if classify.classify(entry.msgid) in PASSTHROUGH:
            continue
        if entry.msgid.strip() in kept:
            continue
        if entry.msgstr.strip() == entry.msgid.strip():
            yield Finding(
                check="L02",
                path=corpus.relative(one.path),
                line=entry.line,
                detail="identical to the English",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("L03", Group.LANGUAGE, hard=True, title="no narration")
def l03_no_narration(corpus: Corpus) -> Iterator[Finding]:
    """No entry contains the model talking about the work rather than doing it.

    Checked against the pair rather than the translation alone. The English is
    what decides whether "Lưu ý:" opening a string is narration or a faithful
    translation of a ``msgid`` that opens with "Note:", and reading the
    Vietnamese on its own cannot tell those apart.
    """
    for one, entry in corpus.translated():
        found = textguard.find(entry.msgstr, entry.msgid)
        if found is not None:
            yield Finding(
                check="L03",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=str(found),
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("L04", Group.LANGUAGE, hard=True, title="no English sentence copied through")
def l04_no_prose_in_passthrough(corpus: Corpus) -> Iterator[Finding]:
    """Nothing recorded as a passthrough is prose by today's rules.

    This is the classifier's false positives, and they are the expensive kind of
    mistake. A false negative costs one wasted call; a false positive silently
    leaves an English sentence sitting in a Vietnamese page, and nothing else in
    the pipeline will ever look at it again.

    The comparison is between what the corpus records and what the classifier
    says now, and it has to be, because anything else is a tautology. An earlier
    version of this check re-derived the verdict from the ``msgid`` and looked
    for English function words in whatever came back a no-op. A no-op is
    *defined* as a string with no run of two letters left once the markup is
    gone, so a function word cannot survive in one, and the check was incapable
    of a finding: zero over all 77 839 distinct strings in the corpus, by
    construction rather than by the corpus being clean.

    What the corpus records is the ``passthrough=`` field ``apply`` writes, and
    the classifier is the part of this tool that has been rewritten most often.
    ``is_literal_block`` alone has three rules in it that the first real tier 1
    run added. So the drift this looks for is not hypothetical: every one of
    those rewrites moved strings across the line, and the entries written before
    it keep the old verdict in a comment until somebody compares them.

    Only ``prose`` is reported. A string that was a no-op and is now a literal
    block is still not being translated and still reads correctly; a string that
    is now prose is one the corpus is showing a Vietnamese reader in English.
    """
    for one, entry in corpus.translated():
        recorded = _recorded_kind(entry.comments)
        if recorded is None:
            continue
        now = classify.classify(entry.msgid)
        if now is not classify.Kind.PROSE:
            continue
        yield Finding(
            check="L04",
            path=corpus.relative(one.path),
            line=entry.line,
            detail=f"copied through as {recorded}, but reads as prose now",
            english=entry.msgid,
            got=entry.msgstr,
            segment=entry.id,
        )


@check("L05", Group.LANGUAGE, hard=False, title="second-person pronoun policy")
def l05_pronouns(corpus: Corpus) -> Iterator[Finding]:
    """No ``anh``, ``chị`` or ``em``, and ``bạn`` only where the English addresses the reader.

    ``GLOSSARY.md`` decides this and it is the single style rule that touches
    the most entries. Vietnamese second-person pronouns encode age and social
    relationship, so choosing one is a claim about who is reading, and technical
    documentation has no standing to make it. ``bạn`` is the neutral option and
    the rule is to drop it wherever the grammar allows, keeping it for direct
    calls to action.

    "Direct call to action" is read here as the English using a second-person
    pronoun or opening with an instruction. That is a rough proxy for a rule a
    person applies by ear, which is most of why this check is soft.
    """
    for one, entry in corpus.translated():
        words = _pronouns_in(entry.msgstr)
        forbidden = sorted(words & set(FORBIDDEN_PRONOUNS))
        for pronoun in forbidden:
            yield Finding(
                check="L05",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"{pronoun!r} assumes a relationship with the reader",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )
        if READER in words and not _addresses_the_reader(entry.msgid):
            yield Finding(
                check="L05",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"{READER!r} where the English does not address the reader",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("L06", Group.LANGUAGE, hard=False, title="headings are noun phrases")
def l06_headings(corpus: Corpus) -> Iterator[Finding]:
    """A heading is a noun phrase in sentence case, not an instruction.

    ``GLOSSARY.md``'s other structural rule. English section titles are often
    verb phrases, "Installing packages", and translating one literally produces
    a Vietnamese heading that tells the reader to do something rather than
    naming what they are reading.

    What counts as a heading is a guess, because a ``.po`` file does not say.
    Short, single-line, no closing punctuation and capitalised is the shape, and
    the guess being imperfect is the second reason this check is soft.
    """
    for one, entry in corpus.translated():
        if not _looks_like_a_heading(entry.msgid):
            continue
        opening = _VIETNAMESE_WORD.search(unicodedata.normalize("NFC", entry.msgstr).casefold())
        if opening is not None and opening.group() == HORTATIVE:
            yield Finding(
                check="L06",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"heading opens with {HORTATIVE!r}, which makes it an instruction",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )
        if len(_title_cased(entry.msgid, entry.msgstr)) >= TITLE_CASE_WORDS:
            yield Finding(
                check="L06",
                path=corpus.relative(one.path),
                line=entry.line,
                detail="heading is in title case, and Vietnamese headings are sentence case",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("L07", Group.LANGUAGE, hard=False, title="no cut-down model wrote an entry")
def l07_no_cut_down_model(corpus: Corpus) -> Iterator[Finding]:
    """No committed translation came from a smaller variant of a model.

    Recorded rather than prevented, which is the whole reason the ``model``
    field is in the provenance. A route can be reconfigured, a proxy can fall
    back under load, and an answer that arrived from something cheaper than the
    run intended is worth redoing. Without this the only evidence would be a
    reviewer noticing that a few hundred entries read worse than the rest.
    """
    if corpus.memory is None:
        return
    for one, entry in corpus.translated():
        known = corpus.memory.lookup(entry.msgid, entry.msgctxt)
        if known is None or known.source != "machine" or not known.model:
            continue
        name = known.model.casefold()
        if any(part in name for part in CUT_DOWN):
            yield Finding(
                check="L07",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"written by {known.model}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


@check("L08", Group.LANGUAGE, hard=False, title="sentence counts are close")
def l08_sentence_parity(corpus: Corpus) -> Iterator[Finding]:
    """The translation has within one sentence of the English's count.

    The cheapest available check for a dropped or invented clause, which is the
    failure that no structural rule can see: an answer that loses the second
    half of a paragraph passes every placeholder rule, reads fluently, and is
    missing something the reader needed.

    Soft because Vietnamese genuinely splits and joins sentences, and because
    an abbreviation with a full stop in it is counted as a sentence end here.
    """
    for one, entry in corpus.translated():
        if classify.classify(entry.msgid) in PASSTHROUGH:
            continue
        wanted = len(_SENTENCE.findall(strip_markup(entry.msgid)))
        got = len(_SENTENCE.findall(strip_markup(entry.msgstr)))
        if abs(wanted - got) > SENTENCE_SLACK:
            yield Finding(
                check="L08",
                path=corpus.relative(one.path),
                line=entry.line,
                detail=f"the English has {wanted} sentences, the translation has {got}",
                english=entry.msgid,
                got=entry.msgstr,
                segment=entry.id,
            )


def _title_cased(msgid: str, msgstr: str) -> list[str]:
    """Capitalised words in a heading that the English did not capitalise.

    Past the first word, because every sentence starts with one. Excluding the
    words the English already has, because those are the terms the glossary says
    to keep in English and they arrive capitalised because that is how they are
    spelled: "Tuples và Sequences" translating "Tuples and Sequences" is the
    glossary being obeyed, not a heading in title case.

    That exclusion is the whole of this function, and the first real run is why
    it is here. The rule used to be an ASCII regex, which found "Tuples",
    "Sequences" and "Internet", all correct, and missed "Danh Sách Rút Gọn"
    entirely, because ``á`` is not in ``[a-z]``. It reported the opposite of
    what it was for.
    """
    borrowed = {word.casefold() for word in _VIETNAMESE_WORD.findall(msgid)}
    words = _VIETNAMESE_WORD.findall(msgstr)
    return [word for word in words[1:] if word[:1].isupper() and word.casefold() not in borrowed]


def _recorded_kind(comments: Sequence[str]) -> str | None:
    """What the provenance comment says this entry was copied through as.

    ``None`` when there is no such comment, which covers every entry a person
    wrote, every entry a model wrote, and every entry still carrying only
    upstream's comments. Only a passthrough records a kind, so only a
    passthrough can have drifted away from one.
    """
    for comment in comments:
        if not comment.startswith(apply.MARKER) or apply.PASSTHROUGH_FIELD not in comment:
            continue
        said = comment.split(apply.PASSTHROUGH_FIELD, 1)[1].split()
        if said:
            return said[0]
    return None


def _pronouns_in(text: str) -> set[str]:
    """The words of a translation, lowercased, tone marks intact, compounds out.

    Intact is the whole point, and it took a run over the real corpus to see it.
    An earlier version stripped the marks first, on the reasoning that ``bạn``
    typed without them is still ``bạn``. It is, but ``bản`` is not: strip the
    marks and "bản dịch", a translation, becomes the pronoun. So does ``ảnh``,
    an image, for ``anh``. Of ninety-three findings that version produced,
    eighty-six were words with nothing to do with the reader, and a check that
    is wrong nine times in ten is a check that gets muted rather than read.

    Tone marks are letters here. Two words that differ only in one are two
    different words, and folding them together is the mistake that produced
    those eighty-six.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    for compound in COMPOUNDS:
        folded = folded.replace(compound, " ")
    return set(_VIETNAMESE_WORD.findall(folded))


def _addresses_the_reader(msgid: str) -> bool:
    """Whether the English is talking to whoever is reading it.

    Two ways to qualify, because English has two. The pronoun is the obvious
    one. The imperative is the one the first real run turned up: "See the
    ``ssl`` module for a list" is addressed to the reader as squarely as "you
    can see", it carries no pronoun at all, and reading it as impersonal made
    ``L05`` report the ``bạn`` in ninety-six correct translations.

    Only the opening word is looked at, and only against
    :data:`READER_IMPERATIVES`. A general test for the imperative mood needs a
    parser, and the openers below are most of what documentation actually uses.
    """
    words = _WORD.findall(msgid.casefold())
    if {"you", "your", "yours", "yourself"} & set(words):
        return True
    return bool(words) and words[0] in READER_IMPERATIVES


def _looks_like_a_heading(msgid: str) -> bool:
    stripped = msgid.strip()
    if not stripped or "\n" in stripped or len(stripped) > HEADING:
        return False
    if not stripped[0].isupper():
        return False
    return stripped[-1] not in ".:;,!?"
