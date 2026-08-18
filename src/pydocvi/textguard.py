"""Catching a model that talked instead of translating.

Every phrase here has one property: no Vietnamese translation of a sentence from
the Python documentation contains it *unless the English said the same thing*.
"Here is the translation" is not a sentence about ``dict.setdefault``, and
neither is an apology.

That qualification is the whole design, and it was not here at first. The guard
read the translation alone, so a faithful rendering of a ``msgid`` beginning
``Note:`` came back as ``Lưu ý:`` and was refused as narration. It cost more
than a wrong entry: ``P06`` rejects the whole batch, so one false positive threw
away 28 good translations in the first real run, and 89 of the corpus's 2 776
batches contain a string shaped that way. Each phrase now carries the English it
is allowed to be a translation of, and a phrase whose licence the ``msgid``
matches is left alone.

The lists are short on purpose. A long list of phrases becomes a filter that
occasionally deletes a real translation, and a wrongly refused entry is worse
than a narrated one, because the narrated one is visible in the report and the
refused one is just missing.
"""

import re
from dataclasses import dataclass

#: A phrase the model has no business writing, and the English that would make
#: it a translation rather than an aside.
#:
#: ``licence`` is deliberately looser than the phrase it excuses. It is asked
#: only whether the source raised the subject at all, because the Vietnamese for
#: "note" is "lưu ý" wherever in the sentence the English put it, and a licence
#: that insisted on the same position would fail on every sentence Vietnamese
#: reorders. Being loose costs a narrated entry that happens to sit beside a
#: source mentioning notes; being strict costs correct translations by the
#: hundred, and the first is the cheaper way to be wrong.
_PHRASES: tuple[tuple[str, str], ...] = (
    # English narration. Anchored where the phrase only means narration at the
    # start of a line, loose where it means nothing else anywhere.
    (r"here (?:is|are) the (?:translation|translated|following)", ""),
    (r"(?:i|we) (?:have|'ve) translated", ""),
    (r"(?:i|we) (?:cannot|can't|am unable|are unable)", ""),
    (r"(?:i'm |i am )?sorry", r"\bsorry\b"),
    (r"as an ai\b", ""),
    (r"(?:note|please note) that", r"\bnotes?\b"),
    (r"the (?:above|following) (?:is|are) (?:the |my )?translation", ""),
    (r"let me know if", ""),
    (r"hope this helps", ""),
    (r"translation:", r"\btranslations?\b"),
    # Vietnamese narration. A model that switches language mid-task narrates in
    # the target language as readily as in the source, and this half of the list
    # is the one that would be missing from a guard written by somebody who only
    # tested in English.
    (r"đây là bản dịch", ""),
    (r"dưới đây là", r"\b(?:the following|below|as follows|here(?:'s| is| are))\b"),
    (r"sau đây là bản dịch", ""),
    (r"tôi (?:đã dịch|không thể|xin lỗi)", ""),
    (r"xin lỗi", r"\bsorry\b"),
    (r"bản dịch(?: tiếng việt)?:", r"\btranslations?\b"),
    (r"lưu ý(?: rằng)?:", r"\bnotes?\b"),
    (r"hy vọng", r"\bhopes?\b"),
)

_FLAGS = re.IGNORECASE | re.MULTILINE


@dataclass(frozen=True, slots=True)
class _Guard:
    phrase: re.Pattern[str]
    licence: re.Pattern[str] | None


_GUARDS = tuple(
    _Guard(
        phrase=re.compile(phrase, _FLAGS),
        licence=re.compile(licence, _FLAGS) if licence else None,
    )
    for phrase, licence in _PHRASES
)

#: A fence or a rule opening the answer. The model presenting its work rather
#: than doing it, which is the same failure wearing different clothes.
#:
#: Licensed by the source opening the same way, which it did not used to be. The
#: reasoning for having no licence was that the corpus has literal blocks but a
#: protected one reaches this module as a placeholder and never as three
#: backticks, and that is true of the translate path and not of the audit.
#: ``L03`` hands over the raw ``msgstr`` and the raw ``msgid``, with nothing
#: protected, so an entry whose ``msgid`` is ``---`` arrived here as ``---`` and
#: was read as a model drawing a rule. Two entries in the corpus are exactly
#: that: a literal ``---`` in ``c-api/call.po`` and the inheritance diagram in
#: ``howto/mro.po``, both copied through correctly and both reported as
#: narration by the check that exists to catch a model talking.
_FENCE = re.compile(r"^\s*(?:```|~~~|---)")

#: A parenthetical aside about the translation itself, in either language.
#: No licence either, for the same reason: the CPython documentation does not
#: contain bracketed notes from its own translator.
_ASIDE = re.compile(
    r"[(\[](?:[^)\]]{0,40})(?:translator|dịch giả|người dịch|note to|ghi chú của)[^)\]]*[)\]]",
    _FLAGS,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Narration:
    """A phrase that has no business in a translated string."""

    phrase: str
    where: int

    def __str__(self) -> str:
        return f"{self.phrase!r} at offset {self.where}"


def find(text: str, source: str = "") -> Narration | None:
    """The first sign the model was talking about the work rather than doing it.

    ``source`` is the ``msgid`` the translation came from. A phrase whose licence
    the source matches is a translation of something the English said and is left
    alone. Passing no source means nothing is licensed, which is the right answer
    for a caller that has a string and no idea what it was made from.

    Only the first phrase is reported. One is enough to refuse the entry, and
    naming three phrases from one narrated paragraph makes the report longer
    without making the decision any different.
    """
    fence = _FENCE.match(text)
    if fence and not _FENCE.match(source):
        return Narration(phrase=fence.group(0).strip(), where=0)

    hits = [
        match
        for guard in _GUARDS
        if (match := guard.phrase.search(text)) and not _licensed(guard, source)
    ]
    aside = _ASIDE.search(text)
    if aside:
        hits.append(aside)
    if not hits:
        return None
    first = min(hits, key=lambda match: match.start())
    return Narration(phrase=first.group(0).strip(), where=first.start())


def _licensed(guard: _Guard, source: str) -> bool:
    return bool(source) and guard.licence is not None and bool(guard.licence.search(source))


def clean(text: str, source: str = "") -> bool:
    """Whether a string is free of narration the source did not license."""
    return find(text, source) is None
