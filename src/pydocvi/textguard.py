"""Catching a model that talked instead of translating.

Every phrase here has one property: no Vietnamese translation of a sentence from
the Python documentation contains it. "Here is the translation" is not a
sentence about ``dict.setdefault``, and neither is an apology.

The lists are short on purpose. A long list of phrases becomes a filter that
occasionally deletes a real translation, and a wrongly refused entry is worse
than a narrated one, because the narrated one is visible in the report and the
refused one is just missing.
"""

import re
from dataclasses import dataclass

#: English narration. Anchored at the start of the string or of a line, because
#: "the following" appears legitimately inside documentation prose all the time
#: and only means narration when the answer opens with it.
_ENGLISH = (
    r"here (?:is|are) the (?:translation|translated|following)",
    r"(?:i|we) (?:have|'ve) translated",
    r"(?:i|we) (?:cannot|can't|am unable|are unable)",
    r"(?:i'm |i am )?sorry",
    r"as an ai\b",
    r"(?:note|please note) that",
    r"the (?:above|following) (?:is|are) (?:the |my )?translation",
    r"let me know if",
    r"hope this helps",
    r"translation:",
)

#: Vietnamese narration. A model that switches language mid-task narrates in the
#: target language as readily as in the source, and this half of the list is the
#: one that would be missing from a guard written by somebody who only tested in
#: English.
_VIETNAMESE = (
    r"đây là bản dịch",
    r"dưới đây là",
    r"sau đây là bản dịch",
    r"tôi (?:đã dịch|không thể|xin lỗi)",
    r"xin lỗi",
    r"bản dịch(?: tiếng việt)?:",
    r"lưu ý(?: rằng)?:",
    r"hy vọng",
)

_NARRATION = re.compile(
    "|".join(f"(?:{phrase})" for phrase in (*_ENGLISH, *_VIETNAMESE)),
    re.IGNORECASE | re.MULTILINE,
)

#: A fence or a rule opening the answer. The model presenting its work rather
#: than doing it, which is the same failure wearing different clothes.
_FENCE = re.compile(r"^\s*(?:```|~~~|---)")

#: A parenthetical aside about the translation itself, in either language.
_ASIDE = re.compile(
    r"[(\[](?:[^)\]]{0,40})(?:translator|dịch giả|người dịch|note to|ghi chú của)[^)\]]*[)\]]",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Narration:
    """A phrase that has no business in a translated string."""

    phrase: str
    where: int

    def __str__(self) -> str:
        return f"{self.phrase!r} at offset {self.where}"


def find(text: str) -> Narration | None:
    """The first sign the model was talking about the work rather than doing it.

    Only the first. One is enough to refuse the entry, and reporting three
    phrases from one narrated paragraph makes the report longer without making
    the decision any different.
    """
    fence = _FENCE.match(text)
    if fence:
        return Narration(phrase=fence.group(0).strip(), where=0)
    for pattern in (_NARRATION, _ASIDE):
        match = pattern.search(text)
        if match:
            return Narration(phrase=match.group(0).strip(), where=match.start())
    return None


def clean(text: str) -> bool:
    """Whether a string is free of narration."""
    return find(text) is None
