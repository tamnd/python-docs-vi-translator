"""The nine hard checks, ``P01`` through ``P09``.

Every one of them is a pure function of the English and the Vietnamese. None
calls a model, none looks at a file, and all of them run on every batch.

What they cannot do is worth stating first, because a green run is exactly the
thing that makes people stop reading the output. A Vietnamese sentence that says
the opposite of the English, with every placeholder intact, passes all nine.
These check that nothing was broken, never that anything was translated
correctly. That is what the review pass in M8 is for.

Failure is per entry, not per batch. A batch where entry 17 lost a placeholder
returns 39 accepted entries and one refusal, because at 151 seconds a call,
throwing away 39 good translations over one bad one produces no better corpus
and a much longer run. The exceptions are ``P03``, which is a property of the
whole answer, and ``P06`` and ``P07``, where narration at the top means the
model was not doing the task at all.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydocvi import textguard
from pydocvi.parse import Answer, aligned
from pydocvi.segment import FORMATS, PLACEHOLDER, PLACEHOLDER_LOOSE, RestorationError, restore

#: Evidence that somebody wrote this in Vietnamese: a combining mark, or the
#: letter đ, which has no decomposition and so has to be named on its own.
#:
#: Matched against the decomposed form, because a composed ``ắ`` is one
#: character that no list of base letters contains, and two strings that differ
#: only in normal form differ only in which keyboard typed them.
_DIACRITICS = re.compile(r"[̀-ͯ]|đ", re.IGNORECASE)

#: Below this many characters, a diacritic-free rendering is ordinary rather
#: than suspicious. "API", "Unicode" and "TCP" are the translation.
_SHORT = 24


@dataclass(frozen=True, slots=True, kw_only=True)
class Violation:
    """One broken invariant, against one entry where there is one."""

    rule: str
    detail: str
    index: int | None = None

    @property
    def rejects_batch(self) -> bool:
        """Whether this failure condemns the whole answer rather than an entry."""
        return self.rule in {"P03", "P06", "P07"}

    def __str__(self) -> str:
        where = f"entry {self.index}: " if self.index is not None else ""
        return f"{self.rule}: {where}{self.detail}"


def check_entry(
    msgid: str, msgstr: str, spans: Sequence[str], *, index: int | None = None
) -> list[Violation]:
    """Run every per-entry invariant over one translation.

    All of them, not the first that fails. An entry that lost a placeholder and
    also came back in English is worth reporting twice, because the second fact
    changes what the retry should say.
    """
    checks = (
        _p01(msgid, msgstr, spans),
        _p02(msgid, msgstr, spans),
        _p04(msgid, msgstr),
        _p05(msgid, msgstr),
        _p06(msgstr),
        _p07(msgstr),
        _p08(msgstr),
        _p09(msgid, msgstr),
    )
    return [_at(violation, index) for violation in checks if violation is not None]


def check_answer(answer: Answer, count: int) -> list[Violation]:
    """The invariants that are properties of the whole answer."""
    out: list[Violation] = []
    if not aligned(answer, count):
        got = sorted(answer.entries)
        out.append(
            Violation(rule="P03", detail=f"expected indices 1..{count}, got {got or 'none'}")
        )
    return out


def _at(violation: Violation, index: int | None) -> Violation:
    return violation if index is None else replace(violation, index=index)


def _p01(msgid: str, msgstr: str, spans: Sequence[str]) -> Violation | None:
    """Placeholders: every marker in, exactly once out, no strays.

    Order is not checked and never will be. Vietnamese does not put modifiers
    where English does, so a translation that moves ``⟦2⟧`` in front of ``⟦1⟧``
    is usually right, and a rule that punished it would be a rule against
    translating well.
    """
    del msgid
    found = [int(n) for n in PLACEHOLDER.findall(msgstr)]
    expected = list(range(1, len(spans) + 1))

    missing = [n for n in expected if n not in found]
    if missing:
        return Violation(rule="P01", detail=f"marker(s) missing: {_markers(missing)}")
    repeated = sorted({n for n in found if found.count(n) > 1})
    if repeated:
        return Violation(rule="P01", detail=f"marker(s) repeated: {_markers(repeated)}")
    strays = sorted({n for n in found if n not in expected})
    if strays:
        return Violation(rule="P01", detail=f"marker(s) invented: {_markers(strays)}")
    malformed = [m for m in PLACEHOLDER_LOOSE.findall(msgstr) if not PLACEHOLDER.fullmatch(m)]
    if malformed:
        return Violation(rule="P01", detail=f"malformed marker(s): {' '.join(malformed)}")
    return None


def _p02(msgid: str, msgstr: str, spans: Sequence[str]) -> Violation | None:
    """Restoration: the spans go back byte-identically, or the entry is refused."""
    try:
        restore(msgstr, tuple(spans), original=msgid)
    except RestorationError as error:
        return Violation(rule="P02", detail=str(error))
    return None


def _p04(msgid: str, msgstr: str) -> Violation | None:
    """Edge whitespace and a trailing newline are carried over.

    53 entries in the corpus have significant leading or trailing space and 33
    end in a newline. Losing one is invisible in a diff viewer and shows up as a
    word welded to the next one in the rendered page.
    """
    for edge, get in (("leading", _leading), ("trailing", _trailing)):
        if get(msgid) != get(msgstr):
            return Violation(
                rule="P04",
                detail=f"{edge} whitespace {get(msgid)!r} became {get(msgstr)!r}",
            )
    return None


def _leading(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]


def _trailing(text: str) -> str:
    return text[len(text.rstrip()) :]


def _p05(msgid: str, msgstr: str) -> Violation | None:
    """Non-empty, and not the English handed straight back."""
    if not msgstr.strip():
        return Violation(rule="P05", detail="empty")
    if msgstr.strip() == msgid.strip():
        return Violation(rule="P05", detail="identical to the source")
    return None


def _p06(msgstr: str) -> Violation | None:
    """No narration, in either language."""
    found = textguard.find(msgstr)
    return Violation(rule="P06", detail=str(found)) if found else None


def _p07(msgstr: str) -> Violation | None:
    """No fence and no horizontal rule opening the answer."""
    if re.match(r"^\s*(?:```|~~~|---)", msgstr):
        return Violation(rule="P07", detail="opens with a fence or a rule")
    return None


def _p08(msgstr: str) -> Violation | None:
    """Written in Vietnamese, as far as a machine can tell.

    The weakest rule here and knowingly so. A short string with no diacritics is
    ordinary, because "API" and "Unicode" are what a Vietnamese reader expects
    to see, so only a long diacritic-free string is suspicious. It catches the
    failure it exists for, which is a batch that came back in English.
    """
    stripped = msgstr.strip()
    if len(stripped) <= _SHORT or _DIACRITICS.search(_decompose(stripped)):
        return None
    return Violation(rule="P08", detail=f"{len(stripped)} characters with no Vietnamese diacritic")


def _decompose(text: str) -> str:
    """Split every accented letter into a base letter and its marks.

    A translation typed on one keyboard and the same translation pasted from
    another differ in normal form and not in meaning, so both are decomposed
    before anything looks for a diacritic.
    """
    return unicodedata.normalize("NFD", text)


def _p09(msgid: str, msgstr: str) -> Violation | None:
    """Format specifiers: same set, same count.

    A dropped ``%s`` is a ``TypeError`` at runtime in whatever program copied
    the string, which makes this the one rule here whose failures escape the
    documentation entirely.
    """
    wanted = sorted(FORMATS.findall(msgid))
    got = sorted(FORMATS.findall(msgstr))
    if wanted != got:
        return Violation(rule="P09", detail=f"source has {wanted}, translation has {got}")
    return None


def _markers(numbers: Sequence[int]) -> str:
    return " ".join(f"⟦{n}⟧" for n in numbers)
