"""Markup protection.

The model never sees reStructuredText. Every construct that must survive a
translation byte for byte is replaced by a placeholder before the prompt is
built and put back afterwards, so the model is asked to do one thing, turn
English prose into Vietnamese prose, rather than two.

That is the whole quality argument. Asking a model to translate a sentence and
simultaneously copy ``:class:`collections.abc.Iterable``` unchanged is asking
for two skills at once, and the second one fails quietly: a translated
cross-reference target is a broken link that renders without an error and reads
as if it works.

Protection is a substitution, not an instruction, so it cannot be disobeyed. The
only thing that can go wrong is the model mangling a placeholder, and that is
caught by counting rather than by trusting.
"""

import re
from dataclasses import dataclass

#: U+27E6 and U+27E7, the mathematical white square brackets.
#:
#: Not ASCII, because the pipeline this replaces used ``zz001zz`` and a model
#: that sees ``zz001zz`` may lowercase it, break a line inside it or decide it
#: is a typo worth fixing. These brackets are not letters in any orthography and
#: no tokeniser treats them as word-internal.
#:
#: Not reST either. ``|sub|``, ``` `x` ``` and ``{n}`` all occur in the corpus,
#: so a scheme that reused any of them would collide with the thing it protects.
OPEN = "⟦"
CLOSE = "⟧"

PLACEHOLDER = re.compile(f"{OPEN}(\\d+){CLOSE}")

#: Anything bracket-shaped, including the malformed ones. ``⟦ 1 ⟧`` with spaces
#: is the failure worth naming: it means the model retyped the placeholder
#: instead of copying it, and a scheme that quietly accepted the spaced form
#: would be accepting a model that edits markers.
PLACEHOLDER_LOOSE = re.compile(f"{OPEN}[^{CLOSE}]*{CLOSE}")

#: The constructs to protect, in the order they are tried.
#:
#: Order is precedence, and one scan does all of them, left to right and
#: non-overlapping. Roles come before inline literals because
#: ``:ref:`a ``b`` ``` shapes exist and the outer span has to win; a regex
#: alternation resolves a tie at the same start position by taking the earlier
#: branch, and an earlier start always wins outright.
#:
#: Both of those matter more than they look. A separate pass per construct gets
#: this wrong twice over: a URL inside a link target is protected on its own and
#: leaves the link tail holding a marker, and a closing role backtick becomes
#: the opening backtick of a link that swallows the prose up to the next role.
#: 200 entries in the corpus hit the second one.
_ROLE = r":[a-zA-Z0-9_+:.-]+:`[^`]*`"
_LITERAL = r"``[^`]+``"
_LINK = r"`(?P<text>[^`<>]+?)(?P<tail>\s*<[^>]*>)?`(?P<anonymous>__?)"
_SUBSTITUTION = r"\|[^|\s][^|]*\|"
_PRINTF = r"%(?:\([^)]*\))?[#0\- +]*\d*(?:\.\d+)?[hlL]?[diouxXeEfFgGcrsa%]"
_BRACE = r"\{[a-zA-Z0-9_.\[\]!:^<>=,+-]*\}"
_URL = r"https?://[^\s<>`\"]+"

#: A hyperlink reference is the one construct where part of the span is prose
#: and part of it is not. ``` `the tutorial <https://…>`_ ``` wants its text
#: translated and its target untouched, inside one span, so it is the only
#: branch that produces two markers rather than one.
_SPAN = re.compile("|".join([_ROLE, _LITERAL, _LINK, _SUBSTITUTION, _PRINTF, _BRACE, _URL]))

#: Format specifiers, kept separately because ``P09`` counts them in the
#: translation and needs the same definition the protector used.
FORMATS = re.compile("|".join([_PRINTF, _BRACE]))


class RestorationError(ValueError):
    """A translation the placeholders cannot be put back into.

    Always a refusal of that entry, never a repair. A placeholder that came back
    doubled, missing or spaced means the model edited a marker, and guessing
    which span it meant is how a wrong cross-reference gets into a catalog with
    nobody having decided to put it there.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class Protected:
    """A string with its markup replaced by placeholders, and the way back."""

    text: str
    spans: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.spans)

    def placeholders(self) -> tuple[str, ...]:
        return tuple(f"{OPEN}{n}{CLOSE}" for n in range(1, len(self.spans) + 1))


def protect(text: str) -> Protected:
    """Replace every protected construct with a numbered placeholder.

    Numbered rather than anonymous because the index is what makes reordering
    legal. Vietnamese does not put modifiers where English does, so a
    translation that moves ``⟦2⟧`` in front of ``⟦1⟧`` is usually right. Order
    is therefore never checked. Presence and count are.
    """
    spans: list[str] = []
    out: list[str] = []
    position = 0

    for match in _SPAN.finditer(text):
        start, end = match.span()
        out.append(text[position:start])
        out.append(_render(match, spans))
        position = end
    out.append(text[position:])
    return Protected(text="".join(out), spans=tuple(spans))


def _render(match: re.Match[str], spans: list[str]) -> str:
    """The replacement for one matched span.

    Every construct but a hyperlink reference becomes one marker. A hyperlink
    reference becomes two, wrapped around its own text, because that text is
    prose the model is meant to translate while the target beside it is a URL
    that must not be touched. Protecting the whole span would leave the link
    text in English; protecting nothing would invite a translated URL, which
    renders as a link that quietly goes nowhere.
    """
    if match.group("text") is None:
        return _placeholder(spans, match.group(0))
    opening = _placeholder(spans, "`")
    tail = f"{match.group('tail') or ''}`{match.group('anonymous')}"
    return f"{opening}{match.group('text')}{_placeholder(spans, tail)}"


def _placeholder(spans: list[str], span: str) -> str:
    """Register a span and return the marker standing in for it.

    A span that occurs twice gets two markers rather than one reused twice.
    Reuse would be smaller, and it would also mean a model that dropped the
    second occurrence produced an answer that still counts as complete.
    """
    spans.append(span)
    return f"{OPEN}{len(spans)}{CLOSE}"


def restore(text: str, spans: tuple[str, ...], *, original: str | None = None) -> str:
    """Put the spans back, or refuse.

    Verified rather than trusted. Every marker must be present exactly once, no
    marker may appear that was never issued, and nothing bracket-shaped may be
    left over afterwards. ``original``, when given, is the ``msgid`` the spans
    came from, and every span is then checked to appear in the result as many
    times as it appeared there.
    """
    found = [int(n) for n in PLACEHOLDER.findall(text)]
    expected = list(range(1, len(spans) + 1))

    missing = [n for n in expected if found.count(n) == 0]
    if missing:
        raise RestorationError(f"marker(s) missing from the translation: {_markers(missing)}")
    doubled = sorted({n for n in found if found.count(n) > 1})
    if doubled:
        raise RestorationError(f"marker(s) repeated in the translation: {_markers(doubled)}")
    strays = sorted({n for n in found if n not in expected})
    if strays:
        raise RestorationError(f"marker(s) the input never had: {_markers(strays)}")

    out = text
    for index, span in enumerate(spans, start=1):
        out = out.replace(f"{OPEN}{index}{CLOSE}", span)

    left = PLACEHOLDER_LOOSE.findall(out)
    if left:
        raise RestorationError(f"malformed marker(s) left in the translation: {' '.join(left)}")
    if original is not None:
        _verify_spans(out, spans, original)
    return out


def _verify_spans(restored: str, spans: tuple[str, ...], original: str) -> None:
    for span in set(spans):
        wanted = original.count(span)
        got = restored.count(span)
        if got != wanted:
            raise RestorationError(f"{span!r} appears {got} time(s), the source has {wanted}")


def _markers(numbers: list[int]) -> str:
    return " ".join(f"{OPEN}{n}{CLOSE}" for n in numbers)


def spans_of(text: str) -> tuple[str, ...]:
    """Every protected span in a string, without building the protected form.

    Used by the audit, which asks whether the spans in a ``msgid`` survived into
    the ``msgstr`` and has no placeholders to work from.
    """
    return protect(text).spans


def strip_markup(text: str) -> str:
    """The prose left over once every protected construct is removed.

    The classifier's view of a string. Whether what remains is worth sending to
    a model is :mod:`pydocvi.classify`'s decision, not this module's.
    """
    protected = protect(text)
    return PLACEHOLDER.sub(" ", protected.text)
