"""Turning a batch into the two messages that go over the wire.

The prompt is a file in the package, not a string literal in this module. That
is what makes ``prompt show`` able to print exactly what was sent, what makes
the hash in the provenance comment mean something a person can look up, and what
lets a prompt change be reviewed as a diff rather than found in a code review of
something else.

Nothing here calls a model or touches the network. Rendering is a pure function
of the batch, the glossary and the shipped file, so the same batch always
produces the same bytes and a prompt change is visible as a hash change.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

from pydocvi.batch import Batch
from pydocvi.glossary import Glossary, Term

#: The shipped prompt, and for now the only one.
TRANSLATE = "translate_vi.md"

#: Where the batch's terminology is spliced into the shipped file. A literal
#: brace pair rather than ``str.format``, because the prompt is full of braces
#: that are examples of Python format specifiers and must survive verbatim.
SLOT = "{glossary}"

#: What goes in that slot when no row in the glossary matches anything in the
#: batch. Said rather than left blank: an empty heading reads like a truncated
#: prompt, and the model treats a section that is there and empty as one it can
#: fill in.
NO_TERMS = "No glossary term appears in these strings."

#: How a ``keep_en`` row is written into the prompt. The glossary's decision is
#: that the English word is the Vietnamese term, so the instruction has to say
#: what to do rather than show a rendering that looks like a no-op.
KEEP = "leave in English"

#: The first line of the user message names the file and what kind of writing it
#: is, because the register is not a detail. The tutorial is addressed to a
#: beginner and the C API reference is addressed to somebody writing an
#: extension, and a translation that gets that wrong reads as machine output
#: even when every word in it is right.
AREAS = {
    "c-api": "the C API reference, written for people writing extension modules",
    "deprecations": "the list of deprecated features",
    "distributing": "the guide to distributing Python modules",
    "extending": "the guide to extending Python with C",
    "faq": "the frequently asked questions",
    "howto": "a how-to guide",
    "installing": "the guide to installing Python modules",
    "library": "the standard library reference",
    "reference": "the language reference, which is precise and formal",
    "tutorial": "the tutorial, written for somebody new to Python",
    "using": "the guide to using Python on a particular platform",
    "whatsnew": "the release notes for one version",
}

#: For the files that sit at the top of the tree rather than in one of those.
ROOT = "the top level of the Python documentation"


@dataclass(frozen=True, slots=True, kw_only=True)
class Prompt:
    """What one batch turned into, and what it was built from.

    ``terms`` is kept beside the text because the audit asks later whether the
    renderings the prompt asked for arrived, and it should ask about the rows
    this batch was actually shown rather than about the whole glossary.
    """

    system: str
    user: str
    terms: tuple[Term, ...] = ()
    fingerprint: str = ""

    @property
    def characters(self) -> int:
        return len(self.system) + len(self.user)


def template(name: str = TRANSLATE) -> str:
    """The prompt file as shipped, unrendered.

    Read from the package rather than from a path on this machine, so that a
    wheel installed anywhere prints and hashes the same prompt as a checkout.
    """
    return resources.files("pydocvi").joinpath("prompts", name).read_text(encoding="utf-8")


def fingerprint(name: str = TRANSLATE) -> str:
    """The prompt's identity, as it goes into the provenance comment.

    Of the file rather than of the rendered prompt. A rendered prompt differs
    per batch because the terminology section does, and a hash that changed with
    every batch would answer no question anybody asks. The question this answers
    is "which version of the prompt produced this translation", and the file is
    that version.
    """
    return hashlib.sha256(template(name).encode("utf-8")).hexdigest()


def area(path: str) -> str:
    """What kind of documentation a corpus path holds."""
    head, _, rest = path.partition("/")
    return AREAS.get(head, ROOT) if rest else ROOT


def context_line(path: str) -> str:
    return f"These strings are from {path}, part of {area(path)}."


def terms_for(batch: Batch, glossary: Glossary) -> tuple[Term, ...]:
    """The rows in force for this batch, in match order and without repeats.

    Selected against the protected text rather than the raw ``msgid``, so a term
    that appears only inside a ``:func:`` role does not put a row in the prompt.
    The model never sees what is inside a marker, and a row it cannot act on is
    a row that dilutes the ones it can.
    """
    matcher = glossary.matcher()
    found: dict[str, Term] = {}
    for item in batch.items:
        for term in matcher.select(item.protected.text, where=batch.path):
            found.setdefault(term.en, term)
    return tuple(found.values())


def terminology(terms: Sequence[Term]) -> str:
    """The terminology section, one row per line."""
    if not terms:
        return NO_TERMS
    return "\n".join(
        f"- {term.en} -> {KEEP if term.keep_en else term.vi}"
        + (f"  ({term.note})" if term.note else "")
        for term in terms
    )


def fill(text: str, terms: Sequence[Term]) -> str:
    """Put the terminology in a prompt template.

    Separate from :func:`system` so that the one thing that can go wrong with a
    template is checkable without a file on disk.
    """
    if SLOT not in text:
        raise RenderError(f"a prompt with no {SLOT} has nowhere to put the terminology")
    return text.replace(SLOT, terminology(terms))


def system(terms: Sequence[Term] = (), *, name: str = TRANSLATE) -> str:
    """The shipped prompt with this batch's terminology in it."""
    return fill(template(name), terms)


def user(batch: Batch) -> str:
    """The file context line, then the numbered entries.

    Numbered from 1 for every batch, which is what ``P03`` checks the answer
    against. The entry's own segment id is not in the prompt and never will be:
    it is 40 characters of hex per entry that the model has no use for and can
    get wrong.
    """
    lines = [context_line(batch.path), ""]
    lines += [f"{number} {item.protected.text}" for number, item in enumerate(batch.items, 1)]
    return "\n".join(lines)


def render(batch: Batch, glossary: Glossary, *, name: str = TRANSLATE) -> Prompt:
    """Everything one call needs, from one batch."""
    terms = terms_for(batch, glossary)
    return Prompt(
        system=system(terms, name=name),
        user=user(batch),
        terms=terms,
        fingerprint=fingerprint(name),
    )


class RenderError(ValueError):
    """The prompt file is not one this code can render."""
