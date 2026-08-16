"""What not to send to a model.

Roughly one entry in eleven has no translatable prose in it at all: a bare
version number, a lone identifier, a block of code. Sending those to a model is
all downside. It costs a call and a wait, and it invites the one failure that is
genuinely embarrassing in a documentation corpus, which is getting
``:mod:`asyncio``` back as ``:mod:`không đồng bộ```.

Every rule here is a pure function of the string. Nothing looks at the file it
came from, and nothing calls anything.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from pydocvi.segment import strip_markup

#: Two or more ASCII letters in a row. The definition of "there is a word here".
#:
#: Deliberately crude, and deliberately biased. A false negative costs one
#: wasted call. A false positive leaves an English sentence sitting in the
#: corpus wearing a translation's clothes, which nobody will notice until a
#: reader does. ``L04`` in the audit hunts for exactly that by looking for
#: English function words in a passthrough string.
_WORD = re.compile(r"[A-Za-z]{2,}")

#: A version number, a dotted identifier, a bare token: whatever a person would
#: write the same way in any language.
_VERSION = re.compile(r"^\d+(\.\d+)*$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

_DOCTEST = ">>>"
_INDENT = 2


class Kind(StrEnum):
    """What an entry is, and therefore what happens to it.

    ``PROSE`` is the only kind that reaches a model. The rest are copied through
    as ``source: passthrough``, which is a claim the audit can check rather than
    a silence.
    """

    PROSE = "prose"
    NOOP = "noop"
    DOCTEST = "doctest"
    LITERAL_BLOCK = "literal_block"
    VERSION_MARKER = "version_marker"

    @property
    def translatable(self) -> bool:
        return self is Kind.PROSE


@dataclass(frozen=True, slots=True, kw_only=True)
class Counts:
    """How a corpus divides up. Written by ``classify --report``."""

    prose: int = 0
    noop: int = 0
    doctest: int = 0
    literal_block: int = 0
    version_marker: int = 0

    @property
    def total(self) -> int:
        return self.prose + self.noop + self.doctest + self.literal_block + self.version_marker

    @property
    def passthrough(self) -> int:
        return self.total - self.prose


def classify(msgid: str) -> Kind:
    """Decide what an entry is.

    The order is the order of confidence. A doctest is recognisable from its
    first line and nothing else looks like one, so it is asked about first. The
    no-op rule is asked last because it is the loosest, and a string that
    something more specific already explained should be reported as that thing.
    """
    if is_doctest(msgid):
        return Kind.DOCTEST
    if is_literal_block(msgid):
        return Kind.LITERAL_BLOCK
    if is_version_marker(msgid):
        return Kind.VERSION_MARKER
    if is_noop(msgid):
        return Kind.NOOP
    return Kind.PROSE


def is_doctest(msgid: str) -> bool:
    """Whether the entry is an interactive example.

    Two thousand code examples with English comments in them are genuinely worth
    translating and genuinely dangerous. One changed character in
    ``>>> sorted(d.keys())`` is a broken example that a reader will type and
    then have to debug. So the code is untouchable here, and the comments inside
    it are a separate pass, with their own prompt and their own check that every
    code line came back byte-identical. That pass is M8, after everything else
    works.
    """
    for line in msgid.split("\n"):
        if line.strip():
            return line.lstrip().startswith(_DOCTEST)
    return False


def is_literal_block(msgid: str) -> bool:
    """Whether the entry is a block of code rather than a paragraph.

    The rule is: more than one line, and at least one of them indented. The
    design notes said every non-blank line had to be indented, and that rule
    selects exactly zero of the 87 008 entries, because a code block in this
    corpus starts at column 0 and indents from the second line on. A C function
    opens with ``static PyObject *`` and a Python one with ``def f():``, both
    flush left, both followed by an indented body.

    So the rule was measured again rather than kept. 2 232 entries have a
    multi-line body with an indented line in it, and reading a sample of them
    finds no prose: the 27 that contain a sentence-shaped line are docstrings
    and captured log output inside code samples. Sending any of these to a
    model, which is what the notes' rule would have done by falling through to
    prose, is the failure this classifier exists to prevent.
    """
    lines = msgid.split("\n")
    if len(lines) < 2:
        return False
    return any(len(line) - len(line.lstrip()) >= _INDENT for line in lines if line.strip())


def is_version_marker(msgid: str) -> bool:
    """Whether the entry is a version number or a bare identifier."""
    stripped = msgid.strip()
    if not stripped or "\n" in stripped:
        return False
    return bool(_VERSION.match(stripped) or _IDENTIFIER.match(stripped))


def is_noop(msgid: str) -> bool:
    """Whether anything is left to translate once the markup is gone.

    An entry is a no-op when, after removing every protected span, no run of two
    or more ASCII letters remains. ``:mod:`os.path``` is a no-op. ``the
    :mod:`os.path` module`` is not, and the difference is the four English words
    a reader would otherwise meet in the middle of a Vietnamese page.
    """
    return not _WORD.search(strip_markup(msgid))


def counts(msgids: list[str]) -> Counts:
    """Classify a corpus and tally it."""
    tally = dict.fromkeys(Kind, 0)
    for msgid in msgids:
        tally[classify(msgid)] += 1
    return Counts(
        prose=tally[Kind.PROSE],
        noop=tally[Kind.NOOP],
        doctest=tally[Kind.DOCTEST],
        literal_block=tally[Kind.LITERAL_BLOCK],
        version_marker=tally[Kind.VERSION_MARKER],
    )
