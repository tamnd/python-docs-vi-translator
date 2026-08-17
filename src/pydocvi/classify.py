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
_CONTINUED = "..."
_INDENT = 2

#: The shapes of a line of code, for the blocks the indent rule cannot see.
#:
#: Each one is a whole-line anchor and each one had to earn its place against
#: the corpus. A line that matches none of them is prose as far as this module
#: is concerned, which is the safe direction to be wrong in: a missed block
#: costs a call, and a paragraph mistaken for code is an English sentence left
#: sitting in a Vietnamese page.
_SHEBANG = re.compile(r"^#!")
_IMPORT = re.compile(r"^(?:import|from)\s+[\w.]+")
_ASSIGNMENT = re.compile(r"^[\w.\[\]'\"]+(?:\s*,\s*[\w.\[\]'\"]+)*\s*(?:[-+*/|&^%@]|//|\*\*)?=[^=]")
_INVOCATION = re.compile(
    r"^(?:python[\d.]*|pip[\d.]*|py|uv|uvx|source|chmod|make|git|sudo|cd|deactivate)\b"
)
_PATH = re.compile(r"^[\w.~-]+(?:[/\\][\w.~-]+)+$")
_TIGHT = re.compile(r"^(?:[\w.]+\(.*\)|[\[{].*[\]}])$")

#: A terminal prompt, with the virtualenv name some transcripts put in front of
#: it. One of these anywhere in an entry makes the whole entry a transcript,
#: because the lines around a prompt are the output of the command in it.
#:
#: ``>`` is not in the set, though pdb frames start with it. Three debugger
#: transcripts in the corpus are reached by including it, and four entries
#: reading ``> (greater)`` are wrongly reached with them: a table of operators
#: glossed in English, which is prose and wants translating. ``#`` stays,
#: because it is both a root prompt and a comment and no reST construct starts
#: a line with a hash and a space.
_PROMPT = re.compile(r"^(?:\([\w.-]+\)\s*)?[$#] \S")


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

    An example quoted from the middle has no ``>>>`` in it at all, only the
    ``...`` continuations, and ``tutorial/errors.po`` has one that cost three
    model calls in the first tier 1 run before dying. So an entry every line of
    which is a continuation counts too. Every line, and more than one of them,
    because 25 entries open with ``...`` and most are prose picking up a
    sentence from the heading above them: "... install packages just for the
    current user?" is a question, not a frame of a session.
    """
    lines = [line.lstrip() for line in msgid.split("\n") if line.strip()]
    if not lines:
        return False
    if len(lines) > 1 and all(line.startswith(_CONTINUED) for line in lines):
        return True
    return lines[0].startswith(_DOCTEST)


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

    The indent rule is necessary and it is not sufficient, and the first real
    tier 1 run is what proved it. 26 entries came back with nothing in the
    memory after three attempts each, and every one of them was a block of code
    this rule called prose: ``x = MyClass()``, ``import sound.effects.echo``,
    ``source tutorial-env/bin/activate``. One line, or several with no indent
    among them. The model did the right thing and returned them unchanged, and
    then ``P05`` refused the answer for being identical to the source and
    ``P08`` refused it for having no Vietnamese in it. Those two rules were 161
    of the 206 refusals in that run, which is to say most of what the retry
    ladder cost was spent arguing with a model that was already correct.

    So a line that looks like code counts as well, and an entry counts when
    every line of it does. That adds 738 entries over the whole corpus, 0.85 per
    cent, and two random samples of them read as code with no prose in either.
    """
    lines = [line for line in msgid.split("\n") if line.strip()]
    if not lines:
        return False
    if any(_PROMPT.match(line.strip()) for line in lines):
        return True
    if all(_is_code(line.strip()) for line in lines):
        return True
    return len(lines) > 1 and any(len(line) - len(line.lstrip()) >= _INDENT for line in lines)


def _is_code(line: str) -> bool:
    """Whether one stripped line is code rather than a sentence."""
    return bool(
        _SHEBANG.match(line)
        or _IMPORT.match(line)
        or _ASSIGNMENT.match(line)
        or _INVOCATION.match(line)
        or _PATH.match(line)
        or _tight(line)
    )


def _tight(line: str) -> bool:
    """Whether the line is a call or a literal written with no prose spacing.

    ``sorted(d.keys())`` and ``["echo", "surround"]`` are code. ``(Contributed
    by Eddie Elizondo in :issue:`35810`.)`` and ``I/O control`` are sentences
    that happen to start and end with the same characters, and an earlier
    version of this rule sent both of them to passthrough. The thing that
    separates them is spacing: code puts a space after a comma and nowhere
    else, and English puts one between every pair of words.
    """
    if not _TIGHT.match(line):
        return False
    return all(line[i - 1] == "," for i, char in enumerate(line) if char == " " and i)


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
