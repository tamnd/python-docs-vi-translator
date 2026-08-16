"""Reading and writing gettext catalogs.

This module has one job that matters more than the rest of the project put
together: a catalog that goes in unchanged must come out byte for byte
identical. 548 files of reformatting churn would bury every real change under
noise on the first run and keep it buried afterwards.

Getting that right turned out not to be a matter of picking the right library.
See ``render_field`` for what the corpus actually does and why this module keeps
the original physical lines rather than trying to regenerate them.
"""

import hashlib
import re
import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

type SegmentId = str

#: Longest physical line a rewritten field will produce, quotes included.
#: Measured against the corpus: 79 is the hard ceiling upstream never crosses
#: except on tokens with no break point in them at all.
LINE_WIDTH = 79

#: Width handed to ``textwrap``. The two quote characters make up the rest.
WRAP_WIDTH = LINE_WIDTH - 2

#: Prefix for the segment id hash. Bumping it invalidates every id, which is why
#: it carries a version number.
SEGMENT_HASH_PREFIX = b"pydocvi.seg.1"

#: The escapes gettext understands, backslash first so that the others can be
#: introduced without being escaped again on the next pass.
#:
#: The corpus only ever uses three of these. The rest are here because a
#: carriage return, a vertical tab or a form feed left unescaped goes into the
#: file as itself, and ``textwrap`` turns every one of them into a space on the
#: way out. That is a silent corruption of a translation, found by the property
#: test rather than by anything in the corpus.
_ESCAPES = (
    (chr(92), chr(92) * 2),
    ('"', chr(92) + '"'),
    ("\n", chr(92) + "n"),
    ("\t", chr(92) + "t"),
    ("\r", chr(92) + "r"),
    ("\v", chr(92) + "v"),
    ("\f", chr(92) + "f"),
    ("\a", chr(92) + "a"),
    ("\b", chr(92) + "b"),
)
_UNESCAPES = {escaped[1]: plain for plain, escaped in _ESCAPES}
_FLAG_LINE = re.compile(r"^#,\s*(.*)$")


def escape(value: str) -> str:
    """Escape a Python string into its .po representation."""
    out = value
    for plain, escaped in _ESCAPES:
        out = out.replace(plain, escaped)
    return out


def unescape(value: str) -> str:
    """Inverse of :func:`escape`, applied to the contents of one quoted line."""
    out: list[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char == chr(92) and i + 1 < len(value):
            nxt = value[i + 1]
            out.append(_UNESCAPES.get(nxt, nxt))
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def segment_id(msgid: str, msgctxt: str | None = None) -> SegmentId:
    """Stable identity for a translatable string.

    Keyed on ``(msgctxt, msgid)`` rather than on a file path or a line number,
    because a string that moves between files is the same string and should keep
    its translation. This is what lets one memory serve several upstream
    branches.
    """
    payload = b"\0".join(
        [SEGMENT_HASH_PREFIX, (msgctxt or "").encode("utf-8"), msgid.encode("utf-8")]
    )
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    """One translatable string, plus the lines it was read from.

    ``raw`` holds the physical lines exactly as they appeared in the file. It is
    dropped the moment a value changes, which is what makes an untouched entry
    round-trip exactly and a rewritten one get our own formatting.
    """

    msgid: str
    msgstr: str = ""
    msgctxt: str | None = None
    comments: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    raw: tuple[str, ...] | None = None

    @property
    def id(self) -> SegmentId:
        return segment_id(self.msgid, self.msgctxt)

    @property
    def fuzzy(self) -> bool:
        return "fuzzy" in self.flags

    @property
    def translated(self) -> bool:
        return bool(self.msgstr)

    @property
    def is_header(self) -> bool:
        return self.msgid == "" and self.msgctxt is None

    def with_msgstr(self, msgstr: str, *, fuzzy: bool = True) -> Entry:
        """Return a copy carrying a new translation.

        Everything this tool writes is fuzzy by default. Sphinx renders the
        English source for a fuzzy string, so the worst a reader sees is the
        English they would have seen anyway.
        """
        flags = tuple(f for f in self.flags if f != "fuzzy")
        if fuzzy:
            flags = (*flags, "fuzzy")
        return replace(self, msgstr=msgstr, flags=flags, raw=None)

    def with_comments(self, comments: Sequence[str]) -> Entry:
        return replace(self, comments=tuple(comments), raw=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class Catalog:
    """A parsed .po file: a header entry, then the translatable entries."""

    path: Path
    header: Entry
    entries: tuple[Entry, ...]
    trailing_newline: bool = True

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def metadata(self) -> dict[str, str]:
        """The header's ``Key: value`` block, parsed."""
        out: dict[str, str] = {}
        for line in self.header.msgstr.split("\n"):
            if ": " in line:
                key, _, value = line.partition(": ")
                out[key] = value
            elif line.endswith(":"):
                out[line[:-1]] = ""
        return out

    def by_id(self) -> dict[SegmentId, Entry]:
        return {entry.id: entry for entry in self.entries}

    def replace_entries(self, entries: Sequence[Entry]) -> Catalog:
        return replace(self, entries=tuple(entries))

    def with_metadata(self, **updates: str) -> Catalog:
        """Return a copy with header fields replaced, order preserved."""
        lines = []
        seen = set()
        for line in self.header.msgstr.split("\n"):
            key = line.partition(":")[0]
            if key in updates:
                lines.append(f"{key}: {updates[key]}")
                seen.add(key)
            elif line:
                lines.append(line)
        lines.extend(f"{key}: {value}" for key, value in updates.items() if key not in seen)
        header = replace(self.header, msgstr="\n".join(lines) + "\n", raw=None)
        return replace(self, header=header)


def render_field(keyword: str, value: str) -> list[str]:
    """Render ``msgid``/``msgstr``/``msgctxt`` as physical lines.

    Only ever called for a field this tool changed. Untouched fields are written
    back from ``Entry.raw``, and the reason is worth stating because it is not
    the obvious design.

    The corpus is a Transifex export, and no wrapping rule reproduces it
    exactly. Breaks land after spaces, but also mid-token after a hyphen or a
    slash, and the effective width shifts depending on which. The closest single
    rule, ``textwrap`` at width 77 with hyphen breaking, reproduces 86.31 % of
    the 175 112 fields in the corpus and no variation on it does better. Chasing
    the remaining 13.69 % would mean reimplementing another project's serialiser
    from its output, and it would still be a guess.

    Keeping the original lines makes the fidelity question go away rather than
    answering it: an entry nobody touched produces no diff bytes by
    construction, whatever wrote it and whatever it does next year. The cost is
    that a string this tool rewrites may be rewrapped by Transifex on the way
    back in. That is one entry's worth of noise on a string that changed anyway,
    which is the cheap direction to be wrong in.
    """
    escaped = escape(value)
    single = f'{keyword} "{escaped}"'
    if len(single) <= LINE_WIDTH and "\n" not in value[:-1]:
        return [single]

    lines = [f'{keyword} ""']
    for chunk in _split_after_newlines(escaped):
        wrapped = textwrap.wrap(
            chunk,
            width=WRAP_WIDTH,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=True,
        )
        lines.extend(f'"{piece}"' for piece in (wrapped or [chunk]))
    return lines


def _split_after_newlines(escaped: str) -> list[str]:
    r"""Split so that every escaped ``\n`` ends its chunk.

    A newline in the middle of a string is a paragraph break in the rendered
    documentation, and gettext convention is to start a fresh physical line
    after one. Splitting here rather than inside the wrapper keeps that true
    regardless of what the wrapper does with the rest.
    """
    chunks: list[str] = []
    current = ""
    i = 0
    while i < len(escaped):
        char = escaped[i]
        if char == chr(92) and i + 1 < len(escaped):
            current += escaped[i : i + 2]
            if escaped[i + 1] == "n":
                chunks.append(current)
                current = ""
            i += 2
            continue
        current += char
        i += 1
    if current:
        chunks.append(current)
    return chunks


def _render_entry(entry: Entry) -> list[str]:
    if entry.raw is not None:
        return list(entry.raw)
    lines = list(entry.comments)
    if entry.flags:
        lines.append("#, " + ", ".join(entry.flags))
    if entry.msgctxt is not None:
        lines.extend(render_field("msgctxt", entry.msgctxt))
    lines.extend(render_field("msgid", entry.msgid))
    lines.extend(render_field("msgstr", entry.msgstr))
    return lines


class CatalogError(ValueError):
    """A .po file this module cannot read."""


def parse(text: str, *, path: Path | None = None) -> Catalog:
    """Parse the text of a .po file."""
    where = path or Path("<string>")
    trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if trailing_newline:
        lines.pop()

    entries: list[Entry] = []
    block: list[str] = []
    for line in lines:
        if line.strip() == "":
            if block:
                entries.append(_parse_block(block, where))
                block = []
            continue
        block.append(line)
    if block:
        entries.append(_parse_block(block, where))

    if not entries:
        raise CatalogError(f"{where}: no entries")
    header, *rest = entries
    if not header.is_header:
        raise CatalogError(f"{where}: first entry is not a header")
    return Catalog(
        path=where, header=header, entries=tuple(rest), trailing_newline=trailing_newline
    )


def _parse_block(block: Sequence[str], where: Path) -> Entry:
    comments: list[str] = []
    flags: tuple[str, ...] = ()
    fields: dict[str, str] = {}
    current: str | None = None

    for line in block:
        if line.startswith("#"):
            if current is not None:
                raise CatalogError(f"{where}: comment after a value in {block[0]!r}")
            match = _FLAG_LINE.match(line)
            if match:
                flags = tuple(f.strip() for f in match.group(1).split(",") if f.strip())
            else:
                comments.append(line)
            continue
        if line.startswith('"'):
            if current is None:
                raise CatalogError(f"{where}: continuation line with no keyword: {line!r}")
            fields[current] += unescape(line[1:-1])
            continue
        keyword, _, rest = line.partition(" ")
        if keyword not in {"msgid", "msgstr", "msgctxt"}:
            raise CatalogError(f"{where}: unsupported keyword {keyword!r}")
        current = keyword
        fields[keyword] = unescape(rest.strip()[1:-1])

    if "msgid" not in fields:
        raise CatalogError(f"{where}: block with no msgid: {block[0]!r}")
    return Entry(
        msgid=fields["msgid"],
        msgstr=fields.get("msgstr", ""),
        msgctxt=fields.get("msgctxt"),
        comments=tuple(comments),
        flags=flags,
        raw=tuple(block),
    )


def read(path: Path) -> Catalog:
    """Read a catalog from disk."""
    return parse(path.read_text(encoding="utf-8"), path=path)


def render(catalog: Catalog) -> str:
    """Render a catalog back to text."""
    blocks = [_render_entry(catalog.header)]
    blocks.extend(_render_entry(entry) for entry in catalog.entries)
    text = "\n\n".join("\n".join(block) for block in blocks)
    return text + "\n" if catalog.trailing_newline else text


def write(catalog: Catalog, path: Path | None = None) -> bool:
    """Write a catalog, atomically, and only if its bytes changed.

    Returns whether anything was written. Skipping unchanged files is what keeps
    the mtime of 548 files from moving on every run, which in turn is what makes
    ``git status`` after a run mean something.
    """
    target = path or catalog.path
    payload = render(catalog).encode("utf-8")
    if target.exists() and target.read_bytes() == payload:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pydocvi-tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return True


def walk(root: Path) -> list[Path]:
    """Every catalog under ``root``, in a stable order.

    ``locales/`` is a build artefact and ``MACHINE/`` is the output of the
    pipeline this project replaces. Neither is part of the corpus.
    """
    skip = {".git", "locales", "MACHINE", "legacy", ".venv"}
    return sorted(
        path for path in root.rglob("*.po") if not skip.intersection(path.relative_to(root).parts)
    )
