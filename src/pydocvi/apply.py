"""The memory and the upstream catalogs into .po files.

``apply`` calls no model and reads no clock. Given a memory, a set of upstream
catalogs and a stamp, it produces the exact bytes of every file in the content
repo, which is what makes ``--check`` possible: CI renders the same inputs and
compares, and a hand edit on the content repo fails the build by name instead of
being overwritten on the next run and mentioned in a log.

The content repo is a projection. It can be deleted and rebuilt from the memory,
and nothing in it is a source of truth except the strings a person has reviewed,
which is why those are the one thing this module will not write over.

Structure comes from upstream and translations come from the memory. Upstream
owns which strings exist, in what order, with which source references and
extracted comments; the memory owns what they say in Vietnamese. Keeping those
apart is what lets one memory serve several version branches, since the 85 192
strings 3.14 and 3.15 share are the same segments in a different arrangement.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydocvi import __version__, catalog, classify
from pydocvi.catalog import Catalog, Entry
from pydocvi.memory import Memory, Segment

#: The header's ``X-Generator``, and a constant rather than something each
#: caller passes, because every caller has to say the same thing for any of this
#: to work. ``S08`` renders the corpus and requires byte identity with what is
#: committed, so a checker that names itself in the header it is comparing
#: reports every file as differing on that one line and can never pass. It did:
#: the audit built its stamp with ``generator="pydocvi audit"`` and reported all
#: 548 files changed in the same minute ``apply --check`` called them identical.
#: Naming the writer here means a checker cannot disagree with the writer about
#: who wrote the file.
GENERATOR = f"pydocvi {__version__}"

#: Written into every header, in plain words rather than as a version string.
#: It is the first thing anybody who opens one of these files in Poedit sees,
#: and it should tell them what they are looking at.
LAST_TRANSLATOR = "pydocvi (machine translation, unreviewed)"

LANGUAGE_TEAM = "Vietnamese <tamnd@liteio.dev>"
LANGUAGE = "vi"

#: Vietnamese has no plural inflection, so gettext gets one form and no rule.
PLURAL_FORMS = "nplurals=1; plural=0;"

#: The header block ``apply`` owns, in order. Everything else upstream put in
#: the header is left where it was.
CONTENT_TYPE = "text/plain; charset=UTF-8"
TRANSFER_ENCODING = "8bit"

#: How many characters of the prompt hash go in the provenance comment. Eight is
#: enough to name a prompt in a corpus that will hold a handful of them, and
#: short enough that the comment stays on one line.
PROMPT_CHARS = 8

MARKER = "# pydocvi:"

#: The field written on an entry that was copied through rather than translated,
#: and the only durable record of what the classifier thought at the time. Named
#: here rather than spelled at each reader, because the audit and the coverage
#: report both parse it back and a second spelling is a second thing to keep in
#: step with :func:`provenance`.
PASSTHROUGH_FIELD = "passthrough="


class ApplyError(ValueError):
    """The catalogs and the memory disagree in a way writing cannot resolve."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stamp:
    """What every file written by one run says about that run.

    Passed in rather than read from a clock, because a function that reads the
    time is a function whose output cannot be compared with anything. ``--check``
    renders with the stamp taken from the file it is checking, so the timestamp
    never makes a clean tree look dirty.

    The generator defaults to :data:`GENERATOR` and no caller in the package
    overrides it. The tests do, to prove the field reaches the header, and that
    is the whole reason it is still a field.
    """

    project: str
    run: str
    generator: str = GENERATOR
    revision: str = ""

    @property
    def revised(self) -> str:
        """The header's ``PO-Revision-Date``, which defaults to the run."""
        return self.revision or self.run


@dataclass(frozen=True, slots=True, kw_only=True)
class Counts:
    """What happened to the entries of one file, or of a corpus."""

    written: int = 0
    kept: int = 0
    #: Entries copied through from the ``msgid`` because they are not prose.
    #: Held apart from ``written`` because spec 12 §5 is explicit that a no-op is
    #: neither translated nor outstanding, and folding 13 900 of them into the
    #: written column would report a doctest copied verbatim as work done.
    copied: int = 0
    untranslated: int = 0

    @property
    def total(self) -> int:
        return self.written + self.kept + self.copied + self.untranslated

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            written=self.written + other.written,
            kept=self.kept + other.kept,
            copied=self.copied + other.copied,
            untranslated=self.untranslated + other.untranslated,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    """One file's worth of output, and whether it differs from what is there.

    The bytes are held rather than written, so that ``apply``, ``--dry-run`` and
    ``--check`` are the same computation followed by three different endings.
    """

    path: Path
    text: str
    counts: Counts
    changed: bool

    @property
    def payload(self) -> bytes:
        return self.text.encode("utf-8")


@dataclass(frozen=True, slots=True, kw_only=True)
class Result:
    """What a whole run would do."""

    plans: tuple[Plan, ...] = ()

    @property
    def changed(self) -> tuple[Plan, ...]:
        return tuple(plan for plan in self.plans if plan.changed)

    @property
    def counts(self) -> Counts:
        total = Counts()
        for plan in self.plans:
            total += plan.counts
        return total

    @property
    def clean(self) -> bool:
        return not self.changed


def provenance(segment: Segment) -> str:
    """The one comment line a machine-written entry carries.

    Five fields on one line so the diff stays readable, and every one of them is
    load bearing later: ``model`` because a string written by a cut-down model is
    worth doing again, ``prompt`` and ``glossary`` because both are staleness
    causes, ``batch`` because a systematic failure has to be traceable back to
    the call that produced it, and ``run`` because a bad hour has to be findable
    after the fact.

    Every one of them comes from the segment and none from the run doing the
    writing. They describe when and how the string was translated, not when it
    was last copied into a file, and a field that fell back to the applying run
    would both say something untrue and make ``--check`` report the whole corpus
    as changed the day after it was written.

    A field the memory does not have is left out rather than guessed at.
    """
    if segment.source == "passthrough":
        return copied_comment(segment.msgid)
    fields = [
        ("model", segment.model),
        ("prompt", (segment.prompt or "")[:PROMPT_CHARS] or None),
        ("glossary", segment.glossary),
        ("batch", segment.batch),
        ("run", segment.run),
    ]
    said = " ".join(f"{key}={value}" for key, value in fields if value is not None)
    return f"{MARKER} {said}".rstrip()


def copied_comment(msgid: str) -> str:
    """The one comment line an entry that was copied rather than translated gets.

    One field, because the other five have nothing to say: there is no model, no
    prompt, no glossary version and no batch behind a string that was never sent
    anywhere. What is worth recording is what the classifier thought it was, and
    that is the claim ``L04`` goes looking for false positives in.
    """
    return f"{MARKER} {PASSTHROUGH_FIELD}{classify.classify(msgid).value}"


def is_human(entry: Entry) -> bool:
    """Whether a person has signed off on this entry.

    Translated and not fuzzy. Fuzzy is what everything this tool writes carries,
    so the absence of it is the reviewer's mark and the only durable record that
    somebody read the string.
    """
    return entry.translated and not entry.fuzzy


def apply_entry(
    upstream: Entry, existing: Entry | None, segment: Segment | None, *, refuzzy: bool = False
) -> Entry:
    """One entry: the upstream string, with whatever the memory has to say.

    A human ``msgstr`` already in the file is returned untouched, with the lines
    it was read from, so a reviewed string produces no diff bytes at all and
    cannot be silently replaced by a machine one. That is the fixed precedence of
    spec 01 §3 enforced where it matters, at the moment of writing.

    The spec says ``apply`` refuses to run rather than overwrite a person's work.
    Refusing the entry is the same guarantee and a workable one: a reviewer
    unfuzzying a string is the normal case, and refusing the run would leave the
    pipeline failing until the memory caught up with the catalog. What matters is
    that no machine string ever lands on top of a human one, and none does.

    An entry the memory has nothing for is either prose nobody has translated
    yet, which is left as upstream had it, or a no-op, which is copied through
    with the classifier's reason on it. See :func:`_copied`.

    Keeping the reviewed entry whole keeps its source reference as it was, which
    can go stale if upstream moves the string. That is the cheaper way to be
    wrong. A line reference goes stale only when the string moves without
    changing, since changing it produces a different segment entirely, and a
    reviewer who has to look one line further up has lost less than a corpus of
    reflowed diffs costs everybody.

    ``refuzzy`` is the one way to get past that guard, and it is off by default.
    See :func:`_contradicted` for what it does and why it exists.
    """
    if existing is not None and is_human(existing) and not _contradicted(segment, refuzzy):
        return existing
    if segment is None or not segment.msgstr:
        return _copied(upstream) if not _translatable(upstream) else _untranslated(upstream)
    if segment.source == "human":
        return upstream.with_msgstr(segment.msgstr, fuzzy=False)
    return upstream.with_msgstr(segment.msgstr).with_comments(
        [*upstream.comments, provenance(segment)]
    )


def _contradicted(segment: Segment | None, refuzzy: bool) -> bool:
    """Whether an unmarked entry in the corpus is known not to be a reviewer's.

    An unfuzzied entry normally means a reviewer read the string and signed off,
    and :func:`apply_entry` protects it for the whole length of the round trip
    through Transifex and back (spec 08 §4), which is days. There is no way to
    tell that apart from an entry some earlier tool wrote without marking it,
    because both are a ``msgstr`` with no ``fuzzy`` flag and no comment.

    So it is asked for rather than guessed at. This repository inherited 241 of
    the second kind from the 2026 ``scripts/mt.py`` run, and ``apply`` was
    protecting them as somebody's work, which is the one thing the corpus must
    never claim about a machine string. What makes them safe to overwrite is not
    the flag on its own: it is that the memory holds a ``machine`` segment for
    every one of them, so the corpus is contradicted by the record rather than
    merely unexplained.

    An entry the memory records as ``human`` is never touched, flag or no flag.
    That is deliberate and it is what keeps this from being the bulk unfuzzy
    command spec 02 §4 refuses to have: the 966 reviewed strings in tier 1 come
    through a ``--refuzzy`` run byte for byte unchanged.
    """
    return refuzzy and (segment is None or segment.source != "human")


def _translatable(upstream: Entry) -> bool:
    """Whether this entry is the kind a model is ever asked about."""
    return classify.classify(upstream.msgid).translatable


def _copied(upstream: Entry) -> Entry:
    """A no-op entry, copied through with the classifier's reason on it.

    Spec 03 §6 says the non-prose entries are classified deterministically and
    copied through, and nothing in the pipeline was doing it: ``batch`` filters
    them out before anything is queued, no command ever mints a segment for
    them, and the branch in :func:`provenance` that formats their comment could
    not be reached. So 13 900 entries sat as English with no record of why, which
    reads exactly like 13 900 entries nobody has got to yet.

    Minted here from :func:`classify.classify` rather than stored in the memory,
    which is the one deliberate departure from the ``source: passthrough`` the
    spec names. The memory is defined as the things nobody can rebuild, and a
    copy of the ``msgid`` is rebuildable from the ``msgid`` by a pure function.
    Putting 13 900 of them in the file that already trips ``H02`` would be paying
    a megabyte to store an answer that is cheaper to recompute than to read.
    ``apply`` stays a function of the upstream, the memory and the glossary
    either way, so ``--check`` says exactly what it said before.

    Fuzzy, like everything else this tool writes. It costs a copied string
    nothing, since Sphinx renders the English and the English is what the
    ``msgstr`` says, and it keeps ``S04``'s rule to one sentence.
    """
    return upstream.with_msgstr(upstream.msgid).with_comments(
        [*upstream.comments, copied_comment(upstream.msgid)]
    )


def _untranslated(upstream: Entry) -> Entry:
    """Upstream's entry with nothing written onto it.

    Returned as it came when there was nothing there to begin with, because
    ``with_msgstr`` drops the physical lines and the corpus is mostly entries
    nobody has translated yet. Rewriting all of them would reflow tens of
    thousands of fields to say exactly what they already said.

    An upstream entry that does carry a fuzzy translation is blanked. Fuzzy
    upstream means gettext is not confident the string still matches its source,
    ``sync`` refuses to take one into the memory as ground truth, and carrying it
    into this repo would launder it into something that looks like our work.
    """
    if not upstream.msgstr and not upstream.fuzzy:
        return upstream
    return upstream.with_msgstr("", fuzzy=False)


def header(upstream: Catalog, stamp: Stamp) -> Catalog:
    """The header block this tool owns, over the one upstream shipped.

    Identical across the corpus except for the project and the date, because 548
    headers that differ in ways nobody chose is 548 diffs nobody can read.
    """
    return upstream.with_metadata(
        **{
            "Project-Id-Version": stamp.project,
            "PO-Revision-Date": stamp.revised,
            "Last-Translator": LAST_TRANSLATOR,
            "Language-Team": LANGUAGE_TEAM,
            "Language": LANGUAGE,
            "MIME-Version": "1.0",
            "Content-Type": CONTENT_TYPE,
            "Content-Transfer-Encoding": TRANSFER_ENCODING,
            "Plural-Forms": PLURAL_FORMS,
            "X-Generator": stamp.generator,
        }
    )


def apply_catalog(
    upstream: Catalog,
    existing: Catalog | None,
    memory: Memory,
    *,
    stamp: Stamp,
    refuzzy: bool = False,
) -> tuple[Catalog, Counts]:
    """One file: upstream's structure, the memory's translations."""
    known = existing.by_id() if existing is not None else {}
    entries: list[Entry] = []
    written = kept = copied = untranslated = 0

    for entry in upstream:
        was = known.get(entry.id)
        segment = memory.get(entry.id)
        applied = apply_entry(entry, was, segment, refuzzy=refuzzy)
        entries.append(applied)
        if was is not None and is_human(was) and not _contradicted(segment, refuzzy):
            kept += 1
        elif not applied.translated:
            untranslated += 1
        elif segment is None and not _translatable(entry):
            copied += 1
        else:
            written += 1

    counts = Counts(written=written, kept=kept, copied=copied, untranslated=untranslated)
    return header(upstream, stamp).replace_entries(entries), counts


def plan_file(
    upstream: Catalog, target: Path, memory: Memory, *, stamp: Stamp, refuzzy: bool = False
) -> Plan:
    """What one file would become, without writing anything."""
    existing = catalog.read(target) if target.exists() else None
    applied, counts = apply_catalog(upstream, existing, memory, stamp=stamp, refuzzy=refuzzy)
    text = catalog.render(applied)
    changed = not target.exists() or target.read_bytes() != text.encode("utf-8")
    return Plan(path=target, text=text, counts=counts, changed=changed)


def plan(
    sources: Iterable[Path],
    memory: Memory,
    *,
    root: Path,
    into: Path,
    stamp: Stamp,
    refuzzy: bool = False,
) -> Result:
    """What a whole run would do, file by file, in a stable order."""
    return Result(
        plans=tuple(
            plan_file(
                catalog.read(source),
                into / source.relative_to(root),
                memory,
                stamp=stamp,
                refuzzy=refuzzy,
            )
            for source in sources
        )
    )


def write(result: Result) -> tuple[Path, ...]:
    """Write the files that changed, atomically, and return which.

    Only the ones that changed. Leaving 548 mtimes alone is what makes ``git
    status`` after a nine-hour run a report of the run rather than a list of
    every file in the repo.
    """
    written: list[Path] = []
    for one in result.changed:
        one.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = one.path.with_name(f".{one.path.name}.pydocvi-tmp")
        temporary.write_bytes(one.payload)
        temporary.replace(one.path)
        written.append(one.path)
    return tuple(written)


def revision_of(path: Path) -> str:
    """The ``PO-Revision-Date`` already in a file, if it has one.

    ``--check`` needs it. Rendering with today's date would report every file in
    the corpus as changed every day, which is a check that fails so reliably that
    it stops being read.
    """
    if not path.exists():
        return ""
    return catalog.read(path).metadata.get("PO-Revision-Date", "")


def check(
    sources: Sequence[Path],
    memory: Memory,
    *,
    root: Path,
    into: Path,
    stamp: Stamp,
    refuzzy: bool = False,
) -> Result:
    """Render every file against what is committed, and report the differences.

    Each file is rendered with the revision date it already carries, so the only
    thing this can report is a difference in content. A file that is missing gets
    the stamp's own date and shows up as changed, which it is.
    """
    plans: list[Plan] = []
    for source in sources:
        target = into / source.relative_to(root)
        at = revision_of(target)
        one = plan_file(
            catalog.read(source),
            target,
            memory,
            stamp=Stamp(
                project=stamp.project,
                run=stamp.run,
                generator=stamp.generator,
                revision=at or stamp.revision,
            ),
            refuzzy=refuzzy,
        )
        plans.append(one)
    return Result(plans=tuple(plans))
