from pathlib import Path

import pytest

from pydocvi import apply, catalog
from pydocvi.apply import Stamp
from pydocvi.catalog import Catalog, Entry
from pydocvi.memory import Memory, Segment

STAMP = Stamp(project="Python 3.15", run="2026-08-17T09:30Z", generator="pydocvi 0.4.1")

HEADER = """\
msgid ""
msgstr ""
"Project-Id-Version: Python 3.14\\n"
"PO-Revision-Date: 2025-01-01 00:00+0000\\n"
"Last-Translator: Someone, 2026\\n"
"Language: vi\\n"
"""


def upstream(*blocks: str, header: str = HEADER) -> Catalog:
    return catalog.parse("\n\n".join([header.rstrip("\n"), *blocks]) + "\n")


def block(msgid: str, msgstr: str = "", *, comment: str = "", flags: str = "") -> str:
    lines = [line for line in (comment, flags) if line]
    lines += [f'msgid "{msgid}"', f'msgstr "{msgstr}"']
    return "\n".join(lines)


def machine(msgid: str, msgstr: str, **provenance: object) -> Segment:
    return Segment.from_entry(Entry(msgid=msgid, msgstr=msgstr), source="machine", **provenance)


def entry_of(cat: Catalog, msgid: str) -> Entry:
    return next(entry for entry in cat if entry.msgid == msgid)


def applied(cat: Catalog, memory: Memory) -> Catalog:
    return apply.apply_catalog(cat, None, memory, stamp=STAMP)[0]


class TestWritingATranslation:
    def test_a_machine_translation_lands_fuzzy(self) -> None:
        """Sphinx renders the English for a fuzzy string, so the worst outcome of
        a bad run is a reader seeing English rather than a confident Vietnamese
        sentence that says the opposite."""
        cat = upstream(block("Dealing with Bugs"))
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert entry.msgstr == "Xử lý lỗi"
        assert entry.fuzzy

    def test_it_carries_one_provenance_comment(self) -> None:
        cat = upstream(block("Dealing with Bugs"))
        memory = Memory(
            [
                machine(
                    "Dealing with Bugs",
                    "Xử lý lỗi",
                    model="gpt-5",
                    prompt="a3f10c9e" + "0" * 56,
                    glossary=7,
                    batch="4b2e1f09aa31",
                    run="2026-08-16T04:11Z",
                )
            ]
        )
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert entry.comments[-1] == (
            "# pydocvi: model=gpt-5 prompt=a3f10c9e glossary=7 "
            "batch=4b2e1f09aa31 run=2026-08-16T04:11Z"
        )

    def test_the_comment_says_when_the_string_was_translated_not_when_applied(self) -> None:
        """Applying is not translating. A run field taken from the applying run
        would claim the model wrote the string today, and would make --check
        report the whole corpus as changed the day after it was written."""
        cat = upstream(block("Dealing with Bugs"))
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi", run="2026-08-16T04:11Z")])
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert entry.comments[-1] == "# pydocvi: run=2026-08-16T04:11Z"

    def test_a_field_the_memory_never_recorded_is_left_out(self) -> None:
        cat = upstream(block("Dealing with Bugs"))
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi", model="gpt-5")])
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert entry.comments[-1] == "# pydocvi: model=gpt-5"

    def test_the_comment_keeps_what_upstream_wrote_above_it(self) -> None:
        """The source reference is how a reviewer finds the string in the
        rendered page. Replacing upstream's comments rather than adding to them
        would take that away on every entry this tool touches."""
        cat = upstream(block("Dealing with Bugs", comment="#: ../../bugs.rst:9"))
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert entry.comments[0] == "#: ../../bugs.rst:9"
        assert len(entry.comments) == 2

    def test_a_passthrough_says_what_kind_it_is_and_names_no_model(self) -> None:
        """A copied identifier has no model behind it, so a comment claiming one
        would be a lie in the one place the audit trusts. The kind is the one
        classify gives, which is how a reader of the file can check the claim
        against the same rule the pipeline used."""
        cat = upstream(block("abs"))
        memory = Memory(
            [Segment.from_entry(Entry(msgid="abs", msgstr="abs"), source="passthrough")]
        )
        entry = entry_of(applied(cat, memory), "abs")
        assert entry.comments[-1] == "# pydocvi: passthrough=version_marker"

    def test_a_human_translation_from_upstream_lands_unfuzzy_and_unstamped(self) -> None:
        """It is somebody's work, carried across because the memory is keyed by
        the string. Stamping it with a model would claim this pipeline wrote it."""
        cat = upstream(block("Dealing with Bugs"))
        memory = Memory(
            [
                Segment.from_entry(
                    Entry(msgid="Dealing with Bugs", msgstr="Xử lý lỗi"), source="human"
                )
            ]
        )
        entry = entry_of(applied(cat, memory), "Dealing with Bugs")
        assert not entry.fuzzy
        assert not any(line.startswith("# pydocvi:") for line in entry.comments)

    def test_a_string_the_memory_has_never_seen_stays_empty(self) -> None:
        cat = upstream(block("Dealing with Bugs"))
        assert entry_of(applied(cat, Memory()), "Dealing with Bugs").msgstr == ""


class TestPrecedence:
    def test_a_machine_string_never_lands_on_a_reviewed_one(self) -> None:
        """The whole point of the review pass. A reviewer who unfuzzies a string
        has to be able to trust that the next nine-hour run leaves it alone."""
        source = upstream(block("Dealing with Bugs"))
        existing = upstream(block("Dealing with Bugs", "Xử lý lỗi đã duyệt"))
        memory = Memory([machine("Dealing with Bugs", "Bản dịch máy mới")])
        out, counts = apply.apply_catalog(source, existing, memory, stamp=STAMP)
        assert entry_of(out, "Dealing with Bugs").msgstr == "Xử lý lỗi đã duyệt"
        assert counts.kept == 1
        assert counts.written == 0

    def test_a_reviewed_entry_keeps_the_lines_it_was_read_from(self) -> None:
        """Byte identity, not equality of value. An entry nobody touched has to
        produce no diff bytes at all, or a run buries its own real changes."""
        source = upstream(block("Dealing with Bugs"))
        existing = upstream(block("Dealing with Bugs", "Xử lý lỗi"))
        memory = Memory([machine("Dealing with Bugs", "Khác")])
        out, _ = apply.apply_catalog(source, existing, memory, stamp=STAMP)
        assert entry_of(out, "Dealing with Bugs").raw == entry_of(existing, "Dealing with Bugs").raw

    def test_a_fuzzy_string_in_the_target_is_not_a_reviewed_one(self) -> None:
        """Fuzzy is what this tool writes. Treating it as a person's work would
        freeze the corpus at the first run and make every later one a no-op."""
        source = upstream(block("Dealing with Bugs"))
        existing = upstream(block("Dealing with Bugs", "Bản cũ", flags="#, fuzzy"))
        memory = Memory([machine("Dealing with Bugs", "Bản mới")])
        out, counts = apply.apply_catalog(source, existing, memory, stamp=STAMP)
        assert entry_of(out, "Dealing with Bugs").msgstr == "Bản mới"
        assert counts.written == 1


class TestWhatUpstreamOwns:
    def test_a_string_upstream_dropped_is_dropped(self) -> None:
        """Upstream decides which strings exist. A memory segment for a string
        CPython no longer has is orphaned, and writing it back would resurrect
        documentation that was deleted."""
        source = upstream(block("Dealing with Bugs"))
        existing = upstream(block("Dealing with Bugs"), block("A string upstream removed."))
        memory = Memory([machine("A string upstream removed.", "Đã xoá")])
        out, _ = apply.apply_catalog(source, existing, memory, stamp=STAMP)
        assert [entry.msgid for entry in out] == ["Dealing with Bugs"]

    def test_an_untranslated_entry_keeps_its_own_lines(self) -> None:
        """Most of the corpus is entries nobody has translated yet. Rewriting
        every one of them would reflow tens of thousands of fields to say exactly
        what they already said."""
        source = upstream(block("Dealing with Bugs"))
        out, _ = apply.apply_catalog(source, None, Memory(), stamp=STAMP)
        assert entry_of(out, "Dealing with Bugs").raw is not None

    def test_a_fuzzy_translation_upstream_is_not_carried_across(self) -> None:
        """Fuzzy upstream means gettext is not confident the string still matches
        its source. sync refuses to take one into the memory, and carrying it
        here would launder it into something that looks like our work."""
        source = upstream(block("Dealing with Bugs", "Bản mờ", flags="#, fuzzy"))
        entry = entry_of(applied(source, Memory()), "Dealing with Bugs")
        assert entry.msgstr == ""
        assert not entry.fuzzy


class TestTheHeader:
    def test_it_says_the_translations_are_machine_made_and_unreviewed(self) -> None:
        """In plain words, because it is the first thing anybody opening the file
        in Poedit reads, and a version string means nothing to them."""
        out = applied(upstream(block("Dealing with Bugs")), Memory())
        assert out.metadata["Last-Translator"] == "pydocvi (machine translation, unreviewed)"

    def test_the_project_and_the_date_come_from_the_stamp(self) -> None:
        out = applied(upstream(block("Dealing with Bugs")), Memory())
        assert out.metadata["Project-Id-Version"] == "Python 3.15"
        assert out.metadata["PO-Revision-Date"] == "2026-08-17T09:30Z"

    def test_vietnamese_gets_one_plural_form(self) -> None:
        out = applied(upstream(block("Dealing with Bugs")), Memory())
        assert out.metadata["Plural-Forms"] == "nplurals=1; plural=0;"

    def test_the_generator_names_the_version_that_wrote_it(self) -> None:
        out = applied(upstream(block("Dealing with Bugs")), Memory())
        assert out.metadata["X-Generator"] == "pydocvi 0.4.1"


class TestPlanningAndWriting:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> tuple[Path, Path]:
        root, into = tmp_path / "upstream", tmp_path / "content"
        (root / "tutorial").mkdir(parents=True)
        (root / "tutorial" / "modules.po").write_text(
            catalog.render(upstream(block("Dealing with Bugs"))), encoding="utf-8"
        )
        return root, into

    def test_a_run_writes_only_the_files_that_changed(self, tree: tuple[Path, Path]) -> None:
        """Leaving 548 mtimes alone is what makes git status after a nine-hour
        run a report of the run rather than a list of every file in the repo."""
        root, into = tree
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        first = apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP)
        assert len(apply.write(first)) == 1
        second = apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP)
        assert apply.write(second) == ()
        assert second.clean

    def test_planning_writes_nothing(self, tree: tuple[Path, Path]) -> None:
        """A tool that has run for nine hours must be safe to point at a repo."""
        root, into = tree
        apply.plan(catalog.walk(root), Memory(), root=root, into=into, stamp=STAMP)
        assert not into.exists()

    def test_the_target_keeps_the_layout_upstream_had(self, tree: tuple[Path, Path]) -> None:
        root, into = tree
        result = apply.plan(catalog.walk(root), Memory(), root=root, into=into, stamp=STAMP)
        assert [one.path for one in result.plans] == [into / "tutorial" / "modules.po"]

    def test_the_counts_add_up_across_files(self, tree: tuple[Path, Path]) -> None:
        root, into = tree
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        result = apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP)
        assert result.counts.written == 1
        assert result.counts.total == 1


class TestCheck:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> tuple[Path, Path]:
        root, into = tmp_path / "upstream", tmp_path / "content"
        root.mkdir()
        (root / "bugs.po").write_text(
            catalog.render(upstream(block("Dealing with Bugs"))), encoding="utf-8"
        )
        return root, into

    def test_a_tree_that_matches_the_memory_is_clean(self, tree: tuple[Path, Path]) -> None:
        root, into = tree
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        apply.write(apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP))
        assert apply.check(catalog.walk(root), memory, root=root, into=into, stamp=STAMP).clean

    def test_a_hand_edit_is_named_rather_than_overwritten(self, tree: tuple[Path, Path]) -> None:
        """This is what stops somebody's change to the content repo from being
        silently reverted by the next run. It fails, loudly, naming the file."""
        root, into = tree
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        apply.write(apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP))
        target = into / "bugs.po"
        target.write_text(
            target.read_text(encoding="utf-8").replace("Xử lý lỗi", "Sửa bằng tay"),
            encoding="utf-8",
        )
        result = apply.check(catalog.walk(root), memory, root=root, into=into, stamp=STAMP)
        assert [one.path for one in result.changed] == [target]

    def test_a_second_check_on_a_different_day_is_still_clean(
        self, tree: tuple[Path, Path]
    ) -> None:
        """Rendering with today's date would report every file as changed every
        day, which is a check that fails so reliably it stops being read."""
        root, into = tree
        memory = Memory([machine("Dealing with Bugs", "Xử lý lỗi")])
        apply.write(apply.plan(catalog.walk(root), memory, root=root, into=into, stamp=STAMP))
        later = Stamp(project="Python 3.15", run="2026-12-25T00:00Z", generator="pydocvi 0.4.1")
        assert apply.check(catalog.walk(root), memory, root=root, into=into, stamp=later).clean

    def test_a_file_that_was_never_written_is_reported(self, tree: tuple[Path, Path]) -> None:
        root, into = tree
        result = apply.check(catalog.walk(root), Memory(), root=root, into=into, stamp=STAMP)
        assert not result.clean
