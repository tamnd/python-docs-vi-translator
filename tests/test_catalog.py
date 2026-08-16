from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import catalog
from pydocvi.catalog import Catalog, CatalogError, Entry, escape, segment_id, unescape


def test_parses_the_header_and_the_entries(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    assert len(cat) == 4
    assert cat.header.is_header
    assert cat.header.fuzzy
    assert cat.metadata["Language"] == "vi"
    assert cat.metadata["Project-Id-Version"] == "Python 3.15"


def test_round_trips_a_fixture_byte_for_byte(data_dir: Path) -> None:
    original = (data_dir / "small.po").read_text(encoding="utf-8")
    assert catalog.render(catalog.parse(original)) == original


def test_translated_and_untranslated_entries(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    assert cat.entries[0].msgstr == "Xử lý lỗi"
    assert cat.entries[0].translated
    assert not cat.entries[1].translated


def test_embedded_newlines_survive(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    assert cat.entries[3].msgid == "A line\nwith an embedded newline."


def test_wrapped_msgid_is_joined_without_a_separator(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    assert cat.entries[2].msgid.startswith("A string long enough that it has to be wrapped")
    assert "wrapped across more than one physical line" in cat.entries[2].msgid


class TestSegmentId:
    def test_is_stable(self) -> None:
        assert segment_id("Dealing with Bugs") == segment_id("Dealing with Bugs")

    def test_context_changes_it(self) -> None:
        assert segment_id("list") != segment_id("list", "the built-in type")

    def test_is_sixteen_hex_characters(self) -> None:
        value = segment_id("anything")
        assert len(value) == 16
        assert set(value) <= set("0123456789abcdef")

    def test_no_collision_between_context_and_id_boundaries(self) -> None:
        """``("ab", "c")`` and ``("a", "bc")`` must not hash the same.

        They would if the two parts were concatenated without a separator, which
        is a real bug in more than one gettext tool.
        """
        assert segment_id("c", "ab") != segment_id("bc", "a")


class TestEscaping:
    @pytest.mark.parametrize(
        "value",
        ["plain", 'quotes "here"', "back" + chr(92) + "slash", "new\nline", "tab\there", ""],
    )
    def test_round_trips(self, value: str) -> None:
        assert unescape(escape(value)) == value

    @given(st.text())
    @settings(max_examples=500)
    def test_round_trips_for_arbitrary_text(self, value: str) -> None:
        assert unescape(escape(value)) == value

    @pytest.mark.parametrize("char", ["\r", "\v", "\f", "\a", "\b"])
    def test_control_characters_are_escaped_rather_than_written_raw(self, char: str) -> None:
        """Left raw, ``textwrap`` turns each of these into a space and loses it."""
        assert char not in escape(f"before{char}after")


class TestRenderField:
    def test_short_values_stay_on_one_line(self) -> None:
        assert catalog.render_field("msgid", "short") == ['msgid "short"']

    def test_long_values_are_wrapped_under_a_bare_keyword(self) -> None:
        lines = catalog.render_field("msgid", "word " * 40)
        assert lines[0] == 'msgid ""'
        assert all(line.startswith('"') for line in lines[1:])

    def test_no_rendered_line_exceeds_the_width_without_cause(self) -> None:
        lines = catalog.render_field("msgstr", "a sentence with ordinary words " * 10)
        assert max(len(line) for line in lines) <= catalog.LINE_WIDTH

    def test_an_embedded_newline_ends_its_line(self) -> None:
        lines = catalog.render_field("msgid", "first\nsecond\nthird")
        assert lines[1].endswith(chr(92) + 'n"')

    @given(st.text(min_size=1))
    @settings(max_examples=300)
    def test_rendering_then_parsing_recovers_the_value(self, value: str) -> None:
        text = "\n".join(
            [
                'msgid ""',
                'msgstr ""',
                '"Language: vi' + chr(92) + 'n"',
                "",
                *catalog.render_field("msgid", value),
                'msgstr ""',
                "",
            ]
        )
        assert catalog.parse(text).entries[0].msgid == value


class TestEntry:
    def test_with_msgstr_marks_fuzzy_by_default(self) -> None:
        entry = Entry(msgid="Hello").with_msgstr("Xin chào")
        assert entry.msgstr == "Xin chào"
        assert entry.fuzzy

    def test_with_msgstr_drops_the_original_lines(self) -> None:
        """A changed value must not be written back from the lines it came from."""
        entry = Entry(msgid="Hello", raw=('msgid "Hello"', 'msgstr ""'))
        assert entry.with_msgstr("Xin chào").raw is None

    def test_fuzzy_is_not_duplicated_on_a_second_write(self) -> None:
        entry = Entry(msgid="Hello").with_msgstr("a").with_msgstr("b")
        assert entry.flags.count("fuzzy") == 1

    def test_unfuzzy_removes_the_flag(self) -> None:
        entry = Entry(msgid="Hello").with_msgstr("a").with_msgstr("a", fuzzy=False)
        assert not entry.fuzzy


class TestErrors:
    def test_a_continuation_with_no_keyword_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="continuation"):
            catalog.parse('msgid ""\nmsgstr ""\n\n"orphan"\n')

    def test_an_unsupported_keyword_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="unsupported keyword"):
            catalog.parse('msgid ""\nmsgstr ""\n\nmsgstuff "x"\n')

    def test_an_empty_file_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="no entries"):
            catalog.parse("")

    def test_a_file_that_does_not_start_with_a_header_is_rejected(self) -> None:
        with pytest.raises(CatalogError, match="not a header"):
            catalog.parse('msgid "Hello"\nmsgstr ""\n')


class TestWrite:
    def test_writes_when_the_bytes_changed(self, tmp_path: Path, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        target = tmp_path / "out.po"
        assert catalog.write(cat, target) is True
        assert target.read_text(encoding="utf-8") == (data_dir / "small.po").read_text(
            encoding="utf-8"
        )

    def test_does_not_write_when_the_bytes_are_the_same(
        self, tmp_path: Path, data_dir: Path
    ) -> None:
        cat = catalog.read(data_dir / "small.po")
        target = tmp_path / "out.po"
        catalog.write(cat, target)
        before = target.stat().st_mtime_ns
        assert catalog.write(cat, target) is False
        assert target.stat().st_mtime_ns == before

    def test_leaves_no_temporary_file_behind(self, tmp_path: Path, data_dir: Path) -> None:
        catalog.write(catalog.read(data_dir / "small.po"), tmp_path / "out.po")
        assert [p.name for p in tmp_path.iterdir()] == ["out.po"]


class TestCatalogHelpers:
    def test_with_metadata_replaces_in_place_and_keeps_order(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po").with_metadata(
            **{"Last-Translator": "Someone Else"}
        )
        assert cat.metadata["Last-Translator"] == "Someone Else"
        assert next(iter(cat.metadata)) == "Project-Id-Version"

    def test_with_metadata_appends_an_unknown_key(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po").with_metadata(**{"X-Generator": "pydocvi"})
        assert cat.metadata["X-Generator"] == "pydocvi"

    def test_by_id_keys_on_the_segment_id(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        assert cat.by_id()[segment_id("Dealing with Bugs")].msgstr == "Xử lý lỗi"

    def test_walk_skips_build_and_legacy_trees(self, tmp_path: Path) -> None:
        for name in ["a.po", "locales/b.po", "MACHINE/c.po", "legacy/d.po", "sub/e.po"]:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        found = [p.relative_to(tmp_path).as_posix() for p in catalog.walk(tmp_path)]
        assert found == ["a.po", "sub/e.po"]


class TestChangedEntriesAreRewritten:
    def test_an_untouched_entry_keeps_its_original_lines(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        assert cat.entries[2].raw is not None
        assert catalog.render(cat) == (data_dir / "small.po").read_text(encoding="utf-8")

    def test_a_changed_entry_is_re_rendered_and_still_parses(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        updated = cat.entries[1].with_msgstr("Một chuỗi chưa được dịch với :func:`role` trong đó.")
        rewritten = Catalog(
            path=cat.path,
            header=cat.header,
            entries=(*cat.entries[:1], updated, *cat.entries[2:]),
        )
        text = catalog.render(rewritten)
        reparsed = catalog.parse(text)
        assert reparsed.entries[1].msgstr == updated.msgstr
        assert reparsed.entries[1].fuzzy
        assert reparsed.entries[0].raw == cat.entries[0].raw
