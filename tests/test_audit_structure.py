"""``S01`` to ``S08``: the catalogs against the upstream pin.

All eight are hard, so each of these tests is also a claim that the rule will
not fire on a corpus that is merely unfinished. Most of the corpus is
unfinished, and a structural check that failed on an untranslated entry would
fail on every run there will ever be.
"""

from dataclasses import replace
from pathlib import Path

from conftest import catalog_of, corpus_of, entry, findings, machine_segment, upstream_of
from pydocvi.audit import structure
from pydocvi.catalog import Entry
from pydocvi.memory import Memory, Segment
from pydocvi.sync import Pin

PIN = Pin(
    repo="python/python-docs-vi",
    branch="3.15",
    commit="0" * 40,
    files=1,
    entries=1,
    words=3,
    characters=14,
    translated=1,
)


def written(where: Path, pin: Pin) -> Path:
    path = where / "upstream.yaml"
    path.write_text(pin.as_yaml(), encoding="utf-8")
    return path


class TestS01:
    """The pin is what every other number in the project is quoted against."""

    def test_a_pin_that_matches_a_recount_is_clean(self, tmp_path: Path) -> None:
        source = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        corpus = corpus_of(pin=written(tmp_path, PIN), upstream=upstream_of(source))
        assert findings(structure.s01_counts_match_the_pin, corpus) == []

    def test_a_drifted_count_is_found_and_named(self, tmp_path: Path) -> None:
        source = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        drifted = written(tmp_path, replace(PIN, entries=900))
        corpus = corpus_of(pin=drifted, upstream=upstream_of(source))
        found = findings(structure.s01_counts_match_the_pin, corpus)
        assert len(found) == 1
        assert "900 entries" in found[0].detail

    def test_no_pin_file_at_all_is_a_finding(self, tmp_path: Path) -> None:
        """Absent and correct are different, and only one of them is checkable."""
        corpus = corpus_of(pin=tmp_path / "upstream.yaml")
        assert len(findings(structure.s01_counts_match_the_pin, corpus)) == 1

    def test_a_run_with_no_pin_configured_says_nothing(self) -> None:
        assert findings(structure.s01_counts_match_the_pin, corpus_of()) == []


class TestS02:
    """The one that catches a msgid edit, which is the most damaging thing
    anyone can do to a catalog."""

    def test_an_edited_msgid_stops_existing_upstream_and_is_found(self) -> None:
        source = catalog_of(entry("Return a list.", ""))
        here = catalog_of(entry("Return a list!", "Trả về một danh sách."))
        found = findings(
            structure.s02_every_msgid_exists_upstream, corpus_of(here, upstream=upstream_of(source))
        )
        assert len(found) == 1
        assert "edited here" in found[0].detail

    def test_an_untouched_msgid_is_clean(self) -> None:
        source = catalog_of(entry("Return a list.", ""))
        here = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert findings(structure.s02_every_msgid_exists_upstream, corpus) == []

    def test_a_file_upstream_does_not_have_is_found_once_per_entry(self) -> None:
        here = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        found = findings(structure.s02_every_msgid_exists_upstream, corpus_of(here))
        assert len(found) == 1
        assert found[0].detail == "no such file in the upstream pin"


class TestS03:
    def test_a_missing_entry_is_reported_as_a_count(self) -> None:
        source = catalog_of(entry("One.", ""), entry("Two.", ""))
        here = catalog_of(entry("One.", "Một."))
        found = findings(
            structure.s03_order_matches_upstream, corpus_of(here, upstream=upstream_of(source))
        )
        assert len(found) == 1
        assert "1 entries here against 2 upstream" in found[0].detail

    def test_a_reordered_catalog_is_found(self) -> None:
        """A reviewer reads a catalog top to bottom against the English file
        beside it, and a corpus that has quietly been sorted costs them that."""
        source = catalog_of(entry("One.", ""), entry("Two.", ""))
        here = catalog_of(entry("Two.", "Hai."), entry("One.", "Một."))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert len(findings(structure.s03_order_matches_upstream, corpus)) == 1

    def test_only_the_first_displacement_is_reported(self) -> None:
        """One inserted entry displaces every entry below it, and reporting all
        of them describes one mistake several thousand times."""
        source = catalog_of(*(entry(f"{n}.", "") for n in range(6)))
        here = catalog_of(*(entry(f"{n}.", "") for n in (5, 0, 1, 2, 3, 4)))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert len(findings(structure.s03_order_matches_upstream, corpus)) == 1

    def test_a_matching_catalog_is_clean(self) -> None:
        source = catalog_of(entry("One.", ""), entry("Two.", ""))
        here = catalog_of(entry("One.", "Một."), entry("Two.", "Hai."))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert findings(structure.s03_order_matches_upstream, corpus) == []


class TestS04:
    """Everything this tool writes is fuzzy, so the absence of the flag is a
    claim that a person read the string."""

    def test_a_translated_entry_with_no_fuzzy_flag_is_found(self) -> None:
        here = catalog_of(Entry(msgid="Return a list.", msgstr="Trả về một danh sách."))
        assert len(findings(structure.s04_translated_entries_are_marked, corpus_of(here))) == 1

    def test_a_fuzzy_entry_is_what_this_tool_writes(self) -> None:
        here = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        assert findings(structure.s04_translated_entries_are_marked, corpus_of(here)) == []

    def test_an_unfuzzied_entry_the_memory_records_as_human_is_allowed(self) -> None:
        """Dropping the fuzzy flag is exactly how a reviewer signs off, so the
        check has to be able to tell that apart from a machine string."""
        here = catalog_of(Entry(msgid="Return a list.", msgstr="Trả về một danh sách."))
        memory = Memory(
            [
                Segment.from_entry(
                    Entry(msgid="Return a list.", msgstr="Trả về một danh sách."),
                    source="human",
                )
            ]
        )
        corpus = corpus_of(here, memory=memory)
        assert findings(structure.s04_translated_entries_are_marked, corpus) == []

    def test_a_machine_record_does_not_excuse_a_missing_flag(self) -> None:
        here = catalog_of(Entry(msgid="Return a list.", msgstr="Trả về một danh sách."))
        memory = Memory([machine_segment("Return a list.", "Trả về một danh sách.")])
        corpus = corpus_of(here, memory=memory)
        assert len(findings(structure.s04_translated_entries_are_marked, corpus)) == 1


class TestS05:
    def test_a_dropped_format_flag_is_found(self) -> None:
        """The flag is what tells gettext to validate the specifiers, so dropping
        it turns a checked string into an unchecked one."""
        source = catalog_of(Entry(msgid="Cannot open %s.", flags=("python-format",)))
        here = catalog_of(entry("Cannot open %s.", "Không thể mở %s."))
        found = findings(
            structure.s05_format_flags_are_preserved, corpus_of(here, upstream=upstream_of(source))
        )
        assert len(found) == 1
        assert "python-format" in found[0].detail

    def test_a_kept_flag_is_clean(self) -> None:
        source = catalog_of(Entry(msgid="Cannot open %s.", flags=("python-format",)))
        here = catalog_of(
            entry("Cannot open %s.", "Không thể mở %s.", flags=("fuzzy", "python-format"))
        )
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert findings(structure.s05_format_flags_are_preserved, corpus) == []

    def test_a_flag_added_here_is_not_this_check_s_business(self) -> None:
        """Only the ones upstream set are checked. An extra flag is a different
        mistake and gettext will say so itself."""
        source = catalog_of(Entry(msgid="Cannot open it."))
        here = catalog_of(
            entry("Cannot open it.", "Không thể mở.", flags=("fuzzy", "python-format"))
        )
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert findings(structure.s05_format_flags_are_preserved, corpus) == []


class TestS06:
    """An obsolete entry is the cheapest thing in the corpus: if the string comes
    back, and they do come back, the work is already done."""

    def test_a_dropped_obsolete_entry_is_found(self) -> None:
        source = catalog_of(
            Entry(msgid="Gone.", msgstr="Đã đi.", flags=("obsolete",)),
            Entry(msgid="Here."),
        )
        here = catalog_of(entry("Here.", "Ở đây."))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert len(findings(structure.s06_obsolete_entries_survive, corpus)) == 1

    def test_a_kept_obsolete_entry_is_clean(self) -> None:
        source = catalog_of(Entry(msgid="Gone.", msgstr="Đã đi.", flags=("obsolete",)))
        here = catalog_of(Entry(msgid="Gone.", msgstr="Đã đi.", flags=("obsolete",)))
        corpus = corpus_of(here, upstream=upstream_of(source))
        assert findings(structure.s06_obsolete_entries_survive, corpus) == []

    def test_a_live_entry_nobody_has_translated_is_not_an_obsolete_one(self) -> None:
        source = catalog_of(Entry(msgid="Here."))
        corpus = corpus_of(catalog_of(), upstream=upstream_of(source))
        assert findings(structure.s06_obsolete_entries_survive, corpus) == []


class TestS07:
    def test_a_plural_needs_a_person(self) -> None:
        """Vietnamese has one plural form and the correct nplurals header is a
        decision for a person, not something a pipeline should guess."""
        here = catalog_of(Entry(msgid="%d file", raw=('msgid_plural "%d files"',)))
        assert len(findings(structure.s07_no_plurals, corpus_of(here))) == 1

    def test_the_corpus_as_it_stands_has_none(self) -> None:
        here = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        assert findings(structure.s07_no_plurals, corpus_of(here)) == []


class TestS08:
    def test_a_run_with_no_memory_cannot_say_anything(self) -> None:
        """Rendering the corpus needs the memory that produced it, and a check
        that guessed would report every file in the repository."""
        here = catalog_of(entry("Return a list.", "Trả về một danh sách."))
        assert findings(structure.s08_apply_is_byte_identical, corpus_of(here)) == []
