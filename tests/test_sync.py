from pathlib import Path

from conftest import catalog_of, entry
from pydocvi import catalog, sync
from pydocvi.catalog import Catalog
from pydocvi.memory import Memory


def test_pin_serialises_the_branch_as_a_string() -> None:
    """``branch: 3.15`` unquoted is a float, and reads back as 3.1."""
    pin = sync.Pin(
        repo="tamnd/python-docs-vi",
        branch="3.15",
        commit="da475ff",
        files=548,
        entries=87_008,
        words=1_711_382,
        characters=12_526_506,
        translated=1_435,
    )
    text = pin.as_yaml()
    assert 'branch: "3.15"' in text
    assert "entries: 87008" in text
    assert text.startswith("# Written by pydocvi sync")


def test_measure_counts_entries_words_and_translations(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    files, entries, words, characters, translated = sync.measure([cat])
    assert files == 1
    assert entries == 4
    assert translated == 1
    assert words > 0
    assert characters > words


def test_human_segments_skip_untranslated_entries(data_dir: Path) -> None:
    segments = sync.human_segments([catalog.read(data_dir / "small.po")])
    assert [s.msgstr for s in segments] == ["Xử lý lỗi"]


def test_human_segments_skip_fuzzy_entries(data_dir: Path) -> None:
    """A fuzzy string is one gettext is not confident about, so it is not truth."""
    cat = catalog.read(data_dir / "small.po")
    fuzzed = cat.replace_entries(
        [cat.entries[0].with_msgstr("Xử lý lỗi", fuzzy=True), *cat.entries[1:]]
    )
    assert sync.human_segments([fuzzed]) == []


def reviewed(msgid: str, msgstr: str) -> Catalog:
    """One catalog holding one entry a person signed off on."""
    return catalog_of(entry(msgid, msgstr, flags=()))


def test_human_segments_skip_code_however_translated_it_looks() -> None:
    """A doctest is copied and never translated, which is ``P07``'s rule, and the
    mirror hands over 136 code entries as somebody's work. 30 of those are a
    person having typed over the code: ``File "<stdin>", line 1, in <module>``
    arrives as ``File "1", line 1, in 2``. ``human`` says who typed the string,
    not that the string is right."""
    coded = reviewed(
        '>>> n\n  File "<stdin>", line 1, in <module>', '>>> n\nFile "1", line 1, in 2'
    )
    assert sync.human_segments([coded]) == []


def test_human_segments_skip_code_that_was_copied_correctly_too() -> None:
    """The other 106 lose nothing by going. ``apply`` mints them from the
    ``msgid`` with ``passthrough=doctest`` on them, which is the same string with
    an accurate account of where it came from."""
    same = ">>> len([1, 2])\n2"
    assert sync.human_segments([reviewed(same, same)]) == []


def test_human_segments_keep_a_no_op_a_person_translated() -> None:
    """Only code is dropped, not everything the classifier calls non-prose.

    This entry is the one that found the ``is_noop`` bug, back when it was a
    no-op: a ``:ref:`` whose display text a person had translated correctly,
    which made it the one non-prose entry in the mirror whose translation was
    not a copy of its source. The classifier calls it prose now and the
    assertion is unchanged, because the point was never what kind it is. A
    translation is kept unless the entry is code."""
    noop = reviewed(
        ":ref:`Documentation on attributes <class-attrs>`.",
        ":ref:`Tài liệu về các thuộc tính <class-attrs>`.",
    )
    assert len(sync.human_segments([noop])) == 1


def test_diff_reports_upstream_strings_the_memory_lacks(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    changes = sync.diff(Memory(), [cat])
    assert len(changes.added) == 4
    assert changes.orphaned == ()
    assert not changes.clean


def test_load_human_drops_a_stale_human_segment() -> None:
    """The case this stopped being an ``extend`` for. A doctest the mirror once
    offered as somebody's translation is still in the memory after the rule that
    admitted it got stricter, and ``apply`` would write it back over the code."""
    doctest = entry('>>> n\n  File "<stdin>", line 1, in <module>', "", flags=())
    memory = Memory([sync.Segment.from_entry(doctest, source="human")])
    loaded = sync.load_human(memory, [catalog_of(entry("Return a list.", "Trả về.", flags=()))])
    assert loaded.stored == 1
    assert loaded.dropped == 1
    assert [s.msgid for s in memory] == ["Return a list."]


def test_load_human_leaves_machine_segments_where_they_are() -> None:
    """A machine segment is the one thing here that cannot be rebuilt without
    spending the run again, and the mirror is no evidence either way about it."""
    memory = Memory([sync.Segment(id="0" * 16, msgid="x", msgstr="y", source="machine")])
    loaded = sync.load_human(memory, [catalog_of(entry("Return a list.", "Trả về.", flags=()))])
    assert loaded.dropped == 0
    assert {s.source for s in memory} == {"machine", "human"}


def test_diff_reports_orphans(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    memory = Memory()
    sync.load_human(memory, [cat])
    memory.add(
        sync.Segment(id="0" * 16, msgid="a string upstream dropped", msgstr="x", source="machine")
    )
    changes = sync.diff(memory, [cat])
    assert changes.orphaned == ("0" * 16,)


def test_a_matching_memory_is_a_clean_diff(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    memory = Memory(
        [
            sync.Segment(id=entry.id, msgid=entry.msgid, msgstr="x", source="machine")
            for entry in cat
        ]
    )
    assert sync.diff(memory, [cat]).clean


def test_the_report_explains_why_orphans_are_kept() -> None:
    text = sync.SyncDiff(orphaned=("abc",)).as_markdown()
    assert "kept rather than deleted" in text
