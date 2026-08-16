from pathlib import Path

from pydocvi import catalog, sync
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


def test_diff_reports_upstream_strings_the_memory_lacks(data_dir: Path) -> None:
    cat = catalog.read(data_dir / "small.po")
    changes = sync.diff(Memory(), [cat])
    assert len(changes.added) == 4
    assert changes.orphaned == ()
    assert not changes.clean


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
