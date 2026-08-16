from pathlib import Path

from pydocvi import batch, catalog
from pydocvi.batch import CHARACTER_CAP, ENTRY_CAP, SPAN_CAP
from pydocvi.catalog import Entry


def item(msgid: str) -> batch.Item:
    entry = Entry(msgid=msgid)
    return batch.items([entry])[0]


class TestCaps:
    def test_a_batch_is_cut_at_the_entry_cap(self) -> None:
        batches = list(batch.pack("f.po", [item(f"Sentence number {n}.") for n in range(90)]))
        assert [len(b) for b in batches] == [ENTRY_CAP, ENTRY_CAP, 10]

    def test_a_batch_is_cut_at_the_character_cap(self) -> None:
        batches = list(batch.pack("f.po", [item("word " * 200) for _ in range(10)]))
        assert all(b.characters <= CHARACTER_CAP for b in batches)
        assert len(batches) > 1

    def test_a_batch_is_cut_at_the_span_cap(self) -> None:
        """The cap that was learned the hard way. A body with 45 opaque spans in
        it was refused four times running, every time for the spans."""
        heavy = item(" ".join(f":func:`f{n}`" for n in range(20)) + " and some prose here.")
        batches = list(batch.pack("f.po", [heavy] * 6))
        assert all(b.spans <= SPAN_CAP for b in batches)

    def test_an_entry_over_a_cap_goes_alone_rather_than_being_split(self) -> None:
        """A batch is never cut below an entry, so a refusal costs one entry
        rather than thirty-nine innocent ones sharing the call."""
        batches = list(batch.pack("f.po", [item("a"), item("x" * (CHARACTER_CAP + 1)), item("b")]))
        assert [len(b) for b in batches] == [1, 1, 1]
        assert batches[1].oversized

    def test_a_file_with_nothing_to_translate_yields_no_batch(self) -> None:
        """Plenty of files are all code blocks and version markers, and an empty
        batch would be a call with nothing in it."""
        assert list(batch.pack("f.po", [])) == []

    def test_an_ordinary_batch_is_not_oversized(self) -> None:
        assert not next(iter(batch.pack("f.po", [item("Short.")]))).oversized


class TestIdentity:
    def test_the_same_entries_give_the_same_id(self) -> None:
        first = list(batch.pack("f.po", [item("One."), item("Two.")]))
        second = list(batch.pack("f.po", [item("One."), item("Two.")]))
        assert first[0].id == second[0].id

    def test_a_different_file_gives_a_different_id(self) -> None:
        assert batch.batch_id("a.po", ["x"]) != batch.batch_id("b.po", ["x"])

    def test_a_different_entry_gives_a_different_id(self) -> None:
        assert batch.batch_id("a.po", ["x"]) != batch.batch_id("a.po", ["y"])

    def test_the_id_is_hex_of_a_fixed_length(self) -> None:
        value = batch.batch_id("a.po", ["x"])
        assert len(value) == batch.ID_LENGTH
        assert set(value) <= set("0123456789abcdef")


class TestBuild:
    def test_passthrough_entries_are_never_batched(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        batched = {i.msgid for b in batch.build([cat]) for i in b.items}
        assert ":func:`sorted`" not in batched

    def test_batches_never_span_two_files(self, tmp_path: Path, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        other = catalog.read(data_dir / "small.po")
        batches = batch.build([cat, other])
        assert len(batches) == 2
        assert all(len({i.segment for i in b.items}) == len(b.items) for b in batches)

    def test_the_path_is_relative_to_the_root_when_given(self, data_dir: Path) -> None:
        cat = catalog.read(data_dir / "small.po")
        assert batch.build([cat], root=data_dir)[0].path == "small.po"

    def test_the_path_falls_back_to_the_file_name(self, data_dir: Path) -> None:
        assert batch.build([catalog.read(data_dir / "small.po")])[0].path == "small.po"


class TestStats:
    def test_totals_and_averages(self) -> None:
        batches = list(batch.pack("f.po", [item("One sentence."), item("Another sentence.")]))
        measured = batch.stats(batches)
        assert measured.batches == 1
        assert measured.entries == 2
        assert measured.entries_per_batch == 2

    def test_stats_of_nothing_does_not_divide_by_zero(self) -> None:
        measured = batch.stats([])
        assert measured.entries_per_batch == 0
        assert measured.characters_per_batch == 0

    def test_by_file_counts_batches(self) -> None:
        batches = list(batch.pack("a.po", [item(f"Sentence {n}.") for n in range(50)]))
        assert batch.by_file(batches) == {"a.po": 2}
