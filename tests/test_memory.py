from pathlib import Path

import pytest

from pydocvi.catalog import Entry, segment_id
from pydocvi.memory import PRECEDENCE, Memory, Segment


def seg(msgid: str, msgstr: str, source: str = "machine", **extra: object) -> Segment:
    return Segment(
        id=segment_id(msgid),
        msgid=msgid,
        msgstr=msgstr,
        source=source,  # type: ignore[arg-type]
        **extra,  # type: ignore[arg-type]
    )


class TestPrecedence:
    def test_human_beats_everything(self) -> None:
        assert PRECEDENCE["human"] > max(
            PRECEDENCE[s] for s in ("machine", "passthrough", "legacy")
        )

    def test_legacy_loses_to_everything(self) -> None:
        assert PRECEDENCE["legacy"] < min(
            PRECEDENCE[s] for s in ("human", "machine", "passthrough")
        )

    def test_a_machine_translation_does_not_overwrite_a_human_one(self) -> None:
        memory = Memory([seg("Hello", "Xin chào", "human")])
        assert memory.add(seg("Hello", "Chào", "machine")) is False
        assert memory.lookup("Hello") is not None
        assert memory.lookup("Hello").msgstr == "Xin chào"  # type: ignore[union-attr]

    def test_a_human_translation_overwrites_a_machine_one(self) -> None:
        memory = Memory([seg("Hello", "Chào", "machine")])
        assert memory.add(seg("Hello", "Xin chào", "human")) is True
        assert memory.lookup("Hello").msgstr == "Xin chào"  # type: ignore[union-attr]

    def test_an_equal_source_does_not_churn_the_store(self) -> None:
        """Re-running a tier must not swap one machine translation for another."""
        memory = Memory([seg("Hello", "first", "machine")])
        assert memory.add(seg("Hello", "second", "machine")) is False
        assert memory.lookup("Hello").msgstr == "first"  # type: ignore[union-attr]


class TestStore:
    def test_lookup_by_string_and_by_id_agree(self) -> None:
        memory = Memory([seg("Hello", "Xin chào")])
        assert memory.lookup("Hello") is memory.get(segment_id("Hello"))

    def test_a_missing_segment_is_none_rather_than_an_error(self) -> None:
        assert Memory().lookup("nothing") is None

    def test_counts_report_every_source_even_at_zero(self) -> None:
        counts = Memory([seg("a", "b", "human")]).counts()
        assert counts == {"human": 1, "machine": 0, "passthrough": 0, "legacy": 0}

    def test_remove(self) -> None:
        memory = Memory([seg("Hello", "Xin chào")])
        assert memory.remove(segment_id("Hello")) is True
        assert memory.remove(segment_id("Hello")) is False
        assert len(memory) == 0

    def test_from_entry_carries_provenance(self) -> None:
        entry = Entry(msgid="Hello", msgstr="Xin chào")
        segment = Segment.from_entry(entry, source="machine", batch="b-1", glossary=7)
        assert (segment.batch, segment.glossary, segment.id) == ("b-1", 7, entry.id)


class TestPersistence:
    def test_saves_and_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        Memory([seg("Hello", "Xin chào", "human", batch="b-1")]).save(path)
        loaded = Memory.load(path)
        assert len(loaded) == 1
        assert loaded.lookup("Hello").batch == "b-1"  # type: ignore[union-attr]

    def test_loading_a_missing_file_gives_an_empty_store(self, tmp_path: Path) -> None:
        assert len(Memory.load(tmp_path / "absent.json")) == 0

    def test_the_same_state_produces_the_same_bytes(self, tmp_path: Path) -> None:
        """Two runs that reached the same state must not show as a changed file."""
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        Memory([seg("one", "một"), seg("two", "hai")]).save(a)
        Memory([seg("two", "hai"), seg("one", "một")]).save(b)
        assert a.read_bytes() == b.read_bytes()

    def test_vietnamese_is_stored_unescaped(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        Memory([seg("Hello", "Xin chào")]).save(path)
        assert "Xin chào" in path.read_text(encoding="utf-8")

    def test_saving_with_no_path_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no path"):
            Memory().save()

    def test_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        Memory([seg("Hello", "Xin chào")]).save(tmp_path / "memory.json")
        assert [p.name for p in tmp_path.iterdir()] == ["memory.json"]
