from pydocvi import stale
from pydocvi.catalog import segment_id
from pydocvi.memory import Memory, Segment


def seg(msgid: str, source: str = "machine", **extra: object) -> Segment:
    return Segment(
        id=segment_id(msgid),
        msgid=msgid,
        msgstr="x",
        source=source,  # type: ignore[arg-type]
        **extra,  # type: ignore[arg-type]
    )


def words(text: str) -> frozenset[str]:
    """A stand-in matcher. The real one arrives with the glossary at M4."""
    return frozenset(text.lower().replace(".", "").split())


class TestGlossary:
    def test_only_segments_containing_a_changed_term_are_stale(self) -> None:
        memory = Memory(
            [
                seg("a list of items"),
                seg("a dictionary of items"),
                seg("nothing relevant here"),
            ]
        )
        result = stale.by_glossary(memory, ["list"], words)
        assert result.ids == (segment_id("a list of items"),)

    def test_no_changed_terms_means_nothing_is_stale(self) -> None:
        memory = Memory([seg("a list of items")])
        result = stale.by_glossary(memory, [], words)
        assert len(result) == 0
        assert "no terms changed" in result.detail

    def test_human_work_is_protected_by_default(self) -> None:
        """A person decided that wording. A term bump does not overrule them."""
        memory = Memory([seg("a list of items", "human")])
        assert stale.by_glossary(memory, ["list"], words).ids == ()

    def test_human_work_can_be_included_deliberately(self) -> None:
        memory = Memory([seg("a list of items", "human")])
        result = stale.by_glossary(memory, ["list"], words, protect_human=False)
        assert len(result) == 1

    def test_the_detail_names_the_terms(self) -> None:
        result = stale.by_glossary(Memory(), ["list", "iterable"], words)
        assert "iterable, list" in result.detail

    def test_a_bump_touching_one_term_does_not_requeue_the_corpus(self) -> None:
        memory = Memory([seg(f"string number {n}") for n in range(200)] + [seg("a list here")])
        assert len(stale.by_glossary(memory, ["list"], words)) == 1


class TestPrompt:
    def test_segments_from_an_older_prompt_are_stale(self) -> None:
        memory = Memory([seg("one", prompt="old"), seg("two", prompt="new")])
        assert stale.by_prompt(memory, "new").ids == (segment_id("one"),)

    def test_human_work_is_never_stale_for_a_prompt_change(self) -> None:
        memory = Memory([seg("one", "human")])
        assert stale.by_prompt(memory, "new").ids == ()

    def test_a_segment_with_no_recorded_prompt_is_stale(self) -> None:
        assert len(stale.by_prompt(Memory([seg("one")]), "new")) == 1


class TestUpstream:
    def test_segments_upstream_dropped_are_reported(self) -> None:
        memory = Memory([seg("kept"), seg("dropped")])
        result = stale.by_upstream(memory, [segment_id("kept")])
        assert result.ids == (segment_id("dropped"),)
        assert "kept rather than deleted" in result.detail

    def test_nothing_is_stale_when_upstream_still_has_everything(self) -> None:
        memory = Memory([seg("kept")])
        assert len(stale.by_upstream(memory, [segment_id("kept")])) == 0
