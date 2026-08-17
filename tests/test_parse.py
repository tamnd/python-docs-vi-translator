import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import parse


class TestMarkers:
    @pytest.mark.parametrize("marker", ["1 ", "1. ", "1) ", "1: ", "1] "])
    def test_the_shapes_a_model_actually_uses(self, marker: str) -> None:
        answer = parse.parse(f"{marker}Câu một.", 1)
        assert answer.entries == {1: "Câu một."}

    def test_a_body_running_over_several_lines(self) -> None:
        answer = parse.parse("1 Dòng một\nvà dòng hai.\n2 Câu hai.", 2)
        assert answer.entries[1] == "Dòng một\nvà dòng hai."

    def test_a_number_inside_a_body_is_not_a_marker(self) -> None:
        """Only a number at the start of a line opens an entry, so a translation
        that happens to mention 3.14 keeps it."""
        answer = parse.parse("1 Kể từ phiên bản 3.14 trở đi.", 1)
        assert answer.entries == {1: "Kể từ phiên bản 3.14 trở đi."}


class TestPartialCredit:
    def test_one_bad_entry_does_not_cost_the_others(self) -> None:
        """The whole reason the format is a numbered list and not JSON. At 151
        seconds a call, partial credit is the game."""
        answer = parse.parse("1 Một.\n2 Hai.\n99 Lạc.\n3 Ba.", 3)
        assert sorted(answer.entries) == [1, 2, 3]

    def test_a_number_that_does_not_continue_the_sequence_is_text(self) -> None:
        """Where a stray number is genuinely stray this puts it in the entry
        before it, and the invariants then have something to refuse. The
        alternative was dropping it silently, which is what used to truncate an
        entry whose own text contains a numbered list."""
        answer = parse.parse("1 Một.\n2 Hai.\n99 Lạc.\n3 Ba.", 3)
        assert answer.entries[2] == "Hai.\n99 Lạc."
        assert any("read as text" in p.kind for p in answer.problems)

    def test_a_repeated_index_does_not_replace_what_it_repeats(self) -> None:
        answer = parse.parse("1 Đầu.\n1 Lặp.\n2 Hai.", 2)
        assert answer.entries[1].startswith("Đầu.")
        assert answer.entries[2] == "Hai."

    def test_a_missing_index_is_reported_against_itself(self) -> None:
        answer = parse.parse("1 Một.\n3 Ba.", 3)
        assert 2 in answer.missing(3)
        assert any(p.index == 2 for p in answer.problems)

    def test_a_gap_costs_everything_after_it_and_that_is_the_trade(self) -> None:
        """An answer with a gap in it fails ``P03`` and the batch is retried
        whole, so nothing that would have been used is lost by being strict
        here."""
        answer = parse.parse("1 Một.\n3 Ba.", 3)
        assert sorted(answer.entries) == [1]

    def test_an_entry_whose_own_text_is_a_numbered_list_survives_whole(self) -> None:
        """From ``logging-cookbook.po``, which has four such entries. This used
        to keep the first line and drop the rest, silently, and a half
        translated string passes every invariant a short one passes."""
        answer = parse.parse("1 Dòng đầu.\n3. Dòng sau.\n2 Câu hai.", 2)
        assert answer.entries[1] == "Dòng đầu.\n3. Dòng sau."
        assert answer.entries[2] == "Câu hai."

    def test_an_empty_body_is_refused_for_that_entry_only(self) -> None:
        answer = parse.parse("1 Một.\n2 \n3 Ba.", 3)
        assert sorted(answer.entries) == [1, 3]

    def test_an_answer_with_no_markers_at_all(self) -> None:
        answer = parse.parse("Tôi không hiểu.", 3)
        assert not answer.usable
        assert answer.problems[0].kind == "no numbered entries found"


class TestNarrationAndFences:
    def test_a_preamble_is_reported_but_the_entries_survive(self) -> None:
        answer = parse.parse("Here is the translation:\n1 Một.", 1)
        assert answer.entries == {1: "Một."}
        assert any(p.kind == "preamble before the first entry" for p in answer.problems)

    def test_a_fenced_answer_is_flagged(self) -> None:
        assert parse.parse("```\n1 Một.\n```", 1).fenced


class TestWhitespace:
    def test_trailing_spaces_in_a_body_are_dropped(self) -> None:
        assert parse.parse("1 Một.   \n2 Hai.", 2).entries[1] == "Một."

    def test_the_newlines_the_format_introduced_are_stripped(self) -> None:
        assert parse.parse("1 \n\nMột.\n\n2 Hai.", 2).entries[1] == "Một."


class TestAlignment:
    def test_a_complete_answer_is_aligned(self) -> None:
        assert parse.aligned(parse.parse("1 Một.\n2 Hai.", 2), 2)

    def test_a_gap_is_not_aligned(self) -> None:
        assert not parse.aligned(parse.parse("1 Một.", 2), 2)

    def test_a_problem_reads_as_a_sentence(self) -> None:
        assert str(parse.Problem(kind="index repeated", index=4)) == "entry 4: index repeated"

    def test_a_problem_with_no_index_names_the_answer(self) -> None:
        assert str(parse.Problem(kind="empty", detail="x")) == "answer: empty (x)"


class TestProperties:
    @given(st.text(), st.integers(min_value=1, max_value=40))
    @settings(max_examples=500)
    def test_no_answer_produces_an_index_outside_the_batch(self, text: str, count: int) -> None:
        """An index the batch does not have would write a translation onto the
        wrong entry, which is the one failure mode nothing downstream can see."""
        answer = parse.parse(text, count)
        assert all(1 <= index <= count for index in answer.entries)

    @given(st.text(), st.integers(min_value=1, max_value=40))
    @settings(max_examples=200)
    def test_a_body_is_never_invented(self, text: str, count: int) -> None:
        answer = parse.parse(text, count)
        assert all(body.strip() for body in answer.entries.values())


class TestTheUnicodeTable:
    def test_a_body_whose_own_lines_count_upwards_is_the_one_that_cannot_work(self) -> None:
        """One entry in the corpus is a codepoint table whose lines begin 0, 1,
        2, 3, 4. That is a numbered sequence, and no parser reading a line at a
        time can tell it from an answer. It fails ``P03``, is retried, dies, and
        is left in English, which is what the ladder is for."""
        answer = parse.parse("1 0 aaa\n1 bbb\n2 ccc", 2)
        assert answer.entries[2] == "ccc"
        assert answer.entries[1] != "0 aaa\n1 bbb\n2 ccc"
