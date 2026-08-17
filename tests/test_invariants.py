import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pydocvi import invariants, parse
from pydocvi.segment import protect

SOURCE = "Return a :class:`list` whose items are ``sorted``."
GOOD = "Trả về một ⟦1⟧ có các phần tử đã được ⟦2⟧."


def rules(msgid: str, msgstr: str) -> set[str]:
    spans = protect(msgid).spans
    return {v.rule for v in invariants.check_entry(msgid, msgstr, spans)}


def test_a_good_translation_breaks_nothing() -> None:
    assert rules(SOURCE, GOOD) == set()


class TestP01:
    def test_a_missing_marker(self) -> None:
        assert "P01" in rules(SOURCE, "Trả về một danh sách có phần tử ⟦2⟧.")

    def test_a_repeated_marker(self) -> None:
        assert "P01" in rules(SOURCE, "Trả về ⟦1⟧ và ⟦1⟧ và ⟦2⟧.")

    def test_an_invented_marker(self) -> None:
        assert "P01" in rules(SOURCE, "Trả về ⟦1⟧ ⟦2⟧ ⟦3⟧.")

    def test_a_spaced_marker(self) -> None:
        assert "P01" in rules(SOURCE, "Trả về ⟦ 1 ⟧ và ⟦2⟧.")

    def test_a_malformed_marker_beside_a_complete_set(self) -> None:
        """Every expected marker is there and correct, so nothing above this
        rule fires, and the leftover bracket pair would otherwise reach the
        catalog as text."""
        assert "P01" in rules(SOURCE, "Trả về ⟦1⟧ và ⟦2⟧ ⟦ ba ⟧.")

    def test_reordering_is_legal(self) -> None:
        """Vietnamese does not put modifiers where English does, so a rule
        against moving a marker would be a rule against translating well."""
        assert rules(SOURCE, "⟦2⟧ là cách các phần tử của ⟦1⟧ được sắp xếp.") == set()


class TestP02:
    def test_a_span_typed_out_beside_its_marker(self) -> None:
        broken = "Mô-đun ⟦1⟧ tức là :mod:`os` đó."
        assert "P02" in rules("The :mod:`os` module.", broken)


class TestP04:
    def test_leading_whitespace_is_carried_over(self) -> None:
        assert "P04" in rules("  indented source", "không thụt lề")

    def test_a_trailing_newline_is_carried_over(self) -> None:
        assert "P04" in rules("Câu.\n", "Câu đã dịch.")

    def test_matching_edges_pass(self) -> None:
        assert "P04" not in rules(" Câu. ", " Câu đã dịch rồi đây. ")


class TestP05:
    def test_an_empty_translation(self) -> None:
        assert "P05" in rules("Return a value.", "   ")

    def test_the_english_handed_straight_back(self) -> None:
        assert "P05" in rules("Return a value.", "Return a value.")


class TestP06AndP07:
    def test_narration(self) -> None:
        assert "P06" in rules("Return a value.", "Here is the translation: Trả về giá trị.")

    def test_a_fence(self) -> None:
        broken = rules("Return a value.", "```\nTrả về một giá trị nào đó.\n```")
        assert {"P06", "P07"} <= broken

    def test_both_reject_the_whole_batch(self) -> None:
        """Narration at the top means the model was not doing the task, so
        there is nothing in the answer worth keeping."""
        violation = invariants.Violation(rule="P06", detail="narrated")
        assert violation.rejects_batch

    def test_a_lost_marker_costs_only_its_entry(self) -> None:
        assert not invariants.Violation(rule="P01", detail="missing").rejects_batch


class TestP08:
    def test_a_long_answer_with_no_diacritics(self) -> None:
        assert "P08" in rules("x", "This sentence never became Vietnamese at all.")

    def test_a_short_answer_without_diacritics_is_ordinary(self) -> None:
        """ "API" and "Unicode" are what a Vietnamese reader expects to see."""
        assert "P08" not in rules("API", "API")

    @pytest.mark.parametrize("text", ["Chuỗi đã được sắp xếp theo thứ tự tăng dần hoàn toàn"])
    def test_composed_accents_count_as_vietnamese(self, text: str) -> None:
        assert "P08" not in rules("x", text)

    def test_the_letter_d_with_a_stroke_is_enough(self) -> None:
        assert "P08" not in rules("x", "day la mot chuoi rat dai khong co dau thanh dieu đ")


class TestP09:
    """A format specifier is itself a protected span, so this rule runs against
    the restored translation. Against the answer as it arrives it compared a
    source holding %s to a translation holding the marker that replaced it, and
    so refused a perfect translation of every entry carrying one: 617 entries of
    this corpus, in 391 of its 2 776 batches."""

    def test_a_translation_that_kept_the_marker_kept_the_specifier(self) -> None:
        assert rules("Return %s items from it.", "Trả về ⟦1⟧ mục từ đó.") == set()

    def test_a_brace_specifier_is_a_span_like_any_other(self) -> None:
        assert rules("Value {name} here.", "Giá trị ⟦1⟧ ở đây.") == set()

    def test_a_specifier_invented_in_the_prose_is_caught(self) -> None:
        assert "P09" in rules("Open the file.", "Không thể mở %s tệp.")

    def test_a_specifier_inside_link_text_is_caught_when_it_is_dropped(self) -> None:
        """The one place a specifier reaches the model unprotected. A hyperlink
        reference is the only span that is part prose, and a specifier sitting in
        that prose is the model's to copy."""
        found = rules("See `the %s guide <https://x.example/g>`_ here.", "Xem ⟦1⟧hướng dẫn⟦2⟧.")
        assert "P09" in found

    def test_an_answer_with_broken_markers_is_not_this_rule_to_report(self) -> None:
        """It cannot be restored, so the specifiers cannot be counted. P01 and
        P02 have already said what went wrong, and a third sentence about the
        same mistake would make the retry advice name three rules for one."""
        assert rules("Cannot open %s here.", "Không thể mở tệp ở đây.") == {"P01", "P02"}


class TestP03:
    def test_a_complete_answer_aligns(self) -> None:
        assert invariants.check_answer(parse.parse("1 Một.\n2 Hai.", 2), 2) == []

    def test_a_gap_rejects_the_answer(self) -> None:
        found = invariants.check_answer(parse.parse("1 Một.", 2), 2)
        assert [v.rule for v in found] == ["P03"]
        assert found[0].rejects_batch


class TestReporting:
    def test_a_violation_carries_its_entry_index(self) -> None:
        spans = protect(SOURCE).spans
        found = invariants.check_entry(SOURCE, "Trả về ⟦1⟧.", spans, index=17)
        assert all(v.index == 17 for v in found)
        assert str(found[0]).startswith("P01: entry 17:")

    def test_every_rule_is_reported_not_just_the_first(self) -> None:
        """An entry that lost a marker and also came back in English is worth
        reporting twice, because the second fact changes what the retry says."""
        assert {"P01", "P05"} <= rules(SOURCE, SOURCE)


class TestProperties:
    @given(st.text(), st.text())
    @settings(max_examples=500)
    def test_checking_arbitrary_model_output_never_raises(self, msgid: str, msgstr: str) -> None:
        """Every string here came out of a model, so the check that decides
        whether to keep it has to survive whatever the model sent."""
        spans = protect(msgid).spans
        for violation in invariants.check_entry(msgid, msgstr, spans):
            assert violation.rule.startswith("P")

    @given(st.text(min_size=1))
    @settings(max_examples=200)
    def test_handing_the_english_back_is_always_refused(self, msgid: str) -> None:
        if not msgid.strip():
            return
        assert "P05" in rules(msgid, msgid)
