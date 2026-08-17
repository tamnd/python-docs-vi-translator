from pydocvi import batch, render, translate
from pydocvi.catalog import Entry
from pydocvi.translate import Attempt


def make(*msgids: str, path: str = "f.po") -> batch.Batch:
    return next(iter(batch.pack(path, batch.items([Entry(msgid=m) for m in msgids]))))


def answered(*bodies: str) -> str:
    return "\n".join(f"{number} {body}" for number, body in enumerate(bodies, 1))


class TestPartialCredit:
    def test_one_broken_entry_costs_only_itself(self) -> None:
        """The single most important operational decision in the design notes.
        At a minute and a half a call, throwing away 39 good translations
        because of one bad one produces no better corpus."""
        one = make("Return :func:`len` of it.", "Chào các bạn.", "Another sentence here.")
        outcome = translate.read(
            one, answered("Trả về ⟦1⟧ của nó.", "Xin chào mọi người.", "Một câu khác ở đây.")
        )
        assert len(outcome.accepted) == 3
        assert outcome.refused == ()

    def test_a_lost_marker_refuses_that_entry_and_no_other(self) -> None:
        one = make("Return :func:`len` of it.", "Chào các bạn.")
        outcome = translate.read(one, answered("Trả về của nó.", "Xin chào mọi người."))
        assert {refused.index for refused in outcome.refused} == {1}
        assert [accepted.index for accepted in outcome.accepted] == [2]
        assert "P01" in {refused.rule for refused in outcome.refused}

    def test_one_entry_that_broke_two_rules_is_reported_under_both(self) -> None:
        """A missing marker is both a marker that is gone and a translation the
        markup cannot go back into. Reporting only the first would name one rule
        in the retry advice where the model needs the other."""
        one = make("Return :func:`len` of it.")
        outcome = translate.read(one, answered("Trả về của nó."))
        assert {refused.rule for refused in outcome.refused} == {"P01", "P02"}

    def test_the_rate_is_the_share_that_came_back_usable(self) -> None:
        """Counted in entries and not in refusals. Entry 1 here breaks two rules
        and is reported twice, and a rate that counted it twice would read 0.333
        and fall further every time the reporting got better."""
        one = make("Return :func:`len` of it.", "Chào các bạn.")
        outcome = translate.read(one, answered("Trả về của nó.", "Xin chào mọi người."))
        assert outcome.rate == 0.5

    def test_an_accepted_entry_carries_the_markup_back(self) -> None:
        one = make("Return :func:`len` of it.")
        outcome = translate.read(one, answered("Trả về ⟦1⟧ của nó."))
        assert outcome.accepted[0].msgstr == "Trả về :func:`len` của nó."

    def test_an_accepted_entry_names_the_segment_it_belongs_to(self) -> None:
        """The index is a position in one call and the segment is the identity
        of a string. A translation written onto the wrong entry is the one
        failure nothing downstream can see."""
        one = make("Chào các bạn.")
        outcome = translate.read(one, answered("Xin chào mọi người."))
        assert outcome.accepted[0].segment == one.items[0].segment


class TestTheWholeBatch:
    def test_a_misaligned_answer_rejects_everything(self) -> None:
        """Alignment is a property of the whole answer, so an entry-by-entry
        report of it would name nine failures and hide the one fact that
        explains them."""
        one = make("Chào các bạn.", "Another sentence here.")
        outcome = translate.read(one, "1 Xin chào mọi người.")
        assert outcome.whole_batch_refused
        assert outcome.rejected.startswith("P03")
        assert outcome.accepted == ()

    def test_narration_rejects_everything(self) -> None:
        """Narration means the model was not doing the task."""
        one = make("Chào các bạn.", "Another sentence here.")
        outcome = translate.read(
            one, answered("Here is the translation: Xin chào.", "Một câu khác ở đây.")
        )
        assert outcome.whole_batch_refused
        assert outcome.rejected.startswith("P06")

    def test_a_fenced_answer_rejects_everything(self) -> None:
        one = make("Chào các bạn.")
        outcome = translate.read(one, "1 ```\nXin chào mọi người.\n```")
        assert outcome.whole_batch_refused

    def test_an_answer_with_nothing_in_it_rejects_everything(self) -> None:
        assert translate.read(make("Chào các bạn."), "Tôi không hiểu.").whole_batch_refused


class TestReporting:
    def test_a_refusal_reads_as_the_rule_the_entry_and_the_reason(self) -> None:
        """The same shape a violation prints in, because a refusal in a trace
        and a violation in an audit are the same fact seen twice."""
        one = translate.Refused(segment="a", index=17, rule="P01", detail="marker(s) missing: ⟦1⟧")
        assert str(one) == "P01: entry 17: marker(s) missing: ⟦1⟧"


class TestAdvice:
    def test_the_failure_is_named_rather_than_described_as_a_failure(self) -> None:
        """A generic "try again" is worth nothing. The same prompt against the
        same session returns the same answer."""
        refused = [translate.Refused(segment="a", index=1, rule="P01", detail="marker missing")]
        assert "⟦n⟧" in translate.advice(refused)

    def test_one_sentence_per_rule_however_many_entries_broke_it(self) -> None:
        refused = [
            translate.Refused(segment=str(n), index=n, rule="P01", detail="marker missing")
            for n in range(30)
        ]
        assert len(translate.advice(refused).splitlines()) == 1

    def test_two_rules_get_two_sentences(self) -> None:
        refused = [
            translate.Refused(segment="a", index=1, rule="P01", detail="marker missing"),
            translate.Refused(segment="b", index=2, rule="P08", detail="no diacritic"),
        ]
        assert len(translate.advice(refused).splitlines()) == 2

    def test_nothing_refused_says_nothing(self) -> None:
        assert translate.advice([]) == ""

    def test_a_rule_with_nothing_specific_to_say_still_says_something(self) -> None:
        refused = [translate.Refused(segment="a", index=1, rule="P99", detail="?")]
        assert translate.GENERIC in translate.advice(refused)

    def test_the_advice_reaches_the_prompt_above_the_entries(self) -> None:
        one = make("Chào các bạn.")
        text = render.user(one, advice="Some strings came back wrong.")
        assert text.index("came back wrong") < text.index("1 Chào các bạn.")


class TestTheLadder:
    def test_the_second_rung_retries_the_failures_together(self) -> None:
        one = make("First one here.", "Second one here.", "Third one here.")
        refused = [
            translate.Refused(segment=one.items[0].segment, index=1, rule="P01", detail=""),
            translate.Refused(segment=one.items[2].segment, index=3, rule="P01", detail=""),
        ]
        again = translate.again(one, refused, attempt=Attempt.NAMED)
        assert len(again) == 1
        assert [item.msgid for item in again[0].items] == ["First one here.", "Third one here."]

    def test_the_third_rung_gives_each_entry_the_call_to_itself(self) -> None:
        one = make("First one here.", "Second one here.", "Third one here.")
        refused = [
            translate.Refused(segment=one.items[0].segment, index=1, rule="P01", detail=""),
            translate.Refused(segment=one.items[2].segment, index=3, rule="P01", detail=""),
        ]
        again = translate.again(one, refused, attempt=Attempt.ALONE)
        assert [len(part) for part in again] == [1, 1]

    def test_the_fourth_rung_is_dead_and_sends_nothing(self) -> None:
        one = make("First one here.")
        refused = [translate.Refused(segment=one.items[0].segment, index=1, rule="P01", detail="")]
        assert translate.again(one, refused, attempt=Attempt.DEAD) == []

    def test_nothing_refused_sends_nothing(self) -> None:
        assert translate.again(make("First one here."), [], attempt=Attempt.NAMED) == []

    def test_a_retry_is_ordered_by_the_batch_rather_than_by_the_refusals(self) -> None:
        one = make("First one here.", "Second one here.")
        refused = [
            translate.Refused(segment=one.items[1].segment, index=2, rule="P01", detail=""),
            translate.Refused(segment=one.items[0].segment, index=1, rule="P01", detail=""),
        ]
        again = translate.again(one, refused, attempt=Attempt.NAMED)
        assert [item.msgid for item in again[0].items] == ["First one here.", "Second one here."]

    def test_an_entry_is_carried_by_segment_and_not_by_index(self) -> None:
        """The indices of the next attempt are not the indices of this one, and
        an entry retried under the wrong number is a translation written onto
        another string."""
        one = make("First one here.", "Second one here.", "Third one here.")
        refused = [translate.Refused(segment=one.items[2].segment, index=3, rule="P01", detail="")]
        again = translate.again(one, refused, attempt=Attempt.NAMED)
        assert again[0].items[0].segment == one.items[2].segment

    def test_a_retry_batch_is_identified_by_what_is_in_it(self) -> None:
        one = make("First one here.", "Second one here.")
        refused = [translate.Refused(segment=one.items[0].segment, index=1, rule="P01", detail="")]
        again = translate.again(one, refused, attempt=Attempt.NAMED)
        assert again[0].id != one.id
        assert again[0].id == translate.again(one, refused, attempt=Attempt.NAMED)[0].id
