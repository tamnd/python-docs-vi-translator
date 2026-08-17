"""Curation: the batches, the prompt, and the four rules on a returned line.

The rule worth reading these tests for is ``G-a``. An off-by-one in the model's
numbering attaches every rendering to the term above it, and every row in the
result is a real Vietnamese phrase, so nothing downstream can see it. ``G-a`` is
the only thing between that and a shipped glossary.
"""

import asyncio
import re

import pytest

from conftest import make_route
from pydocvi import curate
from pydocvi.client import Answer
from pydocvi.curate import PHRASE_WORDS, UNSURE, Batch
from pydocvi.glossary import KEEP
from pydocvi.mine import Candidate, Source


def candidate(en: str, **overrides: object) -> Candidate:
    return Candidate(en=en, source=Source.TERM_PAGE, **overrides)  # type: ignore[arg-type]


def batch(*names: str) -> Batch:
    return Batch(index=1, candidates=tuple(candidate(name) for name in names))


def answered(*lines: str) -> str:
    return "\n".join(lines)


class TestBatching:
    def test_the_candidates_are_cut_into_calls_of_forty(self):
        made = curate.batches([candidate(f"term {n}") for n in range(95)])
        assert [len(one) for one in made] == [40, 40, 15]

    def test_the_order_the_candidates_arrived_in_is_kept(self):
        made = curate.batches([candidate("alpha"), candidate("beta")], size=2)
        assert [one.en for one in made[0].candidates] == ["alpha", "beta"]

    def test_batches_are_numbered_from_one(self):
        made = curate.batches([candidate("a"), candidate("b")], size=1)
        assert [one.index for one in made] == [1, 2]

    def test_nothing_to_curate_is_no_calls(self):
        assert curate.batches([]) == []

    def test_a_batch_is_identified_by_what_is_in_it(self):
        assert batch("a", "b").id == batch("a", "b").id

    def test_a_different_batch_has_a_different_identity(self):
        assert batch("a", "b").id != batch("a", "c").id


class TestPrompt:
    def test_every_candidate_is_numbered(self):
        text = curate.prompt(batch("iterable", "decorator"))
        assert "1. iterable" in text and "2. decorator" in text

    def test_a_definition_goes_in_beside_the_term(self):
        one = Batch(index=1, candidates=(candidate("iterable", definition="An object."),))
        assert "context: An object." in curate.prompt(one)

    def test_a_term_with_no_definition_is_asked_about_bare(self):
        assert "context:" not in curate.prompt(batch("iterable"))

    def test_the_renderings_already_seen_are_shown(self):
        one = Batch(index=1, candidates=(candidate("iterable", seen=("khả lặp", "lặp được")),))
        assert "already rendered as: khả lặp, lặp được" in curate.prompt(one)

    def test_the_answer_format_is_stated(self):
        assert "copied exactly" in curate.prompt(batch("iterable"))

    def test_the_way_out_is_offered_and_called_a_wanted_answer(self):
        """The single most important line in the prompt. A model with no way out
        invents, and an invented rendering survives every mechanical check."""
        text = curate.prompt(batch("iterable"))
        assert f"{UNSURE} is a wanted answer" in text

    def test_keeping_the_english_is_offered_and_called_normal(self):
        assert f"{KEEP} is a normal answer" in curate.prompt(batch("iterable"))

    def test_the_prompt_is_identified_so_a_row_records_what_asked_for_it(self):
        assert len(curate.prompt_id()) == 12

    def test_the_identity_is_a_function_of_the_template(self):
        assert curate.prompt_id() == curate.prompt_id()


class TestAccepting:
    def test_a_well_formed_line_becomes_a_row(self):
        reply = curate.read(batch("iterable"), "1. iterable = khả lặp")
        assert (reply.accepted[0].en, reply.accepted[0].vi) == ("iterable", "khả lặp")

    def test_a_kept_english_answer_becomes_a_keep_en_row(self):
        reply = curate.read(batch("decorator"), f"1. decorator = {KEEP}")
        row = reply.accepted[0]
        assert row.keep_en and row.vi == "decorator"

    def test_several_lines_become_several_rows(self):
        reply = curate.read(
            batch("iterable", "decorator"),
            answered("1. iterable = khả lặp", f"2. decorator = {KEEP}"),
        )
        assert len(reply.accepted) == 2 and not reply.dropped

    def test_whitespace_around_the_separator_is_not_significant(self):
        reply = curate.read(batch("iterable"), "1.    iterable   =   khả lặp   ")
        assert reply.accepted[0].vi == "khả lặp"

    def test_a_multi_word_rendering_is_accepted(self):
        reply = curate.read(batch("context manager"), "1. context manager = trình quản lý ngữ cảnh")
        assert reply.accepted[0].vi == "trình quản lý ngữ cảnh"

    def test_the_batch_is_named_on_the_reply(self):
        one = batch("iterable")
        assert curate.read(one, "1. iterable = khả lặp").batch == one.id


class TestDeclining:
    def test_a_model_that_does_not_know_is_recorded_rather_than_dropped(self):
        reply = curate.read(batch("evaluate function"), f"1. evaluate function = {UNSURE}")
        assert reply.declined == ("evaluate function",) and not reply.dropped

    def test_a_decline_produces_no_row(self):
        reply = curate.read(batch("evaluate function"), f"1. evaluate function = {UNSURE}")
        assert reply.accepted == ()

    def test_declining_counts_as_having_answered(self):
        reply = curate.read(batch("evaluate function"), f"1. evaluate function = {UNSURE}")
        assert reply.answered == 1


class TestRepeatedEnglish:
    def test_an_answer_about_the_wrong_term_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. iterator = trình lặp")
        assert [problem.rule for problem in reply.dropped] == ["G-a"]

    def test_the_rejection_shows_what_the_model_said_instead(self):
        reply = curate.read(batch("iterable"), "1. iterator = trình lặp")
        assert "'iterator'" in reply.dropped[0].detail

    def test_a_line_with_no_separator_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. khả lặp")
        assert [problem.rule for problem in reply.dropped] == ["G-a"]

    def test_an_off_by_one_in_the_numbering_drops_every_line_it_shifted(self):
        """The failure this rule exists for. Without it every rendering attaches
        to the term above it and every row in the result is a real phrase."""
        reply = curate.read(
            batch("iterable", "decorator", "generator"),
            answered(
                "1. decorator = trình trang trí",
                "2. generator = bộ sinh",
                "3. iterable = khả lặp",
            ),
        )
        assert reply.accepted == () and len(reply.dropped) == 3

    def test_the_english_is_compared_exactly_rather_than_loosely(self):
        reply = curate.read(batch("context manager"), "1. Context Manager = trình quản lý")
        assert [problem.rule for problem in reply.dropped] == ["G-a"]


class TestOnePhrase:
    def test_an_answer_ending_a_sentence_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. iterable = khả lặp.")
        assert [problem.rule for problem in reply.dropped] == ["G-b"]

    def test_an_answer_over_the_word_cap_is_dropped(self):
        long = " ".join(["từ"] * (PHRASE_WORDS + 1))
        reply = curate.read(batch("iterable"), f"1. iterable = {long}")
        assert [problem.rule for problem in reply.dropped] == ["G-b"]

    def test_the_rejection_says_how_many_words_there_were(self):
        long = " ".join(["từ"] * (PHRASE_WORDS + 1))
        reply = curate.read(batch("iterable"), f"1. iterable = {long}")
        assert f"is {PHRASE_WORDS + 1} words" in reply.dropped[0].detail

    def test_an_answer_at_the_word_cap_is_accepted(self):
        exact = " ".join(["từ"] * PHRASE_WORDS)
        reply = curate.read(batch("iterable"), f"1. iterable = {exact}")
        assert reply.accepted[0].vi == exact

    def test_an_answer_running_over_lines_is_dropped(self):
        reply = curate.read(
            batch("iterable", "decorator"),
            answered("1. iterable = khả lặp", "   và giải thích thêm", f"2. decorator = {KEEP}"),
        )
        assert [problem.rule for problem in reply.dropped] == ["G-b"]

    def test_an_empty_answer_after_the_separator_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. iterable =")
        assert [problem.rule for problem in reply.dropped] == ["G-b"]

    def test_a_trailing_comma_is_sentence_punctuation(self):
        reply = curate.read(batch("iterable"), "1. iterable = khả lặp,")
        assert [problem.rule for problem in reply.dropped] == ["G-b"]


class TestWrittenInVietnamese:
    def test_an_answer_with_no_tone_mark_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. iterable = kha lap")
        assert [problem.rule for problem in reply.dropped] == ["G-c"]

    def test_the_english_word_returned_as_a_rendering_is_dropped(self):
        reply = curate.read(batch("iterable"), "1. iterable = iterable")
        assert [problem.rule for problem in reply.dropped] == ["G-c"]

    def test_the_keep_answer_is_the_way_to_say_that_legitimately(self):
        reply = curate.read(batch("iterable"), f"1. iterable = {KEEP}")
        assert reply.accepted[0].keep_en and not reply.dropped

    def test_the_letter_d_with_a_stroke_counts_as_vietnamese(self):
        reply = curate.read(batch("path"), "1. path = đường dẫn")
        assert reply.accepted[0].vi == "đường dẫn"

    def test_the_rejection_points_at_the_keep_answer(self):
        reply = curate.read(batch("iterable"), "1. iterable = kha lap")
        assert "keep-English" in reply.dropped[0].detail


class TestRearrangement:
    def test_the_english_with_the_words_swapped_is_dropped(self):
        reply = curate.read(batch("context manager"), "1. context manager = mánager cóntext")
        assert [problem.rule for problem in reply.dropped] == ["G-d"]

    def test_adding_tone_marks_to_the_english_does_not_make_it_vietnamese(self):
        reply = curate.read(batch("decorator"), "1. decorator = décorator")
        assert [problem.rule for problem in reply.dropped] == ["G-d"]

    def test_a_real_rendering_is_not_a_rearrangement(self):
        reply = curate.read(batch("context manager"), "1. context manager = trình quản lý ngữ cảnh")
        assert not reply.dropped


class TestMalformedAnswers:
    def test_a_missing_answer_is_reported_against_its_term(self):
        reply = curate.read(batch("iterable", "decorator"), "1. iterable = khả lặp")
        assert [problem.en for problem in reply.dropped] == ["decorator"]

    def test_a_missing_answer_is_a_format_problem_rather_than_a_rule_break(self):
        reply = curate.read(batch("iterable", "decorator"), "1. iterable = khả lặp")
        assert [problem.rule for problem in reply.dropped] == ["format"]

    def test_an_answer_with_no_numbering_at_all_is_reported_once(self):
        reply = curate.read(batch("iterable"), "khả lặp")
        assert len(reply.dropped) == 1 and reply.accepted == ()

    def test_an_index_the_batch_never_had_is_reported_and_nothing_it_touched_is_kept(self):
        """The parser reads a line that does not continue the sequence as text,
        so the stray lands in the body above it and ``G-b`` refuses a rendering
        that runs over lines. Reported twice and accepted never."""
        reply = curate.read(batch("iterable"), answered("1. iterable = khả lặp", "9. x = y"))
        assert reply.accepted == ()
        assert any("9" in problem.detail for problem in reply.dropped)
        assert any(problem.rule == "G-b" for problem in reply.dropped)

    def test_a_repeated_index_costs_the_term_it_repeats_and_no_other(self):
        reply = curate.read(
            batch("iterable", "decorator"),
            answered("1. iterable = khả lặp", "1. iterable = lặp được", f"2. decorator = {KEEP}"),
        )
        assert [term.en for term in reply.accepted] == ["decorator"]
        assert any(problem.rule == "G-b" for problem in reply.dropped)

    def test_narration_before_the_first_entry_is_reported(self):
        reply = curate.read(
            batch("iterable"), answered("Here are the terms:", "1. iterable = khả lặp")
        )
        assert any("preamble" in problem.detail for problem in reply.dropped)

    def test_one_bad_line_costs_only_itself(self):
        reply = curate.read(
            batch("iterable", "decorator"),
            answered("1. iterable = khả lặp.", f"2. decorator = {KEEP}"),
        )
        assert len(reply.accepted) == 1 and len(reply.dropped) == 1


class TestCollecting:
    def test_the_replies_add_up(self):
        one = curate.read(batch("iterable"), "1. iterable = khả lặp")
        two = curate.read(batch("decorator"), f"1. decorator = {KEEP}")
        found = curate.collect([one, two])
        assert len(found.accepted) == 2 and found.batches == 2

    def test_a_term_answered_twice_keeps_the_first_answer(self):
        one = curate.read(batch("iterable"), "1. iterable = khả lặp")
        two = curate.read(batch("iterable"), "1. iterable = lặp được")
        assert curate.collect([one, two]).accepted[0].vi == "khả lặp"

    def test_the_declines_are_kept_apart_from_the_failures(self):
        one = curate.read(batch("evaluate function"), f"1. evaluate function = {UNSURE}")
        two = curate.read(batch("iterable"), "1. iterable = kha lap")
        found = curate.collect([one, two])
        assert len(found.declined) == 1 and len(found.dropped) == 1

    def test_the_failures_are_counted_by_rule_so_a_report_can_name_them(self):
        one = curate.read(batch("iterable"), "1. iterable = kha lap")
        two = curate.read(batch("decorator"), "1. decorator = trình trang trí.")
        assert curate.collect([one, two]).by_rule == {"G-b": 1, "G-c": 1}

    def test_everything_asked_about_is_accounted_for(self):
        one = curate.read(
            batch("iterable", "decorator", "generator"),
            answered(
                "1. iterable = khả lặp",
                f"2. decorator = {UNSURE}",
                "3. generator = kha lap",
            ),
        )
        assert curate.collect([one]).asked == 3

    def test_the_kept_rows_are_counted_separately(self):
        one = curate.read(
            batch("decorator", "iterable"),
            answered(f"1. decorator = {KEEP}", "2. iterable = khả lặp"),
        )
        assert curate.collect([one]).kept == 1

    def test_collecting_nothing_is_an_empty_outcome(self):
        found = curate.collect([])
        assert found.asked == 0 and found.batches == 0 and found.by_rule == {}


class TestGlossaryFromCuration:
    """The rows a run produces have to pass the list rules before they land."""

    def test_the_accepted_rows_go_into_a_glossary_in_match_order(self):
        from pydocvi import glossary

        reply = curate.read(
            batch("context", "context manager"),
            answered("1. context = ngữ cảnh", "2. context manager = trình quản lý ngữ cảnh"),
        )
        rows = glossary.Glossary(version=0).with_terms(
            glossary.match_order(curate.collect([reply]).accepted)
        )
        assert glossary.check(rows) == []

    def test_a_run_that_produced_a_collision_is_caught_before_the_file_is_written(self):
        from pydocvi import glossary

        reply = curate.read(
            batch("bug", "mistake"),
            answered("1. bug = lỗi", "2. mistake = lỗi"),
        )
        rows = glossary.Glossary(version=0).with_terms(curate.collect([reply]).accepted)
        assert [problem.rule for problem in glossary.check(rows)] == ["G-e", "G-e"]


@pytest.mark.parametrize(
    ("answer", "rule"),
    [
        ("1. iterable = khả lặp.", "G-b"),
        ("1. iterable = kha lap", "G-c"),
        ("1. iterator = trình lặp", "G-a"),
        ("1. iterable = ítérable", "G-d"),
    ],
)
def test_each_rule_catches_its_own_failure(answer, rule):
    """One table, four failures, so a rule that stops firing is visible."""
    assert [problem.rule for problem in curate.read(batch("iterable"), answer).dropped] == [rule]


class TestReport:
    def outcome(self):
        one = curate.read(
            batch("iterable", "decorator", "generator", "coroutine"),
            answered(
                "1. iterable = khả lặp",
                f"2. decorator = {KEEP}",
                f"3. generator = {UNSURE}",
                "4. coroutine = kha lap",
            ),
        )
        return curate.collect([one])

    def test_the_counts_are_in_the_table(self):
        text = curate.report(self.outcome(), prompt="abc123")
        assert "| accepted | 2 |" in text and "| declined | 1 |" in text

    def test_the_prompt_that_asked_is_named(self):
        assert "`abc123`" in curate.report(self.outcome(), prompt="abc123")

    def test_the_declined_terms_are_listed_by_name_because_they_are_a_work_list(self):
        assert "- generator" in curate.report(self.outcome(), prompt="abc123")

    def test_the_rules_that_dropped_something_are_broken_out(self):
        assert "| `G-c` | 1 |" in curate.report(self.outcome(), prompt="abc123")

    def test_a_run_that_dropped_nothing_has_no_rule_table(self):
        one = curate.read(batch("iterable"), "1. iterable = khả lặp")
        assert "Dropped by rule" not in curate.report(curate.collect([one]), prompt="abc")

    def test_a_run_that_declined_nothing_has_no_decline_list(self):
        one = curate.read(batch("iterable"), "1. iterable = khả lặp")
        assert "## Declined" not in curate.report(curate.collect([one]), prompt="abc")


class Answering:
    """A client that answers the terms in a prompt and yields between calls.

    Its own class rather than the shared fake because the shared one never
    awaits, so a single consumer drains the whole queue before the event loop
    ever gets a chance to start the second. That would make the sharing test
    pass or fail on a detail of the fake rather than on the dispatch.
    """

    def __init__(self, answer=None):
        self.answer = answer or renderings
        self.calls: list[tuple[str, str]] = []

    async def complete(self, route, prompt, *, system=None):
        self.calls.append((route.name, prompt))
        await asyncio.sleep(0)
        reply = self.answer(prompt)
        if isinstance(reply, BaseException):
            raise reply
        return Answer(text=reply, route=route.name, model=route.model, seconds=1.0)


def renderings(prompt: str) -> str:
    """Answer every term in the prompt's term block, and nothing else in it.

    The block matters. The instructions above it are a numbered list too, and a
    fake that answered those would be answering the prompt's own prose.
    """
    _, _, block = prompt.partition("Terms:\n")
    lines = []
    for line in block.splitlines():
        found = re.match(r"^(\d+)\. (\S.*)$", line)
        if found:
            lines.append(f"{found.group(1)}. {found.group(2)} = khả lặp")
    return "\n".join(lines)


class TestAsking:
    def one_route(self, **overrides):
        return [make_route(**overrides)]

    def test_every_batch_is_sent(self):
        client = Answering()
        made = curate.batches([candidate(f"term {n}") for n in range(5)], size=2)
        asyncio.run(curate.ask(client, self.one_route(), made))
        assert len(client.calls) == 3

    def test_the_replies_come_back_in_batch_order(self):
        made = curate.batches([candidate(f"term {n}") for n in range(5)], size=2)
        replies = asyncio.run(curate.ask(Answering(), self.one_route(), made))
        assert [reply.index for reply in replies] == [1, 2, 3]

    def test_every_term_asked_about_is_answered(self):
        made = curate.batches([candidate(f"term {n}") for n in range(9)], size=4)
        replies = asyncio.run(curate.ask(Answering(), self.one_route(concurrency=3), made))
        assert curate.collect(replies).asked == 9

    def test_work_is_shared_across_routes(self):
        client = Answering()
        made = curate.batches([candidate(f"term {n}") for n in range(6)], size=1)
        pair = [make_route("a", concurrency=1), make_route("b", concurrency=1)]
        asyncio.run(curate.ask(client, pair, made))
        assert {name for name, _ in client.calls} == {"a", "b"}

    def test_a_call_that_raises_costs_only_its_own_batch(self):
        """A curation run is a hundred calls. One host dropping a connection is
        not a reason to lose the ninety-nine answers that came back."""

        def sometimes(prompt: str):
            if "term 0" in prompt:
                return RuntimeError("connection reset")
            return renderings(prompt)

        made = curate.batches([candidate(f"term {n}") for n in range(4)], size=1)
        replies = asyncio.run(curate.ask(Answering(sometimes), self.one_route(), made))
        outcome = curate.collect(replies)
        assert len(outcome.accepted) == 3 and outcome.by_rule == {"call": 1}

    def test_a_failed_call_names_the_terms_it_cost(self):
        made = curate.batches([candidate("iterable"), candidate("decorator")], size=2)
        client = Answering(lambda _: RuntimeError("boom"))
        replies = asyncio.run(curate.ask(client, self.one_route(), made))
        assert [problem.en for problem in replies[0].dropped] == ["iterable", "decorator"]

    def test_an_empty_answer_is_a_failure_rather_than_a_silent_zero(self):
        made = curate.batches([candidate("iterable")], size=1)
        replies = asyncio.run(curate.ask(Answering(lambda _: "   "), self.one_route(), made))
        assert [problem.rule for problem in replies[0].dropped] == ["call"]

    def test_progress_is_reported_as_each_reply_lands(self):
        seen: list[curate.Reply] = []
        made = curate.batches([candidate(f"term {n}") for n in range(4)], size=1)
        asyncio.run(curate.ask(Answering(), self.one_route(), made, on_reply=seen.append))
        assert len(seen) == 4

    def test_curating_with_no_routes_is_an_error_rather_than_a_silent_no_op(self):
        with pytest.raises(ValueError, match="no routes"):
            asyncio.run(curate.ask(Answering(), [], curate.batches([candidate("x")])))

    def test_nothing_to_ask_makes_no_calls(self):
        client = Answering()
        asyncio.run(curate.ask(client, self.one_route(), []))
        assert client.calls == []


class Refusing:
    """A client where the named routes will not answer and the rest will."""

    def __init__(self, *refuse: str):
        self.refuse = set(refuse)
        self.calls: list[str] = []

    async def complete(self, route, prompt, *, system=None):
        self.calls.append(route.name)
        await asyncio.sleep(0)
        if route.name in self.refuse:
            raise RuntimeError(f"{route.name}: connection reset")
        return Answer(text=renderings(prompt), route=route.name, model=route.model, seconds=1.0)


class TestMovingABatchToAnotherRoute:
    """A host that has stopped answering should not cost the batches it holds.

    Found on the first full run over the real fleet rather than here. One host
    closed the connection on every attempt, the client's three retries all went
    back to that same host because staying on one route is what it is for, and
    80 terms were dropped under ``call`` while two working hosts sat idle.
    """

    def pair(self):
        return [make_route("a", concurrency=1), make_route("b", concurrency=1)]

    def test_a_batch_its_own_route_refuses_is_answered_by_another(self):
        client = Refusing("a")
        made = curate.batches([candidate("iterable")], size=1)
        replies = asyncio.run(curate.ask(client, self.pair(), made))
        assert client.calls == ["a", "b"]
        assert curate.collect(replies).by_rule == {}

    def test_the_terms_survive_the_move(self):
        client = Refusing("a")
        made = curate.batches([candidate("iterable")], size=1)
        replies = asyncio.run(curate.ask(client, self.pair(), made))
        assert [term.en for term in curate.collect(replies).accepted] == ["iterable"]

    def test_it_is_given_up_on_only_once_every_route_has_refused(self):
        client = Refusing("a", "b")
        made = curate.batches([candidate("iterable")], size=1)
        replies = asyncio.run(curate.ask(client, self.pair(), made))
        assert client.calls == ["a", "b"]
        assert curate.collect(replies).by_rule == {"call": 1}

    def test_an_answer_with_a_dropped_row_is_kept_rather_than_moved(self):
        """The second host would drop the same row for the same reason.

        Moving it would spend another call to arrive at the same answer, and on
        a fleet where one host takes four minutes a call that is not free.
        """
        client = Answering(lambda _: "1. iterable = kha lap")
        made = curate.batches([candidate("iterable")], size=1)
        replies = asyncio.run(curate.ask(client, self.pair(), made))
        assert len(client.calls) == 1
        assert curate.collect(replies).by_rule == {"G-c": 1}


class TestWhetherAReplyWasAnswered:
    def test_a_failed_call_counts_as_unanswered(self):
        assert curate.Reply.failed(batch("iterable"), "connection reset").unanswered

    def test_a_row_dropped_by_a_rule_does_not(self):
        assert not curate.read(batch("iterable"), "1. iterable = kha lap").unanswered

    def test_nor_does_a_clean_reply(self):
        assert not curate.read(batch("iterable"), "1. iterable = khả lặp").unanswered

    def test_nor_does_a_decline(self):
        assert not curate.read(batch("iterable"), f"1. iterable = {UNSURE}").unanswered
