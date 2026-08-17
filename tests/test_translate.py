from collections.abc import Sequence
from pathlib import Path

import pytest

from pydocvi import batch, render, translate
from pydocvi.catalog import Entry
from pydocvi.client import Answer, Usage
from pydocvi.glossary import Glossary
from pydocvi.memory import Memory, Segment
from pydocvi.queue import Job, Queue, Stage, State
from pydocvi.translate import Attempt


def make(*msgids: str, path: str = "f.po") -> batch.Batch:
    return next(iter(batch.pack(path, batch.items([Entry(msgid=m) for m in msgids]))))


def answered(*bodies: str) -> str:
    return "\n".join(f"{number} {body}" for number, body in enumerate(bodies, 1))


def reply(text: str, *, served: str = "gpt-5-6") -> Answer:
    """One completion, with the fields the stage reads and nothing else."""
    return Answer(
        text=text,
        route="server3",
        model="gpt-5",
        seconds=90.0,
        usage=Usage(prompt_tokens=100, completion_tokens=50),
        served=served,
    )


def stored(item: batch.Item, msgstr: str = "Câu đầu tiên.") -> Segment:
    """One entry already in the memory, translated by an earlier run."""
    return Segment(id=item.segment, msgid=item.msgid, msgstr=msgstr, source="machine")


def make_run(tmp_path: Path, batches: Sequence[batch.Batch], **overrides: object) -> translate.Run:
    """A stage over a real queue and a real memory, both under ``tmp_path``."""
    values: dict[str, object] = {
        "queue": Queue(tmp_path / "queue", Stage.TRANSLATE),
        "memory": Memory(path=tmp_path / "tm.jsonl"),
        "glossary": Glossary(version=7),
        "batches": batches,
        "run": "2026-01-01T00:00Z",
    }
    values.update(overrides)
    return translate.Run(**values)  # type: ignore[arg-type]


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


class TestTheRunPlan:
    def test_a_plan_queues_a_job_per_batch(self, tmp_path: Path) -> None:
        batches = [make("First one here."), make("Second one here.", path="g.po")]
        run = make_run(tmp_path, batches)
        plan = run.plan(batches, tier=1)
        assert (plan.batches, plan.queued, plan.known) == (2, 2, 0)
        assert len(run.queue) == 2

    def test_a_second_pass_queues_only_what_the_first_did_not_finish(self, tmp_path: Path) -> None:
        """The whole value of a content-addressed queue, and the reason a
        re-run of a tier costs almost nothing."""
        batches = [make("First one here."), make("Second one here.", path="g.po")]
        run = make_run(tmp_path, batches)
        run.plan(batches)
        again = run.plan(batches)
        assert (again.queued, again.known) == (0, 2)

    def test_a_dry_plan_counts_what_it_would_queue_and_writes_nothing(self, tmp_path: Path) -> None:
        batches = [make("First one here.")]
        run = make_run(tmp_path, batches)
        plan = run.plan(batches, write=False)
        assert plan.queued == 1
        assert len(run.queue) == 0

    def test_a_plan_says_where_it_is_and_what_it_holds(self, tmp_path: Path) -> None:
        batches = [make("First one here.", "Second one here.")]
        run = make_run(tmp_path, batches)
        assert str(run.plan(batches, tier=3)) == "tier 3: 1 batches, 2 entries, 1 queued"

    def test_a_plan_without_a_tier_says_the_selection(self, tmp_path: Path) -> None:
        assert "the selection" in str(make_run(tmp_path, []).plan([]))

    def test_a_plan_says_out_loud_what_was_already_known(self, tmp_path: Path) -> None:
        batches = [make("First one here.")]
        run = make_run(tmp_path, batches)
        run.plan(batches)
        assert "1 already known" in str(run.plan(batches))

    def test_a_job_is_addressed_on_the_prompt_and_the_glossary_too(self, tmp_path: Path) -> None:
        """Not only on the entries. A prompt edit or a terminology bump makes
        every job in the corpus a different job, which is what stops a corpus
        being half one prompt and half another with nothing to say which."""
        batches = [make("First one here.")]
        run = make_run(tmp_path, batches)
        run.plan(batches)
        payload = run.queue.jobs(State.PENDING)[0].payload
        assert payload["prompt"] == render.fingerprint()
        assert payload["glossary"] == 7

    def test_a_glossary_bump_requeues_the_corpus(self, tmp_path: Path) -> None:
        batches = [make("First one here.")]
        make_run(tmp_path, batches).plan(batches)
        bumped = make_run(tmp_path, batches, glossary=Glossary(version=8))
        assert bumped.plan(batches).queued == 1


class TestWhatIsWorthCalling:
    def test_a_batch_the_memory_already_holds_is_not_worth_a_call(self, tmp_path: Path) -> None:
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        run.memory.add(stored(one.items[0]))
        assert run.untranslated([one]) == []

    def test_one_missing_entry_is_enough_to_keep_the_whole_batch(self, tmp_path: Path) -> None:
        """Whole batches rather than repacked leftovers, because repacking would
        change every batch id and orphan every trace filed under the old one."""
        one = make("First one here.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.memory.add(stored(one.items[0]))
        assert run.untranslated([one]) == [one]


@pytest.mark.asyncio(loop_scope="function")
class TestHandlingAnAnswer:
    async def test_what_passed_goes_into_the_memory(self, tmp_path: Path) -> None:
        one = make("First one here.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply(answered("Câu đầu tiên.", "Câu thứ hai.")))
        assert len(run.memory) == 2
        assert run.memory.get(one.items[0].segment).msgstr == "Câu đầu tiên."

    async def test_a_stored_segment_records_the_model_that_actually_answered(
        self, tmp_path: Path
    ) -> None:
        """Not the model the route file asked for. This proxy serves gpt-5-6
        whatever is requested, and 85 000 provenance comments naming gpt-5 would
        be a record of something that did not happen."""
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply(answered("Câu đầu tiên."), served="gpt-5-6"))
        assert run.memory.get(one.items[0].segment).model == "gpt-5-6"

    async def test_a_stored_segment_carries_the_batch_the_glossary_and_the_run(
        self, tmp_path: Path
    ) -> None:
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply(answered("Câu đầu tiên.")))
        segment = run.memory.get(one.items[0].segment)
        assert (segment.batch, segment.glossary, segment.run) == (
            one.id,
            7,
            "2026-01-01T00:00Z",
        )

    async def test_an_entry_already_translated_by_a_person_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """Precedence lives in the memory, and this is the test that says the
        stage goes through it rather than around it."""
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        run.memory.add(
            Segment(
                id=one.items[0].segment,
                msgid="First one here.",
                msgstr="Bản dịch của người.",
                source="human",
            )
        )
        await run.handle(_job(run), reply(answered("Câu máy dịch.")))
        assert run.memory.get(one.items[0].segment).msgstr == "Bản dịch của người."

    async def test_only_the_refused_entry_comes_back_on_the_next_rung(self, tmp_path: Path) -> None:
        one = make("Return :func:`len` of it.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        job = _job(run)
        await run.handle(job, reply(answered("Trả về của nó.", "Câu thứ hai.")))
        queued = _next(run, job)
        assert len(queued) == 1
        assert queued[0].payload["attempt"] == Attempt.NAMED
        assert queued[0].payload["segments"] == [one.items[0].segment]

    async def test_the_retry_names_the_rule_to_the_model(self, tmp_path: Path) -> None:
        """A generic "try again" is worth nothing against a session that
        returns the same answer to the same prompt."""
        one = make("Return :func:`len` of it.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        job = _job(run)
        await run.handle(job, reply(answered("Trả về của nó.", "Câu thứ hai.")))
        assert "⟦n⟧" in str(_next(run, job)[0].payload["advice"])

    async def test_a_batch_refused_whole_retries_every_entry_in_it(self, tmp_path: Path) -> None:
        """P03, P06 and P07 produce no per-entry refusals at all, so a ladder
        driven by the refusal list would retry nothing and lose the batch."""
        one = make("First one here.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        job = _job(run)
        await run.handle(job, reply("1 Câu đầu tiên."))
        queued = _next(run, job)
        assert len(queued[0].payload["segments"]) == 2
        assert str(queued[0].payload["advice"]).startswith("Some strings in the last attempt")

    async def test_an_entry_out_of_rungs_is_buried_rather_than_released(
        self, tmp_path: Path
    ) -> None:
        """Two attempt counters run side by side. The queue's counts claims and
        a claim dies to a dropped tunnel. This batch has plenty of claims left
        and nothing left to try, which release has no way to say."""
        one = make("Return :func:`len` of it.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        job = _job(run)
        for rung in (Attempt.FIRST, Attempt.NAMED, Attempt.ALONE):
            job.payload["attempt"] = int(rung)
            await run.handle(job, reply(answered("Trả về của nó.")))
        assert run.queue.count(State.DEAD) == 1
        assert run.tally.dead == 1

    async def test_a_rung_the_queue_already_climbed_is_a_spent_rung(self, tmp_path: Path) -> None:
        """What a second run over the same tier actually does.

        Job ids are content-addressed on the rung and the advice as well as the
        entries, so the second run rebuilds the identical ladder and ``add``
        refuses every step of it as already known. That is the property that
        makes a second pass cheap, and it meant the second tier 1 run made
        three calls, reported no entries out of rungs, and left two entries with
        neither a translation nor a reason.
        """
        one = make("Return :func:`len` of it.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        job = _job(run)
        await run.handle(job, reply(answered("Trả về của nó.")))
        climbed = _next(run, job)[0]
        run.queue.finish(run.queue.claim(now=0.0) or climbed)
        for pending in run.queue.jobs(State.PENDING):
            run.queue.finish(pending)

        again = make_run(tmp_path, [one], queue=run.queue, memory=run.memory)
        await again.handle(job, reply(answered("Trả về của nó.")))
        assert again.queue.count(State.DEAD) == 1
        assert again.tally.dead == 1
        assert "already climbed" in (again.queue.jobs(State.DEAD)[0].error or "")

    async def test_a_run_counts_refusals_by_rule_and_by_rung(self, tmp_path: Path) -> None:
        """Counted as the run goes, because a refusal fixed on rung 2 leaves no
        trace in the corpus and it is the number that says the ladder pays."""
        one = make("Return :func:`len` of it.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply(answered("Trả về của nó.", "Câu thứ hai.")))
        assert run.tally.by_rule["P01"] == 1
        assert run.tally.by_attempt[int(Attempt.FIRST)] == 2
        assert (run.tally.accepted, run.tally.refused, run.tally.batches) == (1, 1, 1)

    async def test_the_rate_is_the_share_of_entries_that_came_back_usable(
        self, tmp_path: Path
    ) -> None:
        one = make("Return :func:`len` of it.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply(answered("Trả về của nó.", "Câu thứ hai.")))
        assert run.tally.rate == 0.5

    async def test_an_empty_tally_has_a_rate_rather_than_a_division_by_zero(self) -> None:
        assert translate.Tally().rate == 0.0

    async def test_a_batch_refused_whole_is_counted_as_one_rejection(self, tmp_path: Path) -> None:
        one = make("First one here.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        await run.handle(_job(run), reply("1 Câu đầu tiên."))
        assert (run.tally.rejected, run.tally.refused) == (1, 0)
        assert run.tally.by_rule["P03"] == 1


class TestBuildingThePrompt:
    def test_a_job_becomes_the_two_messages_of_its_batch(self, tmp_path: Path) -> None:
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        prompt = run.build(_job(run))
        assert "First one here." in prompt.user
        assert prompt.system

    def test_the_advice_from_the_last_rung_reaches_the_prompt(self, tmp_path: Path) -> None:
        one = make("First one here.")
        run = make_run(tmp_path, [one])
        job = run._job(one, attempt=Attempt.NAMED, advice="Some strings came back wrong.")
        assert "came back wrong" in run.build(job).user


class TestRebuildingABatch:
    def test_a_batch_is_recovered_from_the_corpus_and_not_from_the_job(
        self, tmp_path: Path
    ) -> None:
        """The payload carries ids and never the entries, which is what makes a
        run resumable across a restart without a serialised copy going stale."""
        one = make("First one here.", "Second one here.")
        run = make_run(tmp_path, [one])
        run.plan([one])
        rebuilt = run.batch(_job(run))
        assert rebuilt.id == one.id
        assert [item.msgid for item in rebuilt.items] == [item.msgid for item in one.items]

    def test_a_job_with_no_segment_list_is_named_rather_than_crashed_on(
        self, tmp_path: Path
    ) -> None:
        """A job file is JSON on a filesystem a person can edit."""
        run = make_run(tmp_path, [])
        job = Job(id="x", stage=Stage.TRANSLATE, payload={"file": "f.po"})
        with pytest.raises(translate.JobError, match="no list of segments"):
            run.batch(job)

    def test_a_job_on_a_rung_this_ladder_does_not_have_is_named(self) -> None:
        job = Job(id="x", stage=Stage.TRANSLATE, payload={"file": "f.po", "attempt": 9})
        with pytest.raises(translate.JobError, match="no rung"):
            translate._rung(job)


class TestSaving:
    def test_the_memory_is_written_every_so_often_rather_than_at_the_end(
        self, tmp_path: Path
    ) -> None:
        """Nine hours of calls held in a dictionary is nine hours a Ctrl-C can
        throw away."""
        run = make_run(tmp_path, [], save_every=2)
        run.save()
        assert not (tmp_path / "tm.jsonl").exists()
        run.save()
        assert (tmp_path / "tm.jsonl").exists()

    def test_the_end_of_a_run_writes_whatever_the_interval_says(self, tmp_path: Path) -> None:
        run = make_run(tmp_path, [], save_every=1000)
        run.save(force=True)
        assert (tmp_path / "tm.jsonl").exists()


def _job(run: translate.Run) -> Job:
    """The one job a single-batch plan queued."""
    return run.queue.jobs(State.PENDING)[0]


def _next(run: translate.Run, job: Job) -> list[Job]:
    """What the ladder queued after handling ``job``.

    The handled job itself is still pending, because finishing a job is the
    worker's business and this stage is only asked what to do next.
    """
    return [one for one in run.queue.jobs(State.PENDING) if one.id != job.id]
