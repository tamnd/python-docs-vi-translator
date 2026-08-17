from pathlib import Path

import pytest

from pydocvi import queue as queuemod
from pydocvi.queue import LEASE_SECONDS, MAX_ATTEMPTS, Job, Queue, Stage, State, job_id


@pytest.fixture
def queue(tmp_path: Path) -> Queue:
    return Queue(tmp_path / "queue", Stage.TRANSLATE)


def make_job(identifier: str = "a", **overrides: object) -> Job:
    values: dict[str, object] = {
        "id": identifier,
        "stage": Stage.TRANSLATE,
        "payload": {"file": "bugs.po", "entries": [1, 2]},
    }
    values.update(overrides)
    return Job(**values)  # type: ignore[arg-type]


def fill(queue: Queue, count: int) -> None:
    queue.extend([make_job(f"job{index:03d}") for index in range(count)])


class TestJobId:
    def test_the_same_work_gets_the_same_id(self) -> None:
        """This is the whole resume mechanism. Same inputs, same id, and the
        output already on disk means the work is skipped."""
        payload = {"file": "bugs.po", "entries": [1, 2, 3]}
        assert job_id(Stage.TRANSLATE, payload) == job_id(Stage.TRANSLATE, payload)

    def test_the_order_keys_were_written_in_does_not_matter(self) -> None:
        first = job_id(Stage.TRANSLATE, {"file": "bugs.po", "prompt": "v3"})
        second = job_id(Stage.TRANSLATE, {"prompt": "v3", "file": "bugs.po"})
        assert first == second

    def test_a_different_stage_is_a_different_job(self) -> None:
        """Otherwise translating a batch would mark judging it as already done."""
        payload = {"file": "bugs.po"}
        assert job_id(Stage.TRANSLATE, payload) != job_id(Stage.JUDGE, payload)

    def test_changing_the_prompt_changes_every_id(self) -> None:
        """Which is how a prompt revision re-queues the corpus instead of being
        silently ignored because the outputs are already there."""
        old = job_id(Stage.TRANSLATE, {"file": "bugs.po", "prompt": "v3"})
        new = job_id(Stage.TRANSLATE, {"file": "bugs.po", "prompt": "v4"})
        assert old != new

    def test_the_id_is_short_enough_to_read_out_loud(self) -> None:
        assert len(job_id(Stage.TRANSLATE, {"file": "bugs.po"})) == queuemod.ID_LENGTH

    def test_a_payload_holding_something_unjsonable(self) -> None:
        assert job_id(Stage.TRANSLATE, {"path": Path("bugs.po")})


class TestAdding:
    def test_a_new_job_lands_in_pending(self, queue: Queue) -> None:
        assert queue.add(make_job())
        assert queue.count(State.PENDING) == 1

    def test_the_same_job_twice_is_added_once(self, queue: Queue) -> None:
        queue.add(make_job())
        assert not queue.add(make_job())
        assert queue.count(State.PENDING) == 1

    @pytest.mark.parametrize("state", list(State))
    def test_a_job_already_known_in_any_state_is_not_re_added(
        self, queue: Queue, state: State
    ) -> None:
        """Including done, which is what makes a second pass cheap: it queues
        only the batches the first pass did not finish."""
        queuemod._write(queue.path(state, "a"), make_job())
        assert not queue.add(make_job())

    def test_extend_reports_how_many_were_new(self, queue: Queue) -> None:
        queue.add(make_job("a"))
        assert queue.extend([make_job("a"), make_job("b"), make_job("c")]) == 2


class TestClaiming:
    def test_claiming_moves_the_job_and_stamps_a_deadline(self, queue: Queue) -> None:
        queue.add(make_job())
        claimed = queue.claim(now=1_000.0)
        assert claimed is not None
        assert claimed.lease_expires == 1_000.0 + LEASE_SECONDS
        assert queue.count(State.PENDING) == 0
        assert queue.count(State.LEASED) == 1

    def test_claiming_spends_an_attempt(self, queue: Queue) -> None:
        """Attempts count claims rather than failures, so a worker killed mid
        call spends one. Three silent deaths is a job a person should look at."""
        queue.add(make_job())
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        assert claimed.attempts == 1

    def test_an_empty_queue_claims_nothing(self, queue: Queue) -> None:
        assert queue.claim(now=0.0) is None

    def test_two_claims_get_two_different_jobs(self, queue: Queue) -> None:
        fill(queue, 2)
        first = queue.claim(now=0.0)
        second = queue.claim(now=0.0)
        assert first is not None
        assert second is not None
        assert first.id != second.id

    def test_the_deadline_survives_being_written_to_disk(self, queue: Queue) -> None:
        """A different process reads this file after a Ctrl-C, so the number in
        it is the only thing that matters."""
        queue.add(make_job())
        queue.claim(now=1_000.0)
        on_disk = Job.from_json(queue.path(State.LEASED, "a").read_text(encoding="utf-8"))
        assert on_disk.lease_expires == 1_000.0 + LEASE_SECONDS
        assert on_disk.attempts == 1

    def test_a_job_file_that_is_not_readable_is_stepped_over(self, queue: Queue) -> None:
        queue.path(State.PENDING, "broken").parent.mkdir(parents=True, exist_ok=True)
        queue.path(State.PENDING, "broken").write_text("{not json", encoding="utf-8")
        queue.add(make_job("later"))
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        assert claimed.id == "later"


class TestRacing:
    def test_two_workers_reaching_for_the_same_job(
        self, queue: Queue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rename is the claim. The loser gets an error from replace and
        moves on to the next file rather than checking first and racing anyway."""
        fill(queue, 2)
        real = Path.replace
        lost = []

        def replace(self: Path, target: Path) -> Path:
            if not lost:
                lost.append(self.name)
                raise OSError("someone else got there first")
            return real(self, target)

        monkeypatch.setattr(Path, "replace", replace)
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        assert claimed.id == "job001"

    def test_a_queue_where_every_job_was_taken_by_someone_else(
        self, queue: Queue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fill(queue, 2)

        def replace(self: Path, target: Path) -> Path:
            raise OSError("someone else got there first")

        monkeypatch.setattr(Path, "replace", replace)
        assert queue.claim(now=0.0) is None


class TestFinishing:
    def test_a_finished_job_is_in_exactly_one_place(self, queue: Queue) -> None:
        queue.add(make_job())
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        queue.finish(claimed)
        assert [state for state, _ in [queue.locate("a")] if state] == [State.DONE]  # type: ignore[misc]
        assert queue.count(State.LEASED) == 0

    def test_a_failure_with_attempts_left_goes_back_to_pending(self, queue: Queue) -> None:
        queue.add(make_job())
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        assert queue.release(claimed, error="empty answer") is State.PENDING

    def test_the_error_is_kept_for_the_audit(self, queue: Queue) -> None:
        queue.add(make_job())
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        queue.release(claimed, error="empty answer")
        assert queue.jobs(State.PENDING)[0].error == "empty answer"

    def test_three_failures_and_the_job_is_dead(self, queue: Queue) -> None:
        """Three then dead is deliberate. On this transport a fourth attempt has
        never fixed anything and reading the trace has."""
        queue.add(make_job())
        states = []
        for _ in range(MAX_ATTEMPTS):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            states.append(queue.release(claimed, error="empty answer"))
        assert states == [State.PENDING, State.PENDING, State.DEAD]
        assert queue.count(State.PENDING) == 0

    def test_a_dead_job_is_not_claimed_again(self, queue: Queue) -> None:
        queue.add(make_job())
        for _ in range(MAX_ATTEMPTS):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            queue.release(claimed, error="down")
        assert queue.claim(now=0.0) is None


class TestLeaseExpiry:
    def test_an_expired_lease_comes_back(self, queue: Queue) -> None:
        queue.add(make_job())
        queue.claim(now=1_000.0)
        assert len(queue.reap(now=1_000.0 + LEASE_SECONDS + 1)) == 1
        assert queue.count(State.PENDING) == 1

    def test_a_lease_still_running_is_left_alone(self, queue: Queue) -> None:
        """Otherwise a reap during a run would hand a job that is mid call to a
        second worker and pay for it twice."""
        queue.add(make_job())
        queue.claim(now=1_000.0)
        assert queue.reap(now=1_100.0) == []
        assert queue.count(State.LEASED) == 1

    def test_reaping_says_why_the_job_came_back(self, queue: Queue) -> None:
        queue.add(make_job())
        queue.claim(now=0.0)
        queue.reap(now=LEASE_SECONDS + 1)
        assert queue.jobs(State.PENDING)[0].error == "lease expired"

    def test_an_expired_lease_on_its_last_attempt_dies(self, queue: Queue) -> None:
        queue.add(make_job())
        for _ in range(MAX_ATTEMPTS):
            queue.claim(now=0.0)
            queue.reap(now=LEASE_SECONDS + 1)
        assert queue.count(State.DEAD) == 1

    def test_reaping_an_empty_queue_is_quiet(self, queue: Queue) -> None:
        assert queue.reap(now=0.0) == []

    def test_a_leased_file_that_is_not_readable_is_stepped_over(self, queue: Queue) -> None:
        queue.path(State.LEASED, "broken").parent.mkdir(parents=True, exist_ok=True)
        queue.path(State.LEASED, "broken").write_text("{not json", encoding="utf-8")
        assert queue.reap(now=0.0) == []


class TestInterruptedRun:
    def test_a_run_killed_at_twenty_batches_resumes_with_no_duplicated_work(
        self, tmp_path: Path
    ) -> None:
        """The milestone's exit criterion, played out on disk. Sixty jobs, twenty
        finished, twenty in flight when the worker was killed, and the second
        worker picks up exactly the forty that are left."""
        root = tmp_path / "queue"
        first = Queue(root, Stage.TRANSLATE)
        fill(first, 60)

        finished = []
        for _ in range(20):
            claimed = first.claim(now=1_000.0)
            assert claimed is not None
            first.finish(claimed)
            finished.append(claimed.id)
        holding = [first.claim(now=1_000.0) for _ in range(20)]

        # Ctrl-C. The worker holds nothing: the leases are on disk, not in it.
        second = Queue(root, Stage.TRANSLATE)
        recovered = second.reap(now=1_000.0 + LEASE_SECONDS + 1)
        assert {job.id for job in recovered} == {job.id for job in holding if job}

        remaining = []
        while (claimed := second.claim(now=1_000_000.0)) is not None:
            remaining.append(claimed.id)
            second.finish(claimed)

        assert len(remaining) == 40
        assert not set(remaining) & set(finished)
        assert second.stats().done == 60
        assert second.stats().outstanding == 0

    def test_re_queueing_the_same_work_after_a_resume_adds_nothing(self, queue: Queue) -> None:
        fill(queue, 10)
        for _ in range(4):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            queue.finish(claimed)
        assert queue.extend([make_job(f"job{index:03d}") for index in range(10)]) == 0


class TestRetryAndDrain:
    def test_retry_puts_dead_jobs_back_with_a_clean_slate(self, queue: Queue) -> None:
        queue.add(make_job())
        for _ in range(MAX_ATTEMPTS):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            queue.release(claimed, error="down")
        assert queue.retry() == 1
        assert queue.jobs(State.PENDING)[0].attempts == 0

    def test_retry_is_a_decision_a_person_makes(self, queue: Queue) -> None:
        """There is no automatic path from dead back to pending, on purpose."""
        assert queue.retry(dead=False) == 0

    def test_draining_removes_finished_work_only(self, queue: Queue) -> None:
        fill(queue, 3)
        claimed = queue.claim(now=0.0)
        assert claimed is not None
        queue.finish(claimed)
        assert queue.drain() == 1
        assert queue.count(State.PENDING) == 2

    def test_draining_never_touches_outstanding_work(self, queue: Queue) -> None:
        fill(queue, 2)
        queue.claim(now=0.0)
        queue.drain()
        assert queue.stats().outstanding == 2


class TestStats:
    def test_a_queue_that_has_never_been_used(self, queue: Queue) -> None:
        stats = queue.stats()
        assert stats.total == 0
        assert stats.stage is Stage.TRANSLATE

    def test_every_job_is_counted_once(self, queue: Queue) -> None:
        fill(queue, 5)
        queue.claim(now=0.0)
        finished = queue.claim(now=0.0)
        assert finished is not None
        queue.finish(finished)
        stats = queue.stats()
        assert (stats.pending, stats.leased, stats.done, stats.dead) == (3, 1, 1, 0)
        assert stats.total == 5
        assert stats.outstanding == 4

    def test_the_length_of_a_queue_is_what_is_waiting(self, queue: Queue) -> None:
        fill(queue, 3)
        queue.claim(now=0.0)
        assert len(queue) == 2

    def test_an_unreadable_file_is_reported_rather_than_raised_on(self, queue: Queue) -> None:
        queue.path(State.PENDING, "broken").parent.mkdir(parents=True, exist_ok=True)
        queue.path(State.PENDING, "broken").write_text("{not json", encoding="utf-8")
        assert queue.jobs(State.PENDING) == []

    def test_locate_finds_nothing_for_an_unknown_job(self, queue: Queue) -> None:
        assert queue.locate("nobody") is None


class TestJobFile:
    def test_a_job_round_trips_through_json(self) -> None:
        job = make_job(attempts=2, lease_expires=1_234.5, route="server3", error="timeout")
        assert Job.from_json(job.as_json()) == job

    def test_the_file_is_readable_by_a_person_at_three_in_the_morning(self) -> None:
        text = make_job().as_json()
        assert text.endswith("\n")
        assert "bugs.po" in text

    def test_vietnamese_in_a_payload_is_not_escaped(self) -> None:
        assert "Trả về" in Job(id="a", stage=Stage.TRANSLATE, payload={"hint": "Trả về"}).as_json()

    def test_a_job_file_written_by_an_older_run(self) -> None:
        """Only id and stage have ever been required, so a file missing the rest
        still loads rather than stopping a resume."""
        job = Job.from_json('{"id": "a", "stage": "translate"}')
        assert job.attempts == 0
        assert job.payload == {}


class TestQueues:
    def test_every_stage_has_its_own_queue(self, tmp_path: Path) -> None:
        assert [q.stage for q in queuemod.queues(tmp_path)] == list(Stage)

    def test_draining_one_stage_leaves_the_others(self, tmp_path: Path) -> None:
        translate = Queue(tmp_path, Stage.TRANSLATE)
        judge = Queue(tmp_path, Stage.JUDGE)
        translate.add(make_job())
        judge.add(make_job(stage=Stage.JUDGE))
        claimed = translate.claim(now=0.0)
        assert claimed is not None
        translate.finish(claimed)
        translate.drain()
        assert judge.count(State.PENDING) == 1


class TestBurying:
    def test_a_job_goes_straight_to_dead_whatever_its_attempts_say(self, tmp_path: Path) -> None:
        """For a failure more attempts cannot fix, which release has no way to
        express: it decides from the counter, and the counter is about the
        transport rather than about the answer."""
        one = Queue(tmp_path, Stage.TRANSLATE)
        job = make_job()
        one.add(job)
        one.bury(job, error="3 entries refused after 3 rungs")
        assert one.count(State.DEAD) == 1
        assert one.count(State.PENDING) == 0

    def test_a_buried_job_says_why(self, tmp_path: Path) -> None:
        one = Queue(tmp_path, Stage.TRANSLATE)
        one.add(make_job())
        one.bury(make_job(), error="3 entries refused after 3 rungs")
        assert one.jobs(State.DEAD)[0].error == "3 entries refused after 3 rungs"

    def test_a_buried_job_keeps_the_claims_it_had(self, tmp_path: Path) -> None:
        """The two counters stay separate. A job buried on its first claim is
        not a job that used three, and an audit has to be able to tell them
        apart."""
        one = Queue(tmp_path, Stage.TRANSLATE)
        one.add(make_job())
        claimed = one.claim(now=0.0)
        assert claimed is not None
        one.bury(claimed, error="out of rungs")
        assert one.jobs(State.DEAD)[0].attempts == 1

    def test_a_buried_job_can_be_retried_by_a_person(self, tmp_path: Path) -> None:
        """Never automatic. This is the command you run after the traces told
        you what was wrong and you fixed it."""
        one = Queue(tmp_path, Stage.TRANSLATE)
        one.add(make_job())
        one.bury(make_job(), error="out of rungs")
        assert one.retry() == 1
        assert one.count(State.PENDING) == 1
