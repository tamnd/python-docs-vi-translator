from pathlib import Path

import pytest

from conftest import FakeClient, FakeClock, make_route
from pydocvi.client import Answer, FleetError, Usage
from pydocvi.queue import Job, Queue, Stage, State
from pydocvi.render import Prompt
from pydocvi.routes import BASE_COOLDOWN, Router
from pydocvi.worker import IDLE_POLL, Progress, Worker


def asking(user: str, system: str = "you are a translator") -> Prompt:
    """The pair of messages a build returns, with nothing interesting in it."""
    return Prompt(system=system, user=user)


def make_queue(tmp_path: Path, count: int = 1) -> Queue:
    queue = Queue(tmp_path / "queue", Stage.TRANSLATE)
    queue.extend(
        [
            Job(id=f"job{index:03d}", stage=Stage.TRANSLATE, payload={"n": index})
            for index in range(count)
        ]
    )
    return queue


def make_worker(
    queue: Queue,
    *,
    router: Router | None = None,
    client: FakeClient | None = None,
    clock: FakeClock | None = None,
    limit: int | None = None,
) -> tuple[Worker, list[tuple[str, str]]]:
    """A worker and the list of answers it handled, in order."""
    handled: list[tuple[str, str]] = []

    async def handle(job: Job, answer: Answer) -> None:
        handled.append((job.id, answer.text))

    worker = Worker(
        queue=queue,
        router=router or Router.of([make_route("a")]),
        client=client or FakeClient(),
        build=lambda job: asking(f"translate {job.id}"),
        handle=handle,
        clock=clock or FakeClock(),
        limit=limit,
    )
    return worker, handled


@pytest.mark.asyncio
class TestDraining:
    async def test_every_job_is_worked_and_the_queue_empties(self, tmp_path: Path) -> None:
        queue = make_queue(tmp_path, 5)
        worker, handled = make_worker(queue)
        progress = await worker.run()
        assert progress.done == 5
        assert len(handled) == 5
        assert queue.stats().outstanding == 0
        assert queue.stats().done == 5

    async def test_a_job_is_only_finished_after_its_answer_was_handled(
        self, tmp_path: Path
    ) -> None:
        """If the two were the other way round, a crash between them would mark
        a batch done with nothing written for it."""
        queue = make_queue(tmp_path, 1)
        seen: list[tuple[int, int]] = []

        async def handle(job: Job, answer: Answer) -> None:
            seen.append((queue.count(State.LEASED), queue.count(State.DONE)))

        worker = Worker(
            queue=queue,
            router=Router.of([make_route("a")]),
            client=FakeClient(),
            build=lambda job: asking("x"),
            handle=handle,
            clock=FakeClock(),
        )
        await worker.run()
        assert seen == [(1, 0)]

    async def test_the_prompt_comes_from_build(self, tmp_path: Path) -> None:
        client = FakeClient()
        worker, _ = make_worker(make_queue(tmp_path, 1), client=client)
        await worker.run()
        assert client.calls == [("a", "translate job000")]

    async def test_the_system_message_goes_over_too(self, tmp_path: Path) -> None:
        """The whole translating instruction and the batch's terminology are in
        it. A worker that built both messages and sent one would produce a run
        whose prompt hash described a prompt the model never saw."""
        client = FakeClient()
        worker, _ = make_worker(make_queue(tmp_path, 1), client=client)
        await worker.run()
        assert client.systems == ["you are a translator"]

    async def test_an_empty_queue_returns_immediately(self, tmp_path: Path) -> None:
        worker, _ = make_worker(make_queue(tmp_path, 0))
        assert (await worker.run()).calls == 0

    async def test_a_limit_stops_early_and_leaves_the_rest(self, tmp_path: Path) -> None:
        """The flag behind ``--limit 20``, which is how a prompt change is tried
        on twenty batches rather than on two thousand."""
        queue = make_queue(tmp_path, 10)
        worker, _ = make_worker(queue, limit=3)
        progress = await worker.run()
        assert progress.done == 3
        assert queue.count(State.PENDING) == 7

    async def test_expired_leases_are_reaped_before_anything_else(self, tmp_path: Path) -> None:
        """A worker started after a Ctrl-C finds the previous run's jobs waiting
        for it rather than an empty queue and forty leases nobody holds."""
        queue = make_queue(tmp_path, 3)
        for _ in range(3):
            queue.claim(now=0.0)
        clock = FakeClock(start=1_000_000.0)
        worker, _ = make_worker(queue, clock=clock)
        assert (await worker.run()).done == 3


@pytest.mark.asyncio
class TestFailover:
    async def test_a_failing_route_hands_the_job_to_the_next_one(self, tmp_path: Path) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        client = FakeClient([FleetError("a: HTTP 503"), "Một."])
        worker, _ = make_worker(make_queue(tmp_path, 1), router=router, client=client)
        progress = await worker.run()
        assert progress.done == 1
        assert [name for name, _ in client.calls] == ["a", "b"]

    async def test_a_route_that_failed_is_cooling(self, tmp_path: Path) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        client = FakeClient([FleetError("a: HTTP 503"), "Một.", "Một."])
        worker, _ = make_worker(make_queue(tmp_path, 1), router=router, client=client)
        clock = worker.clock
        await worker.run()
        assert router.health["a"].cooling(clock.now())
        assert not router.health["b"].cooling(clock.now())

    async def test_the_same_route_is_never_tried_twice_for_one_job(self, tmp_path: Path) -> None:
        """Two hosts, one job, both refusing. Three attempts against the same
        session would be three times the same nothing."""
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        client = FakeClient([FleetError("down")])
        worker, _ = make_worker(make_queue(tmp_path, 1), router=router, client=client, limit=1)
        await worker.run()
        assert [name for name, _ in client.calls] == ["a", "b"]

    async def test_a_job_no_route_will_take_goes_back_to_the_queue(self, tmp_path: Path) -> None:
        queue = make_queue(tmp_path, 1)
        client = FakeClient([FleetError("down")])
        worker, _ = make_worker(queue, client=client, limit=1)
        progress = await worker.run()
        assert progress.failed == 1
        assert progress.done == 0
        assert queue.count(State.PENDING) == 1

    async def test_a_job_that_ran_out_of_attempts_is_dead(self, tmp_path: Path) -> None:
        queue = make_queue(tmp_path, 1)
        for _ in range(2):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            queue.release(claimed, error="earlier failure")
        client = FakeClient([FleetError("down")])
        worker, _ = make_worker(queue, client=client)
        progress = await worker.run()
        assert progress.dead == 1
        assert queue.count(State.DEAD) == 1


@pytest.mark.asyncio
class TestEmptyAnswers:
    async def test_an_empty_answer_costs_the_route_a_cooldown(self, tmp_path: Path) -> None:
        router = Router.of([make_route("a")])
        clock = FakeClock()
        worker, _ = make_worker(
            make_queue(tmp_path, 1), router=router, client=FakeClient([""]), clock=clock, limit=1
        )
        progress = await worker.run()
        assert progress.empty == 1
        assert router.health["a"].remaining(clock.now()) == BASE_COOLDOWN

    async def test_an_empty_answer_returns_the_job_rather_than_finishing_it(
        self, tmp_path: Path
    ) -> None:
        queue = make_queue(tmp_path, 1)
        worker, handled = make_worker(queue, client=FakeClient([""]), limit=1)
        await worker.run()
        assert handled == []
        assert queue.count(State.PENDING) == 1
        assert queue.count(State.DONE) == 0

    async def test_an_empty_answer_is_not_retried_against_the_same_route(
        self, tmp_path: Path
    ) -> None:
        """The route is cooling by then, so the job comes back to the queue and
        waits for a route that is answering."""
        client = FakeClient([""])
        worker, _ = make_worker(make_queue(tmp_path, 1), client=client, limit=1)
        await worker.run()
        assert len(client.calls) == 1

    async def test_an_empty_answer_on_the_last_attempt_kills_the_job(self, tmp_path: Path) -> None:
        queue = make_queue(tmp_path, 1)
        for _ in range(2):
            claimed = queue.claim(now=0.0)
            assert claimed is not None
            queue.release(claimed, error="empty answer")
        worker, _ = make_worker(queue, client=FakeClient([""]))
        progress = await worker.run()
        assert progress.dead == 1
        assert queue.jobs(State.DEAD)[0].error == "empty answer"


@pytest.mark.asyncio
class TestCoolingFleet:
    async def test_a_fleet_that_is_entirely_cooling_waits_rather_than_spins(
        self, tmp_path: Path
    ) -> None:
        """Without this the worker burns a core asking a router that has already
        said no, several thousand times a second, for five minutes."""
        router = Router.of([make_route("a")])
        clock = FakeClock()
        router.cool("a", clock.now(), "empty answer")
        worker, _ = make_worker(make_queue(tmp_path, 1), router=router, clock=clock)
        progress = await worker.run()
        assert clock.slept == [IDLE_POLL] * 10
        assert sum(clock.slept) == BASE_COOLDOWN
        assert progress.done == 1

    async def test_the_wait_is_capped_so_the_run_notices_a_route_coming_back(
        self, tmp_path: Path
    ) -> None:
        router = Router.of([make_route("a")])
        clock = FakeClock()
        for _ in range(6):
            router.cool("a", clock.now(), "empty answer")
        worker, _ = make_worker(make_queue(tmp_path, 1), router=router, clock=clock)
        await worker.run()
        assert set(clock.slept) == {IDLE_POLL}


@pytest.mark.asyncio
class TestConcurrency:
    async def test_the_pool_is_the_sum_of_the_route_concurrencies(self) -> None:
        router = Router.of([make_route("a", concurrency=4), make_route("b", concurrency=1)])
        worker, _ = make_worker(Queue(Path("/nonexistent"), Stage.TRANSLATE), router=router)
        assert worker.slots == 5

    async def test_no_route_is_asked_for_more_than_it_can_take(self, tmp_path: Path) -> None:
        """One route at concurrency one, four loops. The semaphore is the only
        thing keeping the fourth call from arriving while the first is open."""
        router = Router.of([make_route("a", concurrency=1)])
        client = FakeClient()
        worker, _ = make_worker(make_queue(tmp_path, 8), router=router, client=client)
        assert worker.slots == 1
        await worker.run()
        assert len(client.calls) == 8

    async def test_stopping_asks_the_loops_to_finish(self, tmp_path: Path) -> None:
        queue = make_queue(tmp_path, 5)
        worker, _ = make_worker(queue)
        worker.stop()
        assert (await worker.run()).calls == 0
        assert queue.count(State.PENDING) == 5


@pytest.mark.asyncio
class TestStageFailures:
    async def test_a_stage_that_gives_up_stops_the_worker_without_a_traceback(
        self, tmp_path: Path
    ) -> None:
        """handle writes the catalog, and a stage that cannot write has nothing
        useful left to do. It stops with a logged reason rather than raising
        through nine hours of finished work."""
        queue = make_queue(tmp_path, 5)

        async def refuse(job: Job, answer: Answer) -> None:
            raise FleetError("content checkout is read only")

        worker = Worker(
            queue=queue,
            router=Router.of([make_route("a")]),
            client=FakeClient(),
            build=lambda job: asking("x"),
            handle=refuse,
            clock=FakeClock(),
        )
        progress = await worker.run()
        assert progress.done == 0
        assert queue.count(State.DONE) == 0

    async def test_every_worker_that_gave_up_is_named(self, tmp_path: Path) -> None:
        """Four loops against one host, all of them stopping for the same
        reason. The reason is logged once per loop rather than once."""
        queue = make_queue(tmp_path, 8)

        async def refuse(job: Job, answer: Answer) -> None:
            raise FleetError("content checkout is read only")

        worker = Worker(
            queue=queue,
            router=Router.of([make_route("a", concurrency=4)]),
            client=FakeClient(),
            build=lambda job: asking("x"),
            handle=refuse,
            clock=FakeClock(),
        )
        assert (await worker.run()).done == 0


@pytest.mark.asyncio
class TestProgress:
    async def test_tokens_and_routes_are_totalled(self, tmp_path: Path) -> None:
        worker, _ = make_worker(make_queue(tmp_path, 3))
        progress = await worker.run()
        assert progress.prompt_tokens == 300
        assert progress.completion_tokens == 150
        assert progress.by_route == {"a": 3}

    async def test_calls_counts_everything_that_reached_a_host(self, tmp_path: Path) -> None:
        client = FakeClient([FleetError("down"), "", "Một."])
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        worker, _ = make_worker(make_queue(tmp_path, 3), router=router, client=client)
        progress = await worker.run()
        assert progress.calls == progress.done + progress.empty + progress.failed


class TestProgressArithmetic:
    def test_throughput_is_measured_against_the_wall_clock(self) -> None:
        """The sum of the call durations would count a pool of five as five
        times faster than it is."""
        progress = Progress(done=20, seconds=3600.0)
        assert progress.calls_per_hour == 20.0
        assert progress.average_seconds == 180.0

    def test_a_run_that_did_nothing_reports_zero_rather_than_dividing_by_it(self) -> None:
        assert Progress().calls_per_hour == 0.0
        assert Progress().average_seconds == 0.0

    def test_an_answer_is_recorded_against_its_route(self) -> None:
        progress = Progress()
        answer = Answer(
            text="Một.",
            route="server3",
            model="gpt-5",
            seconds=120.0,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        progress.record(answer)
        progress.record(answer)
        assert progress.by_route == {"server3": 2}
        assert progress.prompt_tokens == 20
