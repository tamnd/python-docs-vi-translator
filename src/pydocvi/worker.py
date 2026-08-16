"""The worker pool.

Roughly what the design said it would be: a ``TaskGroup``, one semaphore per
route at its measured concurrency, and ``asyncio.timeout`` around each call.
That is the whole of the concurrency in this project, and it is deliberately
small. Five to eight requests in flight, each of them minutes long, is not a
concurrency problem; it is a scheduling problem with a very small number in it.

Structured cancellation is what earns the ``TaskGroup``. Ctrl-C on hour nine
cancels the group, the in-flight jobs release their leases, and the next run
picks them up. Nothing has to be cleaned up by hand, because nothing was held.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydocvi.client import Answer, Clock, Completions, FleetError, RealClock
from pydocvi.queue import Job, Queue, State
from pydocvi.routes import Route, Router

log = logging.getLogger(__name__)

#: How long to wait when every route is cooling, so that a fleet in an hour-long
#: cooldown does not produce an hour of log lines saying so.
IDLE_POLL = 30.0

type Build = Callable[[Job], str]
type Handle = Callable[[Job, Answer], Awaitable[None]]


@dataclass(slots=True, kw_only=True)
class Progress:
    """What happened, for the line printed at the end of a nine-hour run."""

    done: int = 0
    empty: int = 0
    failed: int = 0
    dead: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_route: dict[str, int] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return self.done + self.empty + self.failed

    @property
    def average_seconds(self) -> float:
        return self.seconds / self.done if self.done else 0.0

    @property
    def calls_per_hour(self) -> float:
        """Measured throughput, which is the only number worth estimating from.

        Wall clock rather than the sum of call durations, because the whole
        point of the pool is that the calls overlap.
        """
        return self.done / (self.seconds / 3600) if self.seconds else 0.0

    def record(self, answer: Answer) -> None:
        self.by_route[answer.route] = self.by_route.get(answer.route, 0) + 1
        self.prompt_tokens += answer.usage.prompt_tokens
        self.completion_tokens += answer.usage.completion_tokens


class Worker:
    """Drains one queue through one router.

    ``build`` turns a job into a prompt and ``handle`` does something with the
    answer. Both are supplied by the calling stage, so this module knows nothing
    about translation, judging or glossaries and can be tested with two
    lambdas.
    """

    def __init__(
        self,
        *,
        queue: Queue,
        router: Router,
        client: Completions,
        build: Build,
        handle: Handle,
        clock: Clock | None = None,
        limit: int | None = None,
    ) -> None:
        self.queue = queue
        self.router = router
        self.client = client
        self.build = build
        self.handle = handle
        self.clock = clock or RealClock()
        self.limit = limit
        self.progress = Progress()
        self._permits = {
            state.route.name: asyncio.Semaphore(state.route.concurrency) for state in router
        }
        self._claimed = 0
        self._stop = False

    @property
    def slots(self) -> int:
        """How many calls may be in flight, summed over the routes."""
        return sum(route.concurrency for route in self.router.routes)

    async def run(self) -> Progress:
        """Work until the queue is empty, the limit is reached, or Ctrl-C.

        The task group is entered once and the workers loop inside it, rather
        than one task per job. 2 776 tasks would be fine, but the pool size is
        then a property of the scheduler rather than something anybody can see.
        """
        started = self.clock.now()
        self.queue.reap(started)
        try:
            async with asyncio.TaskGroup() as group:
                for index in range(self.slots):
                    group.create_task(self._loop(), name=f"pydocvi-worker-{index}")
        except* FleetError as errors:
            for error in errors.exceptions:
                log.error("worker stopped", extra={"reason": str(error)})
        self.progress.seconds = self.clock.now() - started
        return self.progress

    def stop(self) -> None:
        """Ask the loops to finish their current call and return."""
        self._stop = True

    async def _loop(self) -> None:
        while not self._stop:
            if self.limit is not None and self._claimed >= self.limit:
                return
            now = self.clock.now()
            if self.router.all_cooling(now):
                wait = min(self.router.next_free(now), IDLE_POLL)
                log.info("every route is cooling", extra={"seconds": wait})
                await self.clock.sleep(max(wait, 1.0))
                continue
            job = self.queue.claim(now)
            if job is None:
                return
            self._claimed += 1
            await self._attempt(job)

    async def _attempt(self, job: Job) -> None:
        """One job against the best route that has not already refused it.

        Running out of routes is not itself a failed call. The call that emptied
        the list already counted, and counting it twice would inflate the
        throughput figure that every wall-clock estimate is computed from.
        """
        tried: list[str] = []
        while True:
            route = self.router.pick(self.clock.now(), skip=tried)
            if route is None:
                state = self.queue.release(job, error="every route unavailable")
                self.progress.dead += state is State.DEAD
                return
            tried.append(route.name)
            if await self._call(job, route):
                return

    async def _call(self, job: Job, route: Route) -> bool:
        """Returns whether the job is finished with, one way or another."""
        async with self._permits[route.name]:
            try:
                answer = await self.client.complete(route, self.build(job))
            except FleetError as error:
                self.router.cool(route.name, self.clock.now(), str(error))
                self.progress.failed += 1
                return False

        if answer.empty:
            # A 200 with nothing in it is a refusal, not a transport failure. It
            # costs the route a cooldown and the job an attempt, and it is never
            # retried against the same session, which would return the same
            # nothing.
            self.router.cool(route.name, self.clock.now(), "empty answer")
            self.progress.empty += 1
            state = self.queue.release(job, error="empty answer")
            self.progress.dead += state is State.DEAD
            return True

        self.router.succeed(route.name)
        self.progress.record(answer)
        await self.handle(job, answer)
        self.queue.finish(job)
        self.progress.done += 1
        log.info(
            "job done",
            extra={"job": job.id, "route": route.name, "seconds": round(answer.seconds)},
        )
        return True
