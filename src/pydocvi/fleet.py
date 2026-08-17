"""Tunnels, probes and traces.

The listener on every host is loopback-only, so the laptop reaches it through an
SSH port forward. Every ``subprocess`` call in this module goes through a runner
protocol, which means the whole file is testable with a fake and the test suite
never opens a socket.

``ExitOnForwardFailure=yes`` is the one flag that is not optional. Without it a
tunnel whose local port is already taken comes up reporting success, and every
request through it goes to whatever else is listening on that port. That failure
looks exactly like a model problem and has cost hours.
"""

import asyncio
import logging
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from pydocvi.client import Clock, Completions, FleetError, RealClock, redact
from pydocvi.routes import Route

log = logging.getLogger(__name__)

#: Where ``chatgpt-tool serve`` writes the request and reply for every call.
TRACE_ROOT = "$HOME/data/chatgpt-proxy/traces"

SSH_OPTIONS = (
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    "BatchMode=yes",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Ran:
    """The result of one command."""

    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def message(self) -> str:
        return (self.err or self.out).strip().splitlines()[-1] if (self.err or self.out) else ""


class Runner(Protocol):
    """Runs a command. The seam that keeps SSH out of the test suite."""

    def run(self, command: Sequence[str], *, timeout: float = 30.0) -> Ran: ...


class Subprocess:
    def run(self, command: Sequence[str], *, timeout: float = 30.0) -> Ran:
        log.debug("running", extra={"command": shlex.join(command)})
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return Ran(code=124, err=f"timed out after {timeout:.0f}s")
        except FileNotFoundError as error:
            return Ran(code=127, err=str(error))
        return Ran(code=done.returncode, out=done.stdout, err=done.stderr)


@dataclass(frozen=True, slots=True, kw_only=True)
class Tunnel:
    """What one route's forward looks like right now."""

    route: str
    host: str
    local_port: int
    remote_port: int
    up: bool
    detail: str = ""

    def __str__(self) -> str:
        state = "up" if self.up else "down"
        return (
            f"{self.route}: {state} 127.0.0.1:{self.local_port} -> {self.host}:{self.remote_port}"
        )


class Fleet:
    """Tunnel management for a set of routes."""

    def __init__(self, routes: Sequence[Route], *, runner: Runner | None = None) -> None:
        self.routes = list(routes)
        self.runner = runner or Subprocess()

    def up(self, route: Route) -> Tunnel:
        """Open one forward, or report why not.

        Already up is a success. Running ``fleet up`` twice is what everybody
        does, and the second run should not tear down a working tunnel to prove
        it can build one.
        """
        if self._listening(route.local_port):
            if self.probe(route).up:
                return self._tunnel(route, up=True, detail="already up")
            return self._tunnel(
                route,
                up=False,
                detail=f"port {route.local_port} is taken by something else",
            )
        ran = self.runner.run(
            [
                "ssh",
                "-f",
                "-N",
                *SSH_OPTIONS,
                "-L",
                f"{route.local_port}:127.0.0.1:{route.remote_port}",
                route.host,
            ],
            timeout=45.0,
        )
        if not ran.ok:
            return self._tunnel(route, up=False, detail=redact(ran.message, self.routes))
        return self._tunnel(route, up=True, detail="opened")

    def down(self, route: Route) -> bool:
        """Close one forward. Returns whether anything was closed."""
        pids = self._forwarding(route.local_port)
        for pid in pids:
            self.runner.run(["kill", pid])
        return bool(pids)

    def status(self) -> list[Tunnel]:
        return [self._tunnel(route, up=self._listening(route.local_port)) for route in self.routes]

    def probe(self, route: Route) -> Tunnel:
        """Whether anything answers ``/v1/health`` through the forward.

        Deliberately curl rather than httpx: ``probe`` has to work when the
        Python side is what is broken, and a diagnostic that shares its code
        with the thing being diagnosed is not a diagnostic.
        """
        ran = self.runner.run(
            [
                "curl",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--max-time",
                "20",
                route.health_url,
            ],
            timeout=25.0,
        )
        code = ran.out.strip()
        return self._tunnel(
            route,
            up=ran.ok and code == "200",
            detail=f"health {code or ran.message}",
        )

    def trace(self, route: Route, batch: str, *, day: str) -> str:
        """Fetch the request and reply that one batch produced, over SSH.

        The batch id is in the provenance comment of every entry it wrote, so
        the path from a wrong Vietnamese sentence to the exact prompt that
        produced it is two commands and no ssh session.
        """
        pattern = f"{TRACE_ROOT}/{day}"
        ran = self.runner.run(
            ["ssh", *SSH_OPTIONS, route.host, f"grep -rl {shlex.quote(batch)} {pattern} | head -5"],
            timeout=60.0,
        )
        if not ran.ok or not ran.out.strip():
            return ""
        found = ran.out.split()
        fetched = self.runner.run(
            ["ssh", *SSH_OPTIONS, route.host, f"cat {' '.join(shlex.quote(f) for f in found)}"],
            timeout=120.0,
        )
        return redact(fetched.out, self.routes)

    def _listening(self, port: int) -> bool:
        return bool(self._forwarding(port))

    def _forwarding(self, port: int) -> list[str]:
        """Process ids holding a local port, via ``lsof``.

        ``lsof`` rather than parsing ``ps`` for an ssh command line, because the
        question is who has the port and that is exactly what lsof answers.
        """
        ran = self.runner.run(["lsof", "-t", "-i", f"@127.0.0.1:{port}", "-sTCP:LISTEN"])
        return [line for line in ran.out.split() if line.isdigit()]

    def _tunnel(self, route: Route, *, up: bool, detail: str = "") -> Tunnel:
        return Tunnel(
            route=route.name,
            host=route.host,
            local_port=route.local_port,
            remote_port=route.remote_port,
            up=up,
            detail=detail,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Diagnosis:
    """What ``doctor`` found, and the exit code that follows from it."""

    tunnels: list[Tunnel]
    missing_keys: list[str]
    cooling: list[str]

    @property
    def answering(self) -> list[Tunnel]:
        return [tunnel for tunnel in self.tunnels if tunnel.up]

    @property
    def healthy(self) -> bool:
        """One route answering is enough to start a run."""
        return bool(self.answering) and not self.missing_keys

    @property
    def summary(self) -> str:
        if self.missing_keys:
            return f"key(s) not in the environment: {', '.join(self.missing_keys)}"
        if not self.tunnels:
            return "no routes configured"
        if not self.answering:
            return "no route answers, every tunnel is down or the hosts are not serving"
        return f"{len(self.answering)} of {len(self.tunnels)} routes answering"


@dataclass(frozen=True, slots=True, kw_only=True)
class Bench:
    """One host's measured throughput.

    Every wall-clock estimate this project prints is computed from these
    numbers rather than from the table in the design notes, which was an
    aspiration written before anything had run.
    """

    route: str
    calls: int
    failures: int
    empty: int
    seconds: float
    concurrency: int
    latency: float = 0.0

    @property
    def successes(self) -> int:
        return self.calls - self.failures - self.empty

    @property
    def average_seconds(self) -> float:
        """Wall clock divided by calls, which is throughput and not latency.

        At concurrency 4 this came out at 32 seconds for a host whose calls take
        96. Both numbers are worth having and they are not the same number, which
        is why ``latency`` is kept separately and the report prints both.
        """
        return self.seconds / self.calls if self.calls else 0.0

    @property
    def calls_per_hour(self) -> float:
        return self.successes / (self.seconds / 3600) if self.seconds else 0.0

    @property
    def parallelism(self) -> float:
        """How many calls the host really had in flight, measured rather than asked for.

        A host that serialises answers at 1.0 however high its configured
        concurrency is, and that is the whole of the safe-concurrency question.
        """
        return self.latency / self.average_seconds if self.average_seconds else 0.0

    @classmethod
    def of(
        cls, route: Route, *, durations: Sequence[float], failures: int, empty: int, wall: float
    ) -> Self:
        return cls(
            route=route.name,
            calls=len(durations) + failures + empty,
            failures=failures,
            empty=empty,
            seconds=wall,
            concurrency=route.concurrency,
            latency=sum(durations) / len(durations) if durations else 0.0,
        )


def hours_for(batches: int, calls_per_hour: float, *, retry_rate: float = 0.15) -> float:
    """Wall clock for a number of batches at a measured rate.

    The retry rate is included because the estimate a person acts on is the one
    that says how long it will actually take, and 15 % of calls on this
    transport come back empty or fail.
    """
    if calls_per_hour <= 0:
        return 0.0
    return batches * (1 + retry_rate) / calls_per_hour


def bench_markdown(results: Sequence[Bench], *, batches: int, absent: Sequence[str] = ()) -> str:
    """The report ``fleet bench`` writes, and the source of every estimate.

    A route that was not measured is named rather than left out silently. The
    fleet total is the sum of what was measured, and a reader who does not know
    a host is missing from it will read it as the whole fleet and plan off a
    number that is too small.

    Two seconds columns, because they are two different numbers and the table
    used to print only the second one under a heading that read like the first.
    "Seconds a call" is how long a call takes. "Wall per call" is the wall clock
    divided by the calls, which is what a tier's duration is made of. The last
    column is the first divided by the second, being how many calls the host
    really had in flight, and a host that serialises answers 1.0 there whatever
    its configured concurrency says.
    """
    lines = [
        "# Fleet bench",
        "",
        "| route | calls | empty | failed | seconds a call | wall per call "
        "| calls/hour | concurrency | measured |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.route} | {result.calls} | {result.empty} | {result.failures} "
            f"| {result.latency:.0f} | {result.average_seconds:.0f} "
            f"| {result.calls_per_hour:.1f} "
            f"| {result.concurrency} | {result.parallelism:.1f} |"
        )
    total = sum(result.calls_per_hour for result in results)
    lines += [
        f"| **fleet** | | | | | | **{total:.1f}** | | |",
        "",
        f"A full pass over {batches:,} batches is {hours_for(batches, total):.1f} hours "
        "at this rate, including retries.",
        "",
    ]
    if absent:
        lines += [
            f"Not measured, and not in the total: {', '.join(absent)}.",
            "",
        ]
    return "\n".join(lines)


async def bench(
    client: Completions,
    route: Route,
    *,
    calls: int,
    prompt: str,
    clock: Clock | None = None,
) -> Bench:
    """Measure one route by making real calls at its stated concurrency.

    Wall clock rather than the sum of the call durations, because the number
    everybody wants is how long a tier takes and the calls overlap. A route
    whose concurrency is wrong shows up here as a mean latency that climbs with
    each extra call in flight rather than staying flat.
    """
    ticker = clock or RealClock()
    permits = asyncio.Semaphore(route.concurrency)
    durations: list[float] = []
    failures = empty = 0
    started = ticker.now()

    async def once() -> None:
        nonlocal failures, empty
        async with permits:
            try:
                answer = await client.complete(route, prompt)
            except FleetError as error:
                failures += 1
                log.warning("bench call failed", extra={"route": route.name, "why": str(error)})
                return
        if answer.empty:
            empty += 1
        else:
            durations.append(answer.seconds)

    async with asyncio.TaskGroup() as group:
        for _ in range(calls):
            group.create_task(once())

    return Bench.of(
        route,
        durations=durations,
        failures=failures,
        empty=empty,
        wall=ticker.now() - started,
    )
