"""The HTTP client.

One call is two to ten minutes of a browser session on another continent
pretending to be an API. Every decision here follows from that.

Streaming is on, because a non-streaming request holds a connection open with
nothing travelling on it for eight minutes, which is the exact shape that an
idle timeout in some middlebox kills at minute five.

Retrying inside a call is for failures that look like they will pass on their
own: a dropped tunnel, a session that logged itself out. Moving to another host
is the router's job. Mixing the two produces a client that spends twenty minutes
retrying against a host that is not coming back.

An empty answer is an outcome, not an error. A browser-backed session returns
200 with nothing in it often enough that treating it as an exception would mean
the exceptional path is the common one.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, Self

import httpx

from pydocvi.routes import Route

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 30.0
HARD_CAP = 1200.0
RETRIES = 2
RETRY_DELAY = 30.0

#: Anything key-shaped, so that a pasted log or a fetched trace never carries a
#: live key. The keys here are shared across hosts, so one leak is every host.
KEYISH = re.compile(r"sk-[A-Za-z0-9._-]{8,}")

#: Statuses worth asking again about on the same host. Everything else is either
#: our fault (4xx) or a host that needs to cool down.
RETRYABLE = frozenset({500, 502, 503, 504, 408, 429})


class FleetError(Exception):
    """The call did not produce an answer. Exit code 3 territory."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """What the call cost, as reported on the final chunk."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_payload(cls, payload: dict[str, object] | None) -> Self:
        if not payload:
            return cls()
        return cls(
            prompt_tokens=_count(payload.get("prompt_tokens")),
            completion_tokens=_count(payload.get("completion_tokens")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """One completion, or the fact that there was not one.

    ``empty`` is the interesting case and it is why this is a value rather than
    a string. It consumes an attempt, it cools the route, and it is not retried
    inside the call, because asking the same logged-out session twice in a row
    gets the same nothing twice in a row.
    """

    text: str
    route: str
    model: str
    seconds: float
    usage: Usage = field(default_factory=Usage)
    attempts: int = 1

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def __str__(self) -> str:
        state = "empty" if self.empty else f"{len(self.text):,} chars"
        return f"{self.route}: {state} in {self.seconds:.0f}s"


class Clock(Protocol):
    """Time, so that the retry and cooldown tests do not sleep.

    The fake in ``conftest.py`` advances instantly. A test suite that sleeps
    through a doubling backoff is a test suite nobody runs before pushing.
    """

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall clock, deliberately, rather than the loop's monotonic clock.

    A lease deadline is written into a file that a different process reads after
    a Ctrl-C, and a loop clock starts again from roughly zero in that process.
    Every lease would then look expired, or none of them would, depending on
    which way the numbers fell.
    """

    def now(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class Completions(Protocol):
    """What the worker needs from a client, so a fake can supply it."""

    async def complete(self, route: Route, prompt: str, *, system: str | None = None) -> Answer: ...


class Client:
    """An OpenAI-shaped chat completions client, one per run.

    One ``httpx.AsyncClient`` is shared across routes because connection pools
    are per host and every route here is a different local port on the loopback
    interface. The per-route limit that matters is the semaphore in the worker,
    not the pool.
    """

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
        retries: int = RETRIES,
        retry_delay: float = RETRY_DELAY,
        jitter: bool = True,
    ) -> None:
        self._transport = transport
        self._clock = clock or RealClock()
        self._retries = retries
        self._retry_delay = retry_delay
        self._jitter = jitter
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(HARD_CAP, connect=CONNECT_TIMEOUT),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, route: Route, prompt: str, *, system: str | None = None) -> Answer:
        """One completion, with retries that stay on this route.

        Raises ``FleetError`` when the route is not going to answer. Returns an
        empty ``Answer`` when it answered with nothing, which is a different
        thing and is handled differently upstream.
        """
        started = self._clock.now()
        last = "no attempt made"
        for attempt in range(1, self._retries + 2):
            try:
                async with asyncio.timeout(route.timeout):
                    text, usage = await self._stream(route, _messages(prompt, system))
            except TimeoutError:
                last = f"no answer within {route.timeout:.0f}s"
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                last = f"HTTP {status}"
                if status not in RETRYABLE:
                    raise FleetError(f"{route.name}: {last}") from error
            except httpx.HTTPError as error:
                last = f"{type(error).__name__}: {error}"
            else:
                return Answer(
                    text=text,
                    route=route.name,
                    model=route.model,
                    seconds=self._clock.now() - started,
                    usage=usage,
                    attempts=attempt,
                )

            log.warning(
                "call failed, retrying on the same route",
                extra={"route": route.name, "attempt": attempt, "reason": last},
            )
            if attempt <= self._retries:
                await self._clock.sleep(self._backoff(attempt))

        raise FleetError(f"{route.name}: {last} after {self._retries + 1} attempts")

    async def health(self, route: Route) -> bool:
        """Whether the route answers ``GET /v1/health``.

        A route that fails this is skipped without consuming one of the job's
        three attempts, which is the whole point of having it.
        """
        client = self._require()
        try:
            response = await client.get(
                route.health_url, headers=_headers(route), timeout=CONNECT_TIMEOUT
            )
        except httpx.HTTPError as error:
            log.info("health check failed", extra={"route": route.name, "reason": str(error)})
            return False
        return response.status_code == httpx.codes.OK

    async def _stream(self, route: Route, messages: list[dict[str, str]]) -> tuple[str, Usage]:
        client = self._require()
        body = {
            "model": route.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[str] = []
        usage = Usage()
        async with client.stream(
            "POST",
            f"{route.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=_headers(route),
        ) as response:
            response.raise_for_status()
            async for piece, reported in _events(response):
                chunks.append(piece)
                if reported is not None:
                    usage = reported
        return "".join(chunks), usage

    def _backoff(self, attempt: int) -> float:
        """Doubling, jittered.

        Jittered because two workers that failed on the same dropped tunnel
        otherwise come back at the same instant and drop it again.
        """
        delay = self._retry_delay * 2 ** (attempt - 1)
        return delay * random.uniform(0.5, 1.5) if self._jitter else float(delay)

    def _require(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("client used outside its context manager")
        return self._client


async def _events(response: httpx.Response) -> AsyncIterator[tuple[str, Usage | None]]:
    """Content and usage out of a server-sent event stream.

    Malformed lines are skipped rather than raised on. Half an answer is worth
    more than none, and the invariants downstream will refuse it if it is not.
    """
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            log.debug("skipping unparseable stream line", extra={"line": payload[:120]})
            continue
        usage = Usage.from_payload(event.get("usage")) if event.get("usage") else None
        for choice in event.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                yield content, usage
                usage = None
        if usage is not None:
            yield "", usage


def _count(value: object) -> int:
    """A token count out of a payload nobody validated.

    Hosts in this fleet have reported usage as a string, as null and not at all.
    A wrong estimate is worth more than a stack trace nine hours into a run.
    """
    return int(value) if isinstance(value, int | float | str) and str(value).isdigit() else 0


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": prompt}]


def _headers(route: Route) -> dict[str, str]:
    key = route.key
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    return headers


def redact(text: str, routes: Sequence[Route] = ()) -> str:
    """Remove anything key-shaped from text before it is printed or logged.

    Applied to every trace and every error message that reaches a terminal. The
    keys in this project are shared across hosts, which means one of them in one
    pasted log is one of them everywhere.
    """
    out = text
    for route in routes:
        key = route.key
        if key:
            out = out.replace(key, "***")
    return KEYISH.sub("***", out)
