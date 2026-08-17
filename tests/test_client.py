import asyncio
import json
import logging
import time
from collections.abc import Iterator

import httpx
import pytest

from conftest import FAKE_KEY, FakeClock, make_route
from pydocvi import client
from pydocvi.client import Client, FleetError, Usage
from pydocvi.routes import Route


def sse(*pieces: str, usage: dict[str, int] | None = None) -> str:
    """A stream shaped like the one the host actually sends."""
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n" for piece in pieces
    ]
    if usage is not None:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    return "".join([*lines, "data: [DONE]\n\n"])


def transport(*responses: httpx.Response | Exception) -> httpx.MockTransport:
    """Answers in order, so a test can fail once and then succeed."""
    remaining: Iterator[httpx.Response | Exception] = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        answer = next(remaining)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.MockTransport(handle)


def ok(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
class TestCompletion:
    async def test_the_pieces_of_a_stream_are_joined(self, route: Route) -> None:
        async with Client(transport=transport(ok(sse("Trả về ", "danh sách.")))) as api:
            answer = await api.complete(route, "1 Return a list.")
        assert answer.text == "Trả về danh sách."
        assert answer.route == route.name
        assert not answer.empty

    async def test_usage_comes_off_the_final_chunk(self, route: Route) -> None:
        body = sse("Một.", usage={"prompt_tokens": 900, "completion_tokens": 40})
        async with Client(transport=transport(ok(body))) as api:
            answer = await api.complete(route, "x")
        assert answer.usage == Usage(prompt_tokens=900, completion_tokens=40)
        assert answer.usage.total == 940

    async def test_a_system_message_goes_first(self, route: Route) -> None:
        seen: list[list[dict[str, str]]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content)["messages"])
            return ok(sse("Một."))

        async with Client(transport=httpx.MockTransport(handle)) as api:
            await api.complete(route, "user text", system="system text")
        assert [message["role"] for message in seen[0]] == ["system", "user"]

    async def test_the_key_travels_as_a_bearer_token(
        self, route: Route, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHATGPT_PROXY_KEY", FAKE_KEY)
        seen: list[httpx.Headers] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return ok(sse("Một."))

        async with Client(transport=httpx.MockTransport(handle)) as api:
            await api.complete(route, "x")
        assert seen[0]["authorization"] == f"Bearer {FAKE_KEY}"

    async def test_a_route_with_no_key_sends_no_header(
        self, route: Route, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tunnel is loopback only, so a host that does not ask for a key is
        a legitimate configuration rather than a mistake."""
        monkeypatch.delenv("CHATGPT_PROXY_KEY", raising=False)
        seen: list[httpx.Headers] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return ok(sse("Một."))

        async with Client(transport=httpx.MockTransport(handle)) as api:
            await api.complete(route, "x")
        assert "authorization" not in seen[0]


@pytest.mark.asyncio
class TestEmptyAnswer:
    async def test_a_200_with_nothing_in_it_is_an_answer(self, route: Route) -> None:
        """The case that decided the shape of this module. A browser backed
        session does this often enough that raising would make the exceptional
        path the common one."""
        async with Client(transport=transport(ok(sse()))) as api:
            answer = await api.complete(route, "x")
        assert answer.empty
        assert answer.text == ""

    async def test_an_empty_answer_is_not_retried_inside_the_call(
        self, route: Route, clock: FakeClock
    ) -> None:
        """Asking the same logged out session twice in a row gets the same
        nothing twice in a row. Failing over is the router's job."""
        async with Client(transport=transport(ok(sse())), clock=clock) as api:
            await api.complete(route, "x")
        assert clock.slept == []

    async def test_whitespace_only_counts_as_empty(self, route: Route) -> None:
        async with Client(transport=transport(ok(sse(" ", "\n")))) as api:
            assert (await api.complete(route, "x")).empty


@pytest.mark.asyncio
class TestRetries:
    async def test_a_503_is_retried_on_the_same_route(self, route: Route, clock: FakeClock) -> None:
        api = Client(
            transport=transport(httpx.Response(503), ok(sse("Một."))), clock=clock, jitter=False
        )
        async with api:
            answer = await api.complete(route, "x")
        assert answer.text == "Một."
        assert answer.attempts == 2

    async def test_a_dropped_connection_is_retried(self, route: Route, clock: FakeClock) -> None:
        api = Client(
            transport=transport(httpx.ConnectError("tunnel gone"), ok(sse("Một."))),
            clock=clock,
            jitter=False,
        )
        async with api:
            assert (await api.complete(route, "x")).attempts == 2

    async def test_the_delay_doubles(self, route: Route, clock: FakeClock) -> None:
        api = Client(transport=transport(*[httpx.Response(503)] * 3), clock=clock, jitter=False)
        async with api:
            with pytest.raises(FleetError):
                await api.complete(route, "x")
        assert clock.slept == [30.0, 60.0]

    async def test_a_400_is_not_retried(self, route: Route, clock: FakeClock) -> None:
        """A bad request is our fault, and asking again produces the same bad
        request while spending two more minutes of a nine hour budget."""
        api = Client(transport=transport(httpx.Response(400)), clock=clock)
        async with api:
            with pytest.raises(FleetError, match="HTTP 400"):
                await api.complete(route, "x")
        assert clock.slept == []

    async def test_giving_up_says_what_the_last_failure_was(
        self, route: Route, clock: FakeClock
    ) -> None:
        api = Client(
            transport=transport(*[httpx.Response(502)] * 3), clock=clock, retries=2, jitter=False
        )
        async with api:
            with pytest.raises(FleetError, match="HTTP 502 after 3 attempts"):
                await api.complete(route, "x")

    async def test_the_last_attempt_does_not_say_it_is_retrying(
        self, route: Route, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It said "retrying on the same route" on the attempt it gave up on,
        which is how a route that never answers looked like one that was busy."""
        api = Client(transport=transport(httpx.Response(502)), clock=clock, retries=0)
        with caplog.at_level(logging.WARNING):
            async with api:
                with pytest.raises(FleetError):
                    await api.complete(route, "x")
        assert [record.getMessage() for record in caplog.records] == ["call failed, giving up"]

    async def test_an_attempt_that_will_be_repeated_says_so(
        self, route: Route, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        api = Client(
            transport=transport(httpx.Response(502), ok(sse("Một."))),
            clock=clock,
            retries=1,
            jitter=False,
        )
        with caplog.at_level(logging.WARNING):
            async with api:
                await api.complete(route, "x")
        assert [record.getMessage() for record in caplog.records] == [
            "call failed, retrying on the same route"
        ]

    async def test_jitter_keeps_the_delay_in_a_sane_band(
        self, route: Route, clock: FakeClock
    ) -> None:
        """Two workers that failed on the same dropped tunnel otherwise come
        back at the same instant and drop it again."""
        api = Client(transport=transport(*[httpx.Response(503)] * 3), clock=clock, retries=2)
        async with api:
            with pytest.raises(FleetError):
                await api.complete(route, "x")
        assert all(15.0 <= slept <= 90.0 for slept in clock.slept)


@pytest.mark.asyncio
class TestTimeout:
    async def test_a_route_that_never_answers_gives_up(self, clock: FakeClock) -> None:
        async def hang(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("nothing came back")

        api = Client(transport=httpx.MockTransport(hang), clock=clock, retries=0)
        async with api:
            with pytest.raises(FleetError, match="ReadTimeout"):
                await api.complete(make_route(timeout=1.0), "x")

    async def test_a_route_still_talking_after_its_deadline_is_cut_off(
        self, clock: FakeClock
    ) -> None:
        """A stream that trickles forever would otherwise hold a slot for the
        whole run. The route timeout is the outer bound on one call."""

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return ok(sse("Một."))

        api = Client(transport=httpx.MockTransport(slow), clock=clock, retries=0)
        async with api:
            with pytest.raises(FleetError, match="no answer within"):
                await api.complete(make_route(timeout=0.01), "x")


@pytest.mark.asyncio
class TestRealClock:
    async def test_the_clock_is_the_wall_clock(self) -> None:
        """Not the loop clock. A lease deadline is written into a file that a
        different process reads after a Ctrl-C, and a loop clock starts again
        from roughly zero in that process."""
        assert client.RealClock().now() == pytest.approx(time.time(), abs=5.0)

    async def test_sleeping_actually_yields(self) -> None:
        await client.RealClock().sleep(0.0)


@pytest.mark.asyncio
class TestHealth:
    async def test_a_route_that_answers_200(self, route: Route) -> None:
        async with Client(transport=transport(httpx.Response(200))) as api:
            assert await api.health(route)

    async def test_a_route_that_answers_503(self, route: Route) -> None:
        async with Client(transport=transport(httpx.Response(503))) as api:
            assert not await api.health(route)

    async def test_a_route_that_does_not_answer_at_all(self, route: Route) -> None:
        async with Client(transport=transport(httpx.ConnectError("no tunnel"))) as api:
            assert not await api.health(route)


@pytest.mark.asyncio
class TestStreamParsing:
    async def test_a_malformed_line_is_skipped_rather_than_raised_on(self, route: Route) -> None:
        """Half an answer is worth more than none. The invariants downstream
        refuse it if it is not usable."""
        body = "data: {not json\n\n" + sse("Một.")
        async with Client(transport=transport(ok(body))) as api:
            assert (await api.complete(route, "x")).text == "Một."

    async def test_lines_that_are_not_events_are_ignored(self, route: Route) -> None:
        body = ": keep-alive\n\n" + sse("Một.")
        async with Client(transport=transport(ok(body))) as api:
            assert (await api.complete(route, "x")).text == "Một."

    async def test_the_opening_chunk_carries_a_role_and_no_content(self, route: Route) -> None:
        """Every OpenAI-shaped stream starts with a delta that announces the
        assistant and says nothing. Treating it as content puts an empty string
        at the front of the answer."""
        opening = {"choices": [{"delta": {"role": "assistant"}}]}
        body = f"data: {json.dumps(opening)}\n\n" + sse("Một.")
        async with Client(transport=transport(ok(body))) as api:
            assert (await api.complete(route, "x")).text == "Một."

    async def test_a_chunk_can_carry_content_and_usage_at_once(self, route: Route) -> None:
        event = {
            "choices": [{"delta": {"content": "Một."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        async with Client(transport=transport(ok(body))) as api:
            answer = await api.complete(route, "x")
        assert answer.text == "Một."
        assert answer.usage.prompt_tokens == 10


class TestReportedUsage:
    @pytest.mark.parametrize(
        "reported", [None, {}, {"prompt_tokens": None}, {"prompt_tokens": "x"}, {"other": 1}]
    )
    def test_usage_a_host_reported_badly_is_not_a_crash(
        self, reported: dict[str, object] | None
    ) -> None:
        """Hosts in this fleet have reported usage as a string, as null and not
        at all. A wrong estimate beats a traceback on hour nine."""
        assert Usage.from_payload(reported).total == 0

    def test_a_count_that_arrived_as_a_string(self) -> None:
        assert Usage.from_payload({"prompt_tokens": "900"}).prompt_tokens == 900


@pytest.mark.asyncio
class TestLifecycle:
    async def test_the_client_refuses_to_be_used_unopened(self, route: Route) -> None:
        with pytest.raises(RuntimeError, match="outside its context manager"):
            await Client().complete(route, "x")

    async def test_closing_twice_is_harmless(self) -> None:
        api = Client(transport=transport())
        async with api:
            pass
        await api.__aexit__()

    async def test_an_answer_reads_as_a_sentence(self, route: Route) -> None:
        async with Client(transport=transport(ok(sse("Một.")))) as api:
            assert "chars in" in str(await api.complete(route, "x"))

    async def test_an_empty_answer_says_so(self, route: Route) -> None:
        async with Client(transport=transport(ok(sse()))) as api:
            assert "empty" in str(await api.complete(route, "x"))


class TestRedaction:
    def test_a_key_shaped_string_never_reaches_a_terminal(self) -> None:
        """The keys here are shared across hosts, so one of them in one pasted
        log is one of them everywhere."""
        assert "sk-" not in client.redact(f"failed with {FAKE_KEY} in the header")

    def test_a_configured_key_is_removed_even_if_it_looks_different(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHATGPT_PROXY_KEY", "notevenkeyshaped")
        assert client.redact("saw notevenkeyshaped", [make_route()]) == "saw ***"

    def test_a_route_with_no_key_set_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CHATGPT_PROXY_KEY", raising=False)
        assert client.redact("nothing to remove", [make_route()]) == "nothing to remove"

    def test_ordinary_text_is_left_alone(self) -> None:
        assert client.redact("connection reset by peer") == "connection reset by peer"
