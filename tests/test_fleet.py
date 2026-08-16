import pytest

from conftest import FAKE_KEY, FakeClient, FakeClock, FakeRunner, make_route
from pydocvi import fleet as fleetmod
from pydocvi.client import FleetError
from pydocvi.fleet import Bench, Diagnosis, Fleet, Ran, Tunnel, bench_markdown, hours_for

LSOF = "lsof -t -i @127.0.0.1"
HEALTH = "curl"


def listening(pid: str = "4242") -> dict[str, Ran]:
    return {LSOF: Ran(code=0, out=f"{pid}\n")}


def free() -> dict[str, Ran]:
    """lsof exits 1 and says nothing when nobody holds the port."""
    return {LSOF: Ran(code=1)}


def answering() -> dict[str, Ran]:
    return {HEALTH: Ran(code=0, out="200")}


def make_fleet(*answers: dict[str, Ran]) -> tuple[Fleet, FakeRunner]:
    merged: dict[str, Ran] = {}
    for answer in answers:
        merged.update(answer)
    runner = FakeRunner(merged)
    return Fleet([make_route("a")], runner=runner), runner


class TestBringingATunnelUp:
    def test_a_forward_is_opened_on_the_configured_ports(self) -> None:
        fleet, runner = make_fleet(free())
        tunnel = fleet.up(make_route("a", local_port=8103, remote_port=8080))
        assert tunnel.up
        assert runner.ran("-L 8103:127.0.0.1:8080")

    def test_the_flag_that_is_not_optional(self) -> None:
        """Without ExitOnForwardFailure a tunnel whose local port is already
        taken comes up reporting success, and every request through it goes to
        whatever else is listening. That looks exactly like a model problem."""
        fleet, runner = make_fleet(free())
        fleet.up(make_route("a"))
        assert runner.ran("ExitOnForwardFailure=yes")

    def test_ssh_is_never_left_waiting_for_a_passphrase(self) -> None:
        """A run started in the background and blocked on a prompt nobody can
        see is nine hours of nothing."""
        fleet, runner = make_fleet(free())
        fleet.up(make_route("a"))
        assert runner.ran("BatchMode=yes")

    def test_a_tunnel_already_up_is_left_alone(self) -> None:
        """Running fleet up twice is what everybody does, and the second run
        should not tear down a working tunnel to prove it can build one."""
        fleet, runner = make_fleet(listening(), answering())
        tunnel = fleet.up(make_route("a"))
        assert tunnel.up
        assert tunnel.detail == "already up"
        assert not runner.ran("ssh")

    def test_a_port_held_by_something_that_is_not_the_tunnel(self) -> None:
        """The failure this whole module exists to name. Something is listening
        on the port and it is not the proxy."""
        fleet, _ = make_fleet(listening(), {HEALTH: Ran(code=0, out="404")})
        tunnel = fleet.up(make_route("a", local_port=8103))
        assert not tunnel.up
        assert "taken by something else" in tunnel.detail

    def test_ssh_failing_is_reported_rather_than_raised(self) -> None:
        fleet, _ = make_fleet(free(), {"ssh": Ran(code=255, err="Permission denied (publickey).")})
        tunnel = fleet.up(make_route("a"))
        assert not tunnel.up
        assert "Permission denied" in tunnel.detail

    def test_a_key_in_an_ssh_error_never_reaches_the_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHATGPT_PROXY_KEY", FAKE_KEY)
        fleet, _ = make_fleet(free(), {"ssh": Ran(code=255, err=f"failed: {FAKE_KEY}")})
        assert "sk-" not in fleet.up(make_route("a")).detail


class TestTakingATunnelDown:
    def test_whatever_holds_the_port_is_killed(self) -> None:
        fleet, runner = make_fleet(listening("4242"))
        assert fleet.down(make_route("a"))
        assert runner.ran("kill 4242")

    def test_nothing_to_close_is_not_a_failure(self) -> None:
        fleet, runner = make_fleet(free())
        assert not fleet.down(make_route("a"))
        assert not runner.ran("kill")

    def test_every_holder_of_the_port_is_killed(self) -> None:
        fleet, runner = make_fleet({LSOF: Ran(code=0, out="4242\n4243\n")})
        fleet.down(make_route("a"))
        assert runner.ran("kill 4242")
        assert runner.ran("kill 4243")

    def test_lsof_output_that_is_not_a_pid_is_ignored(self) -> None:
        fleet, _ = make_fleet({LSOF: Ran(code=0, out="lsof: WARNING: something\n")})
        assert not fleet.down(make_route("a"))


class TestStatus:
    def test_every_route_is_reported(self) -> None:
        fleet = Fleet([make_route("a"), make_route("b")], runner=FakeRunner(listening()))
        assert [tunnel.up for tunnel in fleet.status()] == [True, True]

    def test_a_tunnel_that_is_down(self) -> None:
        fleet, _ = make_fleet(free())
        assert not fleet.status()[0].up

    def test_a_tunnel_reads_as_a_sentence(self) -> None:
        line = str(Tunnel(route="a", host="h", local_port=8103, remote_port=8080, up=True))
        assert line == "a: up 127.0.0.1:8103 -> h:8080"


class TestProbe:
    def test_a_host_that_answers_health(self) -> None:
        fleet, _ = make_fleet(answering())
        probe = fleet.probe(make_route("a"))
        assert probe.up
        assert probe.detail == "health 200"

    def test_the_probe_does_not_share_code_with_what_it_diagnoses(self) -> None:
        """curl on purpose. A diagnostic written in the same client that is
        being diagnosed cannot tell you the client is the problem."""
        fleet, runner = make_fleet(answering())
        fleet.probe(make_route("a"))
        assert runner.ran("curl")
        assert runner.ran("/v1/health")

    @pytest.mark.parametrize("code", ["000", "404", "502"])
    def test_anything_that_is_not_200_is_down(self, code: str) -> None:
        fleet, _ = make_fleet({HEALTH: Ran(code=0, out=code)})
        assert not fleet.probe(make_route("a")).up

    def test_curl_itself_failing(self) -> None:
        fleet, _ = make_fleet({HEALTH: Ran(code=7, err="Failed to connect")})
        probe = fleet.probe(make_route("a"))
        assert not probe.up
        assert "Failed to connect" in probe.detail


class TestTrace:
    def test_a_batch_is_found_and_fetched(self) -> None:
        runner = FakeRunner(
            {
                "grep -rl": Ran(code=0, out="/traces/2026-08-15/aa.json\n"),
                "cat ": Ran(code=0, out="prompt and reply"),
            }
        )
        fleet = Fleet([make_route("a")], runner=runner)
        assert fleet.trace(make_route("a"), "3f2a1c", day="2026-08-15") == "prompt and reply"
        assert runner.ran("grep -rl 3f2a1c")

    def test_a_batch_nobody_has_a_trace_for(self) -> None:
        fleet, _ = make_fleet({"grep -rl": Ran(code=0, out="")})
        assert fleet.trace(make_route("a"), "3f2a1c", day="2026-08-15") == ""

    def test_grep_failing_is_not_a_traceback(self) -> None:
        fleet, _ = make_fleet({"grep -rl": Ran(code=2, err="No such file or directory")})
        assert fleet.trace(make_route("a"), "3f2a1c", day="2026-08-15") == ""

    def test_a_key_in_a_trace_never_reaches_the_terminal(self) -> None:
        """Traces are the thing you paste into an issue, so this is the highest
        risk path in the project."""
        runner = FakeRunner(
            {
                "grep -rl": Ran(code=0, out="/traces/aa.json\n"),
                "cat ": Ran(code=0, out=f'{{"authorization": "Bearer {FAKE_KEY}"}}'),
            }
        )
        fetched = Fleet([make_route("a")], runner=runner).trace(
            make_route("a"), "3f2a1c", day="2026-08-15"
        )
        assert "sk-" not in fetched

    def test_a_batch_id_is_quoted_before_it_reaches_a_shell(self) -> None:
        """The id is content addressed and hexadecimal, so this is defence in
        depth rather than a live risk, but it is a shell command."""
        fleet, runner = make_fleet({"grep -rl": Ran(code=0, out="")})
        fleet.trace(make_route("a"), "; rm -rf /", day="2026-08-15")
        assert runner.ran("'; rm -rf /'")


class TestDiagnosis:
    def test_one_route_answering_is_enough_to_start(self) -> None:
        """Waiting for the whole fleet would mean not starting, most days."""
        diagnosis = Diagnosis(
            tunnels=[_tunnel("a", up=True), _tunnel("b", up=False)], missing_keys=[], cooling=[]
        )
        assert diagnosis.healthy
        assert diagnosis.summary == "1 of 2 routes answering"

    def test_a_missing_key_is_not_healthy_however_many_tunnels_are_up(self) -> None:
        diagnosis = Diagnosis(
            tunnels=[_tunnel("a", up=True)], missing_keys=["CHATGPT_PROXY_KEY"], cooling=[]
        )
        assert not diagnosis.healthy
        assert "CHATGPT_PROXY_KEY" in diagnosis.summary

    def test_no_routes_configured_says_so(self) -> None:
        assert Diagnosis(tunnels=[], missing_keys=[], cooling=[]).summary == "no routes configured"

    def test_every_tunnel_down_says_what_to_look_at(self) -> None:
        diagnosis = Diagnosis(tunnels=[_tunnel("a", up=False)], missing_keys=[], cooling=[])
        assert not diagnosis.healthy
        assert "no route answers" in diagnosis.summary


class TestEstimates:
    def test_a_measured_rate_becomes_a_number_of_hours(self) -> None:
        """The point of the milestone: no estimate is copied from the design
        notes, every one is computed from a rate somebody measured."""
        assert hours_for(2_776, 60.0) == pytest.approx(53.2, abs=0.1)

    def test_retries_are_in_the_estimate(self) -> None:
        """The number a person acts on has to be the one that says how long it
        will actually take."""
        assert hours_for(100, 100.0, retry_rate=0.0) == 1.0
        assert hours_for(100, 100.0, retry_rate=0.15) == pytest.approx(1.15)

    def test_a_fleet_that_has_not_been_measured_estimates_nothing(self) -> None:
        """Zero hours would read as instant rather than as unknown, so this is
        checked wherever the number is printed."""
        assert hours_for(2_776, 0.0) == 0.0

    def test_throughput_is_successes_over_wall_clock(self) -> None:
        result = Bench(route="a", calls=10, failures=1, empty=1, seconds=3600.0, concurrency=4)
        assert result.successes == 8
        assert result.calls_per_hour == 8.0
        assert result.average_seconds == 360.0

    def test_a_bench_that_made_no_calls(self) -> None:
        result = Bench(route="a", calls=0, failures=0, empty=0, seconds=0.0, concurrency=1)
        assert result.calls_per_hour == 0.0
        assert result.average_seconds == 0.0

    def test_the_report_totals_the_fleet(self) -> None:
        results = [
            Bench(route="a", calls=10, failures=0, empty=0, seconds=3600.0, concurrency=4),
            Bench(route="b", calls=5, failures=0, empty=0, seconds=3600.0, concurrency=1),
        ]
        report = bench_markdown(results, batches=2_776)
        assert "| a | 10 |" in report
        assert "**15.0**" in report
        assert "2,776 batches" in report


@pytest.mark.asyncio
class TestBench:
    async def test_a_route_is_measured_at_its_stated_concurrency(self) -> None:
        clock = FakeClock()
        result = await fleetmod.bench(
            FakeClient(), make_route("a", concurrency=4), calls=8, prompt="x", clock=clock
        )
        assert result.calls == 8
        assert result.successes == 8
        assert result.concurrency == 4

    async def test_failures_and_empties_are_counted_separately(self) -> None:
        """They mean different things. A failure is a transport problem and an
        empty answer is a session that needs a person to log in again."""
        client = FakeClient([FleetError("down"), "", "Một.", "Một."])
        result = await fleetmod.bench(
            client, make_route("a", concurrency=1), calls=4, prompt="x", clock=FakeClock()
        )
        assert (result.failures, result.empty, result.successes) == (1, 1, 2)

    async def test_a_bench_that_measured_nothing_does_not_divide_by_zero(self) -> None:
        result = await fleetmod.bench(
            FakeClient(), make_route("a"), calls=0, prompt="x", clock=FakeClock()
        )
        assert result.calls_per_hour == 0.0


class TestRan:
    def test_the_last_line_is_the_one_worth_printing(self) -> None:
        """ssh writes three lines of preamble before the line that says why."""
        ran = Ran(code=255, err="OpenSSH_9.0\ndebug: something\nPermission denied (publickey).")
        assert ran.message == "Permission denied (publickey)."

    def test_stdout_is_used_when_there_is_no_stderr(self) -> None:
        assert Ran(code=1, out="not found").message == "not found"

    def test_a_command_that_said_nothing(self) -> None:
        assert Ran(code=0).message == ""
        assert Ran(code=0).ok


class TestSubprocess:
    def test_a_program_that_is_not_installed(self) -> None:
        """127 rather than a traceback, so doctor can say which program is
        missing instead of dying while trying to."""
        ran = fleetmod.Subprocess().run(["pydocvi-no-such-program"])
        assert ran.code == 127
        assert not ran.ok

    def test_a_command_that_takes_too_long(self) -> None:
        ran = fleetmod.Subprocess().run(["sleep", "5"], timeout=0.1)
        assert ran.code == 124
        assert "timed out" in ran.err

    def test_a_command_that_works(self) -> None:
        assert fleetmod.Subprocess().run(["echo", "hello"]).out.strip() == "hello"


def _tunnel(name: str, *, up: bool) -> Tunnel:
    return Tunnel(route=name, host=name, local_port=8103, remote_port=8080, up=up)
