import json
from pathlib import Path

import pytest

from conftest import FAKE_KEY, make_route
from pydocvi import routes
from pydocvi.routes import BASE_COOLDOWN, MAX_COOLDOWN, Route, RouteError, Router

FILE = {
    "routes": [
        {
            "name": "server3",
            "base_url": "http://127.0.0.1:8103/v1",
            "model": "gpt-5",
            "host": "server3",
            "local_port": 8103,
            "rank": 0,
            "concurrency": 4,
            "timeout": "20m",
        },
        {
            "name": "server1",
            "base_url": "http://127.0.0.1:8101/v1",
            "model": "gpt-5",
            "host": "server1",
            "local_port": 8101,
            "rank": 1,
            "concurrency": 1,
            "timeout": "20m",
        },
    ]
}


def write(tmp_path: Path, payload: object) -> Path:
    target = tmp_path / "routes.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


class TestDuration:
    @pytest.mark.parametrize(
        ("written", "seconds"), [("30s", 30), ("20m", 1200), ("1h", 3600), ("45", 45)]
    )
    def test_the_units_a_route_file_uses(self, written: str, seconds: float) -> None:
        assert routes.duration(written) == seconds

    def test_a_number_is_already_seconds(self) -> None:
        assert routes.duration(90.0) == 90.0

    def test_something_unreadable_says_so(self) -> None:
        with pytest.raises(RouteError, match="unreadable duration"):
            routes.duration("soon")


class TestLoading:
    def test_a_route_file_of_the_documented_shape(self, tmp_path: Path) -> None:
        loaded = routes.load(write(tmp_path, FILE))
        assert [route.name for route in loaded] == ["server3", "server1"]
        assert loaded[0].timeout == 1200.0

    def test_a_bare_list_is_accepted_too(self, tmp_path: Path) -> None:
        assert len(routes.load(write(tmp_path, FILE["routes"]))) == 2

    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(RouteError, match="no route file at"):
            routes.load(tmp_path / "absent.json")

    def test_a_file_that_is_not_json(self, tmp_path: Path) -> None:
        target = tmp_path / "routes.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(RouteError, match="not valid JSON"):
            routes.load(target)

    def test_an_empty_file_is_an_error_rather_than_no_routes(self, tmp_path: Path) -> None:
        """A run that silently finds nothing to call looks exactly like a run
        that finished with nothing to do."""
        with pytest.raises(RouteError, match="no routes"):
            routes.load(write(tmp_path, {"routes": []}))

    def test_a_typo_in_a_field_name_is_refused(self, tmp_path: Path) -> None:
        """Silently ignoring an unknown field means a route file that says
        concurrency: 4 under a misspelled key runs at one and nobody notices."""
        broken = {"routes": [{**FILE["routes"][0], "concurency": 4}]}
        with pytest.raises(RouteError, match="unknown field"):
            routes.load(write(tmp_path, broken))

    def test_a_missing_required_field(self, tmp_path: Path) -> None:
        without = {k: v for k, v in FILE["routes"][0].items() if k != "model"}
        with pytest.raises(RouteError, match="missing model"):
            routes.load(write(tmp_path, {"routes": [without]}))

    def test_two_routes_with_the_same_name(self, tmp_path: Path) -> None:
        with pytest.raises(RouteError, match="duplicate route name"):
            routes.load(write(tmp_path, {"routes": [FILE["routes"][0]] * 2}))

    def test_two_routes_on_the_same_local_port(self, tmp_path: Path) -> None:
        """Both tunnels would come up and one of them would be forwarding
        somewhere nobody asked for."""
        second = {**FILE["routes"][1], "local_port": 8103}
        with pytest.raises(RouteError, match="share local port"):
            routes.load(write(tmp_path, {"routes": [FILE["routes"][0], second]}))


class TestKeys:
    def test_no_field_can_hold_a_literal_key(self, tmp_path: Path) -> None:
        """The route file is the thing somebody pastes into an issue when a
        route stops working."""
        with pytest.raises(RouteError, match="unknown field"):
            routes.load(write(tmp_path, {"routes": [{**FILE["routes"][0], "api_key": "sk-x"}]}))

    def test_the_key_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHATGPT_PROXY_KEY", FAKE_KEY)
        assert make_route().key == FAKE_KEY

    def test_an_unset_variable_is_reported_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CHATGPT_PROXY_KEY", raising=False)
        assert routes.missing_keys([make_route()]) == ["CHATGPT_PROXY_KEY"]

    def test_the_config_path_is_outside_the_repository(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PYDOCVI_ROUTES", raising=False)
        assert routes.config_file().name == "routes.json"
        assert Path.cwd() not in routes.config_file().parents

    def test_the_path_can_be_pointed_somewhere_else(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYDOCVI_ROUTES", "/tmp/elsewhere.json")
        assert routes.config_file() == Path("/tmp/elsewhere.json")


class TestRouter:
    def test_routes_come_out_in_rank_order(self) -> None:
        router = Router.of([make_route("b", rank=1), make_route("a", rank=0)])
        assert [route.name for route in router.routes] == ["a", "b"]

    def test_a_disabled_route_is_not_in_the_router(self) -> None:
        assert len(Router.of([make_route("a", enabled=False)])) == 0

    def test_the_best_available_route_is_picked(self) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        picked = router.pick(now=0.0)
        assert picked is not None
        assert picked.name == "a"

    def test_a_route_already_tried_for_this_job_is_skipped(self) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        picked = router.pick(now=0.0, skip=["a"])
        assert picked is not None
        assert picked.name == "b"

    def test_a_cooling_route_is_not_picked(self) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        router.cool("a", 0.0, "empty answer")
        picked = router.pick(now=1.0)
        assert picked is not None
        assert picked.name == "b"

    def test_nothing_to_pick_when_every_route_is_cooling(self) -> None:
        router = Router.of([make_route("a")])
        router.cool("a", 0.0, "down")
        assert router.pick(now=1.0) is None
        assert router.all_cooling(now=1.0)

    def test_a_router_with_no_routes_is_not_all_cooling(self) -> None:
        """An empty fleet is a configuration problem, not a fleet that is
        resting, and the two need different messages."""
        assert not Router.of([]).all_cooling(now=0.0)
        assert Router.of([]).next_free(now=0.0) == 0.0


class TestCooldown:
    def test_the_first_failure_costs_five_minutes(self) -> None:
        router = Router.of([make_route("a")])
        assert router.cool("a", 0.0, "empty") == BASE_COOLDOWN

    def test_consecutive_failures_double(self) -> None:
        router = Router.of([make_route("a")])
        waits = [router.cool("a", 0.0, "empty") for _ in range(4)]
        assert waits == [300.0, 600.0, 1200.0, 2400.0]

    def test_the_doubling_stops_at_an_hour(self) -> None:
        router = Router.of([make_route("a")])
        waits = [router.cool("a", 0.0, "empty") for _ in range(12)]
        assert waits[-1] == MAX_COOLDOWN
        assert max(waits) == MAX_COOLDOWN

    def test_a_success_forgives_completely(self) -> None:
        """A half-forgiven route that fails once an hour drifts into a permanent
        hour-long cooldown without anything actually having got worse."""
        router = Router.of([make_route("a")])
        router.cool("a", 0.0, "empty")
        router.cool("a", 0.0, "empty")
        router.succeed("a")
        assert router.cool("a", 0.0, "empty") == BASE_COOLDOWN

    def test_a_route_comes_back_when_its_cooldown_passes(self) -> None:
        router = Router.of([make_route("a")])
        router.cool("a", 0.0, "empty")
        assert router.available(now=BASE_COOLDOWN + 1) == router.routes

    def test_next_free_is_how_long_the_worker_should_sleep(self) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        router.cool("a", 0.0, "empty")
        router.cool("b", 0.0, "empty")
        router.cool("b", 0.0, "empty")
        assert router.next_free(now=0.0) == BASE_COOLDOWN

    def test_next_free_is_zero_when_something_is_available(self) -> None:
        router = Router.of([make_route("a", rank=0), make_route("b", rank=1)])
        router.cool("a", 0.0, "empty")
        assert router.next_free(now=0.0) == 0.0

    def test_the_reason_is_kept_for_the_status_line(self) -> None:
        router = Router.of([make_route("a")])
        router.cool("a", 0.0, "daily quota")
        assert router.health["a"].reason == "daily quota"


class TestRoute:
    def test_the_health_url_hangs_off_the_base_url(self) -> None:
        assert make_route().health_url == "http://127.0.0.1:8103/v1/health"

    def test_a_trailing_slash_does_not_double_up(self) -> None:
        assert (
            Route(name="a", base_url="http://x/v1/", model="m", host="h").health_url
            == "http://x/v1/health"
        )

    def test_a_route_reads_as_a_sentence(self) -> None:
        assert "server3" in str(make_route())
