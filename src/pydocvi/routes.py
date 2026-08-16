"""Routes: what to call, in what order, and what to do when one stops working.

A route is one tunnelled endpoint on one host. There are two of them and there
will never be twenty, so everything here is a list comprehension over a handful
of objects rather than a scheduler.

The route file lives outside the repository and names an environment variable
for its key rather than holding one. A file that holds a key is a file nobody
can paste into an issue, and the first time somebody needs help with a route the
thing they will want to paste is the route file.
"""

import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import platformdirs

log = logging.getLogger(__name__)

#: Cooldown after a failure, doubling on each consecutive failure up to the cap.
BASE_COOLDOWN = 300.0
MAX_COOLDOWN = 3600.0

#: What a route file that does not say otherwise gets.
DEFAULT_TIMEOUT = 600.0
DEFAULT_CONCURRENCY = 1

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smh]?)$")
_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}


class RouteError(Exception):
    """A route file that cannot be used as written."""


def config_file() -> Path:
    """Where the route file lives. Never inside the repository."""
    override = os.environ.get("PYDOCVI_ROUTES")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_dir("pydocvi")) / "routes.json"


def duration(value: str | float) -> float:
    """Seconds from ``20m``, ``30s``, ``1h`` or a bare number.

    Written as ``20m`` in the route file because a timeout measured in minutes
    should read as minutes. A four-digit number of seconds in a config file is
    read wrong by somebody eventually.
    """
    if isinstance(value, int | float):
        return float(value)
    match = _DURATION.match(value.strip())
    if match is None:
        raise RouteError(f"unreadable duration {value!r}, expected a number with s, m or h")
    return float(match.group(1)) * _UNITS[match.group(2)]


@dataclass(frozen=True, slots=True, kw_only=True)
class Route:
    """One endpoint.

    ``concurrency`` is a measurement from ``fleet bench``, not an aspiration.
    Each unit of it is one verified browser profile on the host, and asking for
    more than there are profiles is how a run earns a rate-limit cooldown that
    outlasts the run.
    """

    name: str
    base_url: str
    model: str
    host: str
    api_key_env: str = "CHATGPT_PROXY_KEY"
    remote_port: int = 8077
    local_port: int = 0
    rank: int = 0
    concurrency: int = DEFAULT_CONCURRENCY
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> Self:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise RouteError(f"unknown field(s) in route: {', '.join(unknown)}")
        for required in ("name", "base_url", "model", "host"):
            if not payload.get(required):
                raise RouteError(f"route is missing {required}")
        values = dict(payload)
        values["timeout"] = duration(values.get("timeout", DEFAULT_TIMEOUT))  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"

    @property
    def key(self) -> str | None:
        """The key for this route, read from the environment at call time.

        Read at call time rather than at load time so that a key never sits in
        a long-lived object that something might print.
        """
        return os.environ.get(self.api_key_env)

    def __str__(self) -> str:
        return f"{self.name} ({self.host}:{self.remote_port} via {self.base_url})"


@dataclass(slots=True)
class Health:
    """What has happened to one route lately.

    Mutable and deliberately so. It is per-process state about the world, not
    part of the domain model, and freezing it would mean rebuilding the router
    on every failure.
    """

    route: Route
    failures: int = 0
    cooling_until: float = 0.0
    calls: int = 0
    reason: str = ""

    def cooling(self, now: float) -> bool:
        return now < self.cooling_until

    def remaining(self, now: float) -> float:
        return max(0.0, self.cooling_until - now)

    def cool(self, now: float, reason: str) -> float:
        """Put the route into cooldown and return how long for.

        Doubling from five minutes to an hour. The failures that put a route
        here are a logged-out session or an exhausted daily quota, and neither
        of those is fixed by asking again in thirty seconds.
        """
        self.failures += 1
        self.reason = reason
        wait = float(min(BASE_COOLDOWN * 2 ** (self.failures - 1), MAX_COOLDOWN))
        self.cooling_until = now + wait
        log.warning(
            "route cooling down",
            extra={"route": self.route.name, "seconds": wait, "reason": reason},
        )
        return wait

    def succeed(self) -> None:
        """Clear the failure history.

        A route that works is fully forgiven rather than half-forgiven. Keeping
        a decaying penalty would mean a route that fails once an hour drifts
        into a permanent hour-long cooldown without anything having got worse.
        """
        self.failures = 0
        self.cooling_until = 0.0
        self.reason = ""
        self.calls += 1


@dataclass(slots=True)
class Router:
    """Routes in rank order, with their health.

    ``rank`` is the order to try things in and not a quality score. The fastest
    host being rank 0 is a coincidence of this fleet, not a rule.
    """

    health: dict[str, Health] = field(default_factory=dict)

    @classmethod
    def of(cls, routes: Sequence[Route]) -> Self:
        ordered = sorted((r for r in routes if r.enabled), key=lambda r: (r.rank, r.name))
        return cls(health={route.name: Health(route=route) for route in ordered})

    def __len__(self) -> int:
        return len(self.health)

    def __iter__(self) -> Iterator[Health]:
        return iter(self.health.values())

    @property
    def routes(self) -> list[Route]:
        return [state.route for state in self.health.values()]

    def available(self, now: float) -> list[Route]:
        """Routes that could take a job right now, best first."""
        return [state.route for state in self.health.values() if not state.cooling(now)]

    def pick(self, now: float, *, skip: Sequence[str] = ()) -> Route | None:
        """The best route not cooling and not already tried for this job."""
        for state in self.health.values():
            if state.route.name not in skip and not state.cooling(now):
                return state.route
        return None

    def cool(self, name: str, now: float, reason: str) -> float:
        return self.health[name].cool(now, reason)

    def succeed(self, name: str) -> None:
        self.health[name].succeed()

    def next_free(self, now: float) -> float:
        """Seconds until the earliest route comes out of cooldown.

        Zero when something is available now. The worker sleeps on this rather
        than spinning, because a worker that polls a fleet in cooldown produces
        a log with ten thousand lines and no information in it.
        """
        if not self.health:
            return 0.0
        if self.available(now):
            return 0.0
        return min(state.remaining(now) for state in self.health.values())

    def all_cooling(self, now: float) -> bool:
        return bool(self.health) and not self.available(now)


def load(path: Path | None = None) -> list[Route]:
    """Read the route file.

    A missing file is an error with the path in it rather than an empty list,
    because a run that silently finds no routes looks exactly like a run that
    finished with nothing to do.
    """
    target = path or config_file()
    if not target.exists():
        raise RouteError(f"no route file at {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RouteError(f"{target} is not valid JSON: {error}") from error

    rows = payload.get("routes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise RouteError(f"{target} has no routes")

    routes = [Route.from_json(row) for row in rows]
    names = [route.name for route in routes]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise RouteError(f"duplicate route name(s): {', '.join(duplicated)}")
    ports = [route.local_port for route in routes if route.local_port]
    clashing = sorted({port for port in ports if ports.count(port) > 1})
    if clashing:
        raise RouteError(f"two routes share local port(s): {clashing}")
    return routes


def missing_keys(routes: Sequence[Route]) -> list[str]:
    """The environment variables the routes need and the shell does not have."""
    return sorted({route.api_key_env for route in routes if not route.key})
