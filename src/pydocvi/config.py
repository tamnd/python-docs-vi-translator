"""Where things live.

Paths only, resolved once, overridable by environment variable so that a test
never touches a real checkout and a run can be pointed at a scratch copy.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "PYDOCVI_"

#: The Transifex mirror. Read only, always. A commit pushed here is overwritten
#: within the hour, and more to the point, PEP 545 wants a person to have agreed
#: with a string before it lands there.
DEFAULT_UPSTREAM = Path.home() / "github" / "tamnd" / "python-docs-vi"

#: Where translated catalogs are written.
DEFAULT_CONTENT = Path.home() / "github" / "tamnd" / "python-docs-vi-machine-translation"

#: Run state: the memory, the queue, traces. Never committed.
DEFAULT_WORK = Path.cwd() / "work"


@dataclass(frozen=True, slots=True, kw_only=True)
class Paths:
    upstream: Path
    content: Path
    work: Path

    @property
    def memory(self) -> Path:
        return self.work / "memory.json"

    @property
    def manifests(self) -> Path:
        return self.work / "manifests"

    @property
    def upstream_pin(self) -> Path:
        return self.manifests / "upstream.yaml"

    @property
    def reports(self) -> Path:
        return self.work / "reports"

    @property
    def queue(self) -> Path:
        return self.work / "queue"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(f"{ENV_PREFIX}{name}")
    return Path(raw).expanduser() if raw else default


def paths() -> Paths:
    """Resolve the paths for this run."""
    return Paths(
        upstream=_env_path("UPSTREAM", DEFAULT_UPSTREAM),
        content=_env_path("CONTENT", DEFAULT_CONTENT),
        work=_env_path("WORK", DEFAULT_WORK),
    )
