"""The durable queue.

``work/queue/<stage>/<state>/<job>.json``. A directory per state and a rename to
move between them, which is atomic on every filesystem this will ever run on.

Four rules, each of which is a class of bug that then does not happen.

**Leases, not locks.** A worker claims a job by renaming it into ``leased/``
with a deadline written inside. A worker that is killed holds nothing: the lease
expires and ``reap`` puts the job back. There is no lock file to clean up and no
process id to trust.

**Content-addressed ids.** The id is a hash of everything that determines the
answer, so re-running the pipeline produces the same ids and work already done
is skipped because its output is already on disk. There is no ledger to keep in
step with the filesystem, because the filesystem is the ledger.

**One side effect.** A job writes one file, through a temporary and a rename, so
a job that dies halfway leaves either the old output or the new one.

**Bounded attempts.** Three, then ``dead``, and the audit has to name it. A job
that retries forever is an entry that never reaches the corpus and never reaches
a report either.
"""

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
LEASE_SECONDS = 1800.0
ID_LENGTH = 16
_DOMAIN = b"pydocvi.job.1"


class Stage(StrEnum):
    """One queue per stage, so draining the judge does not touch translation."""

    TRANSLATE = "translate"
    JUDGE = "judge"
    ROUNDTRIP = "roundtrip"
    GLOSSARY = "glossary"
    COMMENTS = "comments"


class State(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    DONE = "done"
    DEAD = "dead"


def job_id(stage: Stage | str, payload: Mapping[str, object]) -> str:
    """A content address for a unit of work.

    Everything that changes the answer goes in: the file, the entries, the
    prompt and the glossary version. Change the prompt and every id changes,
    which is the mechanism by which a prompt change re-queues the corpus rather
    than being silently ignored because the outputs are already there.
    """
    material = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(_DOMAIN + b"\x00" + str(stage).encode() + b"\x00" + material.encode())
    return digest.hexdigest()[:ID_LENGTH]


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """A unit of work and its history.

    ``attempts`` counts claims rather than failures, so a worker that is killed
    mid-call spends one. That is the conservative direction: three claims that
    each died silently is a job that needs a person to look at it, not a fourth
    claim.
    """

    id: str
    stage: Stage
    payload: dict[str, object] = field(default_factory=dict)
    attempts: int = 0
    lease_expires: float = 0.0
    route: str | None = None
    error: str | None = None

    def as_json(self) -> str:
        return (
            json.dumps(
                {
                    "id": self.id,
                    "stage": str(self.stage),
                    "payload": self.payload,
                    "attempts": self.attempts,
                    "lease_expires": self.lease_expires,
                    "route": self.route,
                    "error": self.error,
                },
                ensure_ascii=False,
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> Self:
        payload = json.loads(text)
        return cls(
            id=payload["id"],
            stage=Stage(payload["stage"]),
            payload=payload.get("payload") or {},
            attempts=payload.get("attempts", 0),
            lease_expires=payload.get("lease_expires", 0.0),
            route=payload.get("route"),
            error=payload.get("error"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Stats:
    stage: Stage
    pending: int = 0
    leased: int = 0
    done: int = 0
    dead: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.leased + self.done + self.dead

    @property
    def outstanding(self) -> int:
        return self.pending + self.leased


class Queue:
    """One stage's queue, rooted at a directory.

    Every method is a rename or a write. Nothing here holds state between calls,
    so two workers on the same directory need no coordination beyond the atomic
    rename that ``claim`` does.
    """

    def __init__(self, root: Path, stage: Stage) -> None:
        self.stage = stage
        self.root = root / str(stage)

    def directory(self, state: State) -> Path:
        return self.root / str(state)

    def path(self, state: State, identifier: str) -> Path:
        return self.directory(state) / f"{identifier}.json"

    def locate(self, identifier: str) -> tuple[State, Path] | None:
        """Which state a job is in, if it is anywhere."""
        for state in State:
            candidate = self.path(state, identifier)
            if candidate.exists():
                return state, candidate
        return None

    def add(self, job: Job) -> bool:
        """Queue a job unless it is already known. Returns whether it was added.

        Already known includes ``done``, which is what makes a re-run cheap: the
        second pass over a tier queues only the batches the first pass did not
        finish.
        """
        if self.locate(job.id) is not None:
            return False
        _write(self.path(State.PENDING, job.id), job)
        return True

    def extend(self, jobs: Sequence[Job]) -> int:
        return sum(1 for job in jobs if self.add(job))

    def claim(self, now: float, *, lease: float = LEASE_SECONDS) -> Job | None:
        """Take the next pending job, or return None.

        The rename is the claim. Two workers racing on the same file means one
        of them gets ``FileNotFoundError`` and moves on to the next, which is
        why the loop catches it rather than checking first.
        """
        for source in sorted(self.directory(State.PENDING).glob("*.json")):
            try:
                job = Job.from_json(source.read_text(encoding="utf-8"))
            except OSError, json.JSONDecodeError:
                continue
            leased = Job(
                id=job.id,
                stage=job.stage,
                payload=job.payload,
                attempts=job.attempts + 1,
                lease_expires=now + lease,
                route=job.route,
                error=job.error,
            )
            target = self.path(State.LEASED, job.id)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                source.replace(target)
            except OSError:
                continue
            target.write_text(leased.as_json(), encoding="utf-8")
            return leased
        return None

    def finish(self, job: Job) -> None:
        """Move a leased job to done."""
        self._move(job, State.DONE)

    def release(self, job: Job, *, error: str | None = None) -> State:
        """Hand a job back after a failure.

        Back to pending if it has attempts left, otherwise dead. Three and then
        dead is deliberate: on this transport a fourth attempt has never fixed
        anything, and reading the trace has.
        """
        state = State.PENDING if job.attempts < MAX_ATTEMPTS else State.DEAD
        self._move(
            Job(
                id=job.id,
                stage=job.stage,
                payload=job.payload,
                attempts=job.attempts,
                lease_expires=0.0,
                route=job.route,
                error=error,
            ),
            state,
        )
        if state is State.DEAD:
            log.error("job dead", extra={"job": job.id, "attempts": job.attempts, "error": error})
        return state

    def reap(self, now: float) -> list[Job]:
        """Return every expired lease to the queue.

        This is what a worker killed on hour nine costs: the jobs it held come
        back the next time anything looks at the queue.
        """
        recovered = []
        for path in sorted(self.directory(State.LEASED).glob("*.json")):
            try:
                job = Job.from_json(path.read_text(encoding="utf-8"))
            except OSError, json.JSONDecodeError:
                continue
            if job.lease_expires > now:
                continue
            self.release(job, error="lease expired")
            recovered.append(job)
        if recovered:
            log.info("reaped expired leases", extra={"count": len(recovered)})
        return recovered

    def retry(self, *, dead: bool = True) -> int:
        """Move dead jobs back to pending with their attempt count cleared.

        A person's decision, never automatic. It is the command you run after
        fixing whatever the traces said was wrong.
        """
        if not dead:
            return 0
        moved = 0
        for job in self.jobs(State.DEAD):
            self._move(
                Job(id=job.id, stage=job.stage, payload=job.payload, attempts=0),
                State.PENDING,
            )
            moved += 1
        return moved

    def drain(self) -> int:
        """Delete everything that is finished. Never touches outstanding work."""
        removed = 0
        for path in self.directory(State.DONE).glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def jobs(self, state: State) -> list[Job]:
        out = []
        for path in sorted(self.directory(state).glob("*.json")):
            try:
                out.append(Job.from_json(path.read_text(encoding="utf-8")))
            except OSError, json.JSONDecodeError:
                log.warning("unreadable job file", extra={"path": str(path)})
        return out

    def count(self, state: State) -> int:
        directory = self.directory(state)
        return sum(1 for _ in directory.glob("*.json")) if directory.exists() else 0

    def stats(self) -> Stats:
        return Stats(
            stage=self.stage,
            pending=self.count(State.PENDING),
            leased=self.count(State.LEASED),
            done=self.count(State.DONE),
            dead=self.count(State.DEAD),
        )

    def __len__(self) -> int:
        return self.count(State.PENDING)

    def _move(self, job: Job, state: State) -> Path:
        target = self.path(state, job.id)
        _write(target, job)
        for other in State:
            if other is not state:
                self.path(other, job.id).unlink(missing_ok=True)
        return target


def _write(target: Path, job: Job) -> None:
    """One job file, atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pydocvi-tmp")
    temporary.write_text(job.as_json(), encoding="utf-8")
    temporary.replace(target)


def queues(root: Path) -> Iterator[Queue]:
    """Every stage's queue, whether or not it has been used yet."""
    for stage in Stage:
        yield Queue(root, stage)
