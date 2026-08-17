"""Where things live.

Paths only, resolved once, overridable by environment variable so that a test
never touches a real checkout and a run can be pointed at a scratch copy.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "PYDOCVI_"

#: The only language this tool translates into. Named rather than assumed,
#: because ``pydocvi audit --lang`` exists in the runbook and a flag that
#: silently ignored what it was given would be worse than one that refuses.
LANGUAGE = "vi"

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
        """Run artefacts: the last bench, the last curation. Never committed."""
        return self.work / "reports"

    @property
    def published(self) -> Path:
        """The committed reports, which are the project's public face.

        On the content repo rather than under ``work`` because they are the
        argument that the corpus is worth reviewing, and because ``A01``,
        ``A02``, ``A05`` and ``H06`` all check the corpus against what these
        files claim. A report that lived in a scratch directory would be checked
        against nothing on anybody else's machine.
        """
        return self.content / "reports"

    @property
    def tallies(self) -> Path:
        """One file per translation run, holding its refusal counts.

        Under ``work`` because it is run state, and kept per run rather than
        overwritten because a tier is translated over several sittings and the
        question the quality report answers is what the tier cost.
        """
        return self.work / "tallies"

    @property
    def queue(self) -> Path:
        return self.work / "queue"

    @property
    def glossary(self) -> Path:
        """The machine-readable terminology contract.

        On the content repo rather than under ``work`` because it is reviewed,
        committed and cited in every prompt. Run state is disposable and this is
        not.
        """
        return self.content / "manifests" / "glossary.yaml"

    @property
    def candidates(self) -> Path:
        return self.content / "manifests" / "glossary-candidates.yaml"

    @property
    def proposal(self) -> Path:
        """What curation produced, before a person has read it.

        A separate file from both the candidates and the glossary, because the
        human pass happens here and neither overwriting the mining output nor
        writing unreviewed rows into the contract is acceptable.
        """
        return self.content / "manifests" / "glossary-proposed.yaml"

    @property
    def versions(self) -> Path:
        """One file per past version, so ``glossary diff 6 7`` has two sides."""
        return self.content / "manifests" / "glossary"

    @property
    def glossary_markdown(self) -> Path:
        return self.content / "GLOSSARY.md"


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
