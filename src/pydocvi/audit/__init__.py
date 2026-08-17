"""The corpus audit: forty-one checks in six groups over committed catalogs.

Every check runs off files. Nothing here calls a model, opens a socket, reads a
key or touches the fleet, and that is a design constraint rather than an
accident of what has been written so far. CI runs this on every push to the
content repo, so it has to be fast, and a reviewer has to be able to believe a
green run without knowing which route happened to answer that day.

The groups run in the order they are composed below, because a structural
failure explains most of what follows it. An ``S02`` says a ``msgid`` was
edited, and the forty ``L02`` findings printed after it are that same edit seen
from downstream. Printing them first would waste the reader's first minute.
"""

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from pydocvi import apply, catalog, config, sync
from pydocvi.audit import availability, glossary, hygiene, language, placeholders, structure
from pydocvi.audit.model import (
    COUNTS,
    Body,
    Check,
    Corpus,
    Finding,
    Group,
    Registry,
    Report,
    Result,
    UnknownCheckError,
    counts,
    plural,
)
from pydocvi.glossary import load as load_glossary
from pydocvi.memory import Memory

__all__ = [
    "COUNTS",
    "REGISTRY",
    "Body",
    "Check",
    "Corpus",
    "Finding",
    "Group",
    "Registry",
    "Report",
    "Result",
    "UnknownCheckError",
    "assemble",
    "counts",
    "markdown",
    "plural",
    "run",
]

#: Every check there is, group by group, in report order.
REGISTRY = Registry()
for _group in (structure, placeholders, glossary, language, availability, hygiene):
    for _check in _group.registry.checks:
        REGISTRY.add(_check)


def assemble(where: config.Paths, *, branch: str = sync.DEFAULT_BRANCH) -> Corpus:
    """Read everything the checks are allowed to look at, once.

    Once, rather than per check. Forty-one checks each parsing 548 catalogs off
    a disk is forty-one times the slowest part of the run, and the parse is by
    a wide margin the slowest part.

    Anything that is not there is left as ``None`` rather than defaulted, and
    the checks that need it return without a finding. A missing glossary is not
    a glossary failure, and reporting it as forty thousand ``G02`` findings
    would bury the run.
    """
    upstream = {
        _relative(path, where.upstream): catalog.read(path) for path in catalog.walk(where.upstream)
    }
    return Corpus(
        root=where.content,
        catalogs=tuple(catalog.read(path) for path in catalog.walk(where.content)),
        upstream=upstream,
        memory=Memory.load(where.memory) if where.memory.exists() else None,
        glossary=load_glossary(where.glossary) if where.glossary.exists() else None,
        markdown=_text(where.glossary_markdown),
        tracked=_tracked(where.content),
        coverage=_text(where.published / "coverage.md"),
        quality=_text(where.published / "quality.md"),
        readme=_text(where.content / "README.md"),
        queue=where.queue if where.queue.exists() else None,
        pin=where.upstream_pin,
        upstream_root=where.upstream,
        stamp=apply.Stamp(
            project=f"Python {branch}",
            run=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
            generator="pydocvi audit",
        ),
    )


def run(
    corpus: Corpus,
    *,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
    fail_soft: bool = False,
) -> Report:
    """Run the selected checks and collect what they found."""
    selected = REGISTRY.selected(only=only, skip=skip)
    return Report(results=tuple(one.run(corpus) for one in selected), fail_soft=fail_soft)


def markdown(report: Report, *, limit: int = 20) -> str:
    """``reports/audit.md``: the verdict, a table, then the findings.

    Capped at ``limit`` findings per check, and the cap is announced rather than
    silently applied. A check with 4 000 findings has one problem, not 4 000,
    and a report that printed all of them would be a file nobody opens. A report
    that truncated without saying so would be worse: it would read as though the
    problem were twenty entries wide.
    """
    lines = [
        "# Audit",
        "",
        f"{'Pass' if report.ok else 'Fail'}. "
        f"{plural(len(report.results), 'check')}, {len(report.failing())} failing, "
        f"{plural(len(report.findings), 'finding')}.",
        "",
    ]
    for group in Group:
        results = report.of_group(group)
        if not results:
            continue
        lines += [
            f"## {group.value.title()}",
            "",
            "| Check | Hard | Findings | What it checks |",
            "| --- | --- | --- | --- |",
        ]
        for one in results:
            hard = "yes" if one.check.hard else "no"
            lines.append(
                f"| `{one.check.id}` | {hard} | {len(one.findings):,} | {one.check.title} |"
            )
        lines.append("")
    failing = report.failing()
    if not failing:
        return "\n".join(lines)
    lines += ["## Findings", ""]
    for one in failing:
        lines += [f"### `{one.check.id}` {one.check.title}", ""]
        for finding in one.findings[:limit]:
            lines.append(f"- {finding}")
        if len(one.findings) > limit:
            lines.append(f"- and {len(one.findings) - limit:,} more, not printed")
        lines.append("")
    return "\n".join(lines)


def _text(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError, OSError:
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _tracked(root: Path) -> tuple[Path, ...]:
    """Every file git is tracking under ``root``.

    Asked of git rather than walked, and the difference is the whole point of
    the hygiene group. A walk finds the ``.DS_Store`` and the ``venv/`` that are
    correctly ignored and reports them, and after the third false alarm nobody
    reads ``H05`` again. Only a tracked file can be published.
    """
    if not (root / ".git").exists():
        return ()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return ()
    names = result.stdout.decode("utf-8", errors="replace").split("\0")
    return tuple(root / name for name in names if name)
