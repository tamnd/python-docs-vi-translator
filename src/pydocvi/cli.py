"""The ``pydocvi`` command line.

This module wires arguments to library functions and does nothing else. No other
module in the package imports it, and every command here should stay short
enough to read in one screen.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from pydocvi import (
    __version__,
    batch,
    classify,
    config,
    curate,
    fleet,
    glossary,
    mine,
    queue,
    routes,
    sync,
)
from pydocvi.catalog import Catalog
from pydocvi.client import Client
from pydocvi.logs import configure_logging
from pydocvi.memory import Memory

app = typer.Typer(
    name="pydocvi",
    help="Translate the CPython documentation into Vietnamese gettext catalogs.",
    no_args_is_help=True,
    add_completion=False,
)
tm_app = typer.Typer(help="Inspect the translation memory.", no_args_is_help=True)
app.add_typer(tm_app, name="tm")
fleet_app = typer.Typer(help="Tunnels, probes and traces.", no_args_is_help=True)
app.add_typer(fleet_app, name="fleet")
queue_app = typer.Typer(help="Inspect and repair the work queue.", no_args_is_help=True)
app.add_typer(queue_app, name="queue")
glossary_app = typer.Typer(help="Mine, curate and version the terminology.", no_args_is_help=True)
app.add_typer(glossary_app, name="glossary")

console = Console()


class ExitCode:
    """Stable exit codes.

    The runbook and both repositories' CI depend on telling a failed check apart
    from a usage error, and both apart from an unreachable fleet.
    """

    OK = 0
    CHECK_FAILED = 1
    USAGE = 2
    FLEET_UNREACHABLE = 3


@app.callback()
def root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Raise the log level.")] = False,
) -> None:
    """Keep the app a command group.

    Typer collapses a single-command app into a bare command, which would make
    ``pydocvi version`` a usage error until the second command lands.
    """
    configure_logging(verbose=verbose)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command(name="sync")
def sync_command(
    human: Annotated[
        bool, typer.Option("--human", help="Load human translations into the memory.")
    ] = False,
    branch: Annotated[str, typer.Option(help="Upstream branch to record.")] = sync.DEFAULT_BRANCH,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would change and write nothing.")
    ] = False,
) -> None:
    """Measure upstream, write the pin, and optionally load human translations."""
    where = config.paths()
    if not where.upstream.exists():
        console.print(f"[red]no upstream checkout at {where.upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    catalogs = sync.read_corpus(where.upstream)
    pin = sync.pin(
        where.upstream,
        repo="tamnd/python-docs-vi",
        branch=branch,
        commit=sync.head_commit(where.upstream),
    )

    table = Table(title=f"{pin.repo} @ {pin.branch} {pin.commit[:8]}", box=None)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("files", f"{pin.files:,}")
    table.add_row("entries", f"{pin.entries:,}")
    table.add_row("English words", f"{pin.words:,}")
    table.add_row("msgid characters", f"{pin.characters:,}")
    table.add_row("already translated", f"{pin.translated:,}")
    console.print(table)

    memory = Memory.load(where.memory)
    changes = sync.diff(memory, catalogs)
    console.print(
        f"upstream strings not in the memory: {len(changes.added):,}   "
        f"orphaned segments: {len(changes.orphaned):,}"
    )

    if human:
        stored = sync.load_human(memory, catalogs)
        console.print(f"human segments stored: {stored:,}")

    if dry_run:
        console.print("[yellow]dry run, nothing written[/yellow]")
        return

    where.upstream_pin.parent.mkdir(parents=True, exist_ok=True)
    where.upstream_pin.write_text(pin.as_yaml(), encoding="utf-8")
    where.reports.mkdir(parents=True, exist_ok=True)
    (where.reports / "sync.md").write_text(changes.as_markdown(), encoding="utf-8")
    memory.save(where.memory)
    console.print(f"wrote {where.upstream_pin} and {where.memory}")


@app.command(name="classify")
def classify_command(
    report: Annotated[
        bool, typer.Option("--report", help="Write the breakdown to reports/classify.md.")
    ] = False,
) -> None:
    """Count what the corpus is made of, and what a model will never see."""
    where = config.paths()
    catalogs = _corpus(where.upstream)
    counts = classify.counts([entry.msgid for cat in catalogs for entry in cat])

    table = Table(title="entries by kind", box=None)
    table.add_column("kind")
    table.add_column("entries", justify="right")
    table.add_column("share", justify="right")
    for kind, count in _kinds(counts):
        table.add_row(kind, f"{count:,}", f"{count / counts.total:.1%}")
    table.add_row("[bold]total", f"[bold]{counts.total:,}", "")
    console.print(table)
    console.print(f"never sent to a model: {counts.passthrough:,}")

    if report:
        where.reports.mkdir(parents=True, exist_ok=True)
        target = where.reports / "classify.md"
        target.write_text(_classify_markdown(counts), encoding="utf-8")
        console.print(f"wrote {target}")


def _kinds(counts: classify.Counts) -> list[tuple[str, int]]:
    return [
        ("prose", counts.prose),
        ("no-op", counts.noop),
        ("doctest", counts.doctest),
        ("literal block", counts.literal_block),
        ("version marker", counts.version_marker),
    ]


def _classify_markdown(counts: classify.Counts) -> str:
    lines = [
        "# Classification",
        "",
        "| kind | entries | share |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {kind} | {count} | {count / counts.total:.1%} |" for kind, count in _kinds(counts)
    ]
    lines += [
        f"| **total** | **{counts.total}** | |",
        "",
        f"{counts.passthrough} entries are copied through without a model call.",
        "",
    ]
    return "\n".join(lines)


@app.command(name="batch")
def batch_command(
    stats: Annotated[bool, typer.Option("--stats", help="Print the batching numbers.")] = False,
    top: Annotated[int, typer.Option(help="How many files to list by batch count.")] = 10,
) -> None:
    """Cut the corpus into batches and report what came out."""
    where = config.paths()
    batches = batch.build(_corpus(where.upstream), root=where.upstream)
    measured = batch.stats(batches)

    console.print(f"{measured.batches:,} batches over {measured.files:,} files")
    if not stats:
        return

    table = Table(box=None)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("entries batched", f"{measured.entries:,}")
    table.add_row("msgid characters", f"{measured.characters:,}")
    table.add_row("protected spans", f"{measured.spans:,}")
    table.add_row("entries per batch", f"{measured.entries_per_batch:.1f}")
    table.add_row("characters per batch", f"{measured.characters_per_batch:,.0f}")
    table.add_row("batches over a cap alone", f"{measured.oversized:,}")
    console.print(table)

    heaviest = sorted(batch.by_file(batches).items(), key=lambda pair: -pair[1])[:top]
    files = Table(title=f"heaviest {top} files", box=None)
    files.add_column("file")
    files.add_column("batches", justify="right")
    files.add_column("share", justify="right")
    for path, count in heaviest:
        files.add_row(path, f"{count:,}", f"{count / measured.batches:.1%}")
    console.print(files)


def _corpus(upstream: Path) -> list[Catalog]:
    """Read the upstream corpus, or exit saying it is not there."""
    if not upstream.exists():
        console.print(f"[red]no upstream checkout at {upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    return sync.read_corpus(upstream)


def _routes() -> list[routes.Route]:
    """The route file, or an exit that names the path it looked at."""
    try:
        return routes.load()
    except routes.RouteError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE) from error


@fleet_app.command("up")
def fleet_up() -> None:
    """Open a tunnel for every enabled route."""
    known = _routes()
    manager = fleet.Fleet(known)
    failed = 0
    for route in known:
        tunnel = manager.up(route)
        colour = "green" if tunnel.up else "red"
        console.print(f"[{colour}]{tunnel}[/{colour}]  {tunnel.detail}")
        failed += not tunnel.up
    if failed:
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)


@fleet_app.command("down")
def fleet_down() -> None:
    """Close every tunnel this tool opened.

    Worth doing when a run finishes. A forgotten tunnel is a bound port that the
    next run's ``ExitOnForwardFailure`` correctly refuses to work around.
    """
    known = _routes()
    manager = fleet.Fleet(known)
    for route in known:
        closed = manager.down(route)
        console.print(f"{route.name}: {'closed' if closed else 'nothing to close'}")


@fleet_app.command("status")
def fleet_status() -> None:
    """What is forwarded where."""
    known = _routes()
    table = Table(box=None)
    table.add_column("route")
    table.add_column("host")
    table.add_column("forward")
    table.add_column("state")
    for tunnel in fleet.Fleet(known).status():
        state = "[green]up[/green]" if tunnel.up else "[red]down[/red]"
        table.add_row(
            tunnel.route,
            tunnel.host,
            f"127.0.0.1:{tunnel.local_port} -> :{tunnel.remote_port}",
            state,
        )
    console.print(table)


@fleet_app.command("probe")
def fleet_probe() -> None:
    """Ask every route for its health, through the tunnel and with curl."""
    known = _routes()
    manager = fleet.Fleet(known)
    answering = 0
    for route in known:
        tunnel = manager.probe(route)
        colour = "green" if tunnel.up else "red"
        console.print(f"[{colour}]{route.name}[/{colour}]  {tunnel.detail}")
        answering += tunnel.up
    if not answering:
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)


@app.command()
def doctor() -> None:
    """Say whether a run can start, and exit 3 if it cannot.

    The command to put in front of anything expensive. Its exit code is what the
    runbook's ``set -e`` keys on, which is why it is a top-level command rather
    than a subcommand of ``fleet``.
    """
    known = _routes()
    manager = fleet.Fleet(known)
    diagnosis = fleet.Diagnosis(
        tunnels=[manager.probe(route) for route in known],
        missing_keys=routes.missing_keys(known),
        cooling=[],
    )
    for tunnel in diagnosis.tunnels:
        colour = "green" if tunnel.up else "red"
        console.print(f"[{colour}]{tunnel}[/{colour}]  {tunnel.detail}")
    for name in diagnosis.missing_keys:
        console.print(f"[red]{name} is not set in this shell[/red]")
    console.print(diagnosis.summary)
    if not diagnosis.healthy:
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)


@fleet_app.command("bench")
def fleet_bench(
    calls: Annotated[int, typer.Option(help="Calls to make per route.")] = 20,
    prompt: Annotated[
        str, typer.Option(help="What to send.")
    ] = "Reply with the single word: ready.",
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
) -> None:
    """Measure calls per hour and safe concurrency, and write the report.

    Every wall-clock estimate this tool prints comes from the report this
    writes. Without it the estimates are the design notes' aspirations, which
    were written before anything had run.
    """
    known = _routes()
    total = calls * len(known)
    if not yes:
        typer.confirm(f"{total} real calls across {len(known)} routes, continue?", abort=True)

    results = asyncio.run(_bench(known, calls=calls, prompt=prompt))
    for result in results:
        console.print(
            f"{result.route}: {result.successes}/{result.calls} in {result.seconds:.0f}s, "
            f"{result.calls_per_hour:.1f} calls/hour at concurrency {result.concurrency}"
        )

    where = config.paths()
    where.reports.mkdir(parents=True, exist_ok=True)
    target = where.reports / "fleet-bench.md"
    batches = len(batch.build(sync.read_corpus(where.upstream), root=where.upstream))
    target.write_text(fleet.bench_markdown(results, batches=batches), encoding="utf-8")
    console.print(f"wrote {target}")


async def _bench(known: list[routes.Route], *, calls: int, prompt: str) -> list[fleet.Bench]:
    async with Client() as client:
        return [await fleet.bench(client, route, calls=calls, prompt=prompt) for route in known]


@fleet_app.command("trace")
def fleet_trace(
    batch_id: Annotated[str, typer.Argument(help="The batch id from a provenance comment.")],
    day: Annotated[str, typer.Option(help="Trace day on the host, YYYY-MM-DD.")],
    route: Annotated[str | None, typer.Option(help="Which host to look on.")] = None,
) -> None:
    """Fetch the exact prompt and reply that produced a batch.

    This is the evidence for every failure diagnosis in the project, and it is
    two commands away from any wrong sentence in the corpus because the batch id
    is in the entry's provenance comment.
    """
    known = _routes()
    wanted = [r for r in known if route is None or r.name == route]
    if not wanted:
        console.print(f"[red]no route named {route}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    manager = fleet.Fleet(known)
    for candidate in wanted:
        found = manager.trace(candidate, batch_id, day=day)
        if found:
            console.print(found)
            return
    console.print(f"[red]no trace for {batch_id} on {day}[/red]")
    raise typer.Exit(ExitCode.CHECK_FAILED)


@queue_app.command("stats")
def queue_stats() -> None:
    """Counts per stage and state."""
    root = config.paths().queue
    table = Table(box=None)
    table.add_column("stage")
    for column in ("pending", "leased", "done", "dead"):
        table.add_column(column, justify="right")
    outstanding = 0
    for each in queue.queues(root):
        stats = each.stats()
        if not stats.total:
            continue
        outstanding += stats.outstanding
        table.add_row(
            str(stats.stage),
            f"{stats.pending:,}",
            f"{stats.leased:,}",
            f"{stats.done:,}",
            f"[red]{stats.dead:,}[/red]" if stats.dead else "0",
        )
    console.print(table)
    console.print(f"outstanding: {outstanding:,}")


@queue_app.command("reap")
def queue_reap() -> None:
    """Return expired leases to the queue.

    What a run killed on hour nine costs: the jobs it held come back, and no
    work is repeated because the ids are content addresses.
    """
    total = sum(len(each.reap(time.time())) for each in queue.queues(config.paths().queue))
    console.print(f"reaped {total:,} expired lease(s)")


@queue_app.command("retry")
def queue_retry(
    stage: Annotated[
        queue.Stage, typer.Option(help="Which stage to retry.")
    ] = queue.Stage.TRANSLATE,
) -> None:
    """Move dead jobs back to pending.

    Never automatic. This is the command you run after the traces told you what
    was wrong and you fixed it.
    """
    moved = queue.Queue(config.paths().queue, stage).retry()
    console.print(f"{moved:,} job(s) back to pending")


@queue_app.command("dead")
def queue_dead(
    stage: Annotated[
        queue.Stage, typer.Option(help="Which stage to list.")
    ] = queue.Stage.TRANSLATE,
    top: Annotated[int, typer.Option(help="How many to show.")] = 20,
) -> None:
    """List the jobs that used all three attempts, with why."""
    jobs = queue.Queue(config.paths().queue, stage).jobs(queue.State.DEAD)
    if not jobs:
        console.print("nothing dead")
        return
    table = Table(box=None)
    table.add_column("job")
    table.add_column("attempts", justify="right")
    table.add_column("error")
    for job in jobs[:top]:
        table.add_row(job.id, str(job.attempts), job.error or "")
    console.print(table)
    console.print(f"{len(jobs):,} dead job(s) in {stage}")


@queue_app.command("drain")
def queue_drain(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
) -> None:
    """Delete finished jobs. Never touches outstanding work."""
    if not yes:
        typer.confirm("delete every done job across all stages?", abort=True)
    removed = sum(each.drain() for each in queue.queues(config.paths().queue))
    console.print(f"removed {removed:,} finished job(s)")


@tm_app.command("stats")
def tm_stats() -> None:
    """Counts by provenance."""
    memory = Memory.load(config.paths().memory)
    table = Table(box=None)
    table.add_column("source")
    table.add_column("segments", justify="right")
    for source, count in memory.counts().items():
        table.add_row(source, f"{count:,}")
    table.add_row("[bold]total", f"[bold]{len(memory):,}")
    console.print(table)


@tm_app.command("show")
def tm_show(segment: str) -> None:
    """Print one segment by its id."""
    found = Memory.load(config.paths().memory).get(segment)
    if found is None:
        console.print(f"[red]no segment {segment}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    console.print(found)


@glossary_app.command("mine")
def glossary_mine(
    limit: Annotated[
        int, typer.Option(help="How many frequent phrases to take.")
    ] = mine.FREQUENCY_LIMIT,
    minimum: Annotated[
        int, typer.Option(help="Fewest occurrences a phrase may have.")
    ] = mine.MIN_COUNT,
) -> None:
    """Propose candidate terms from the four sources.

    The sources are read in descending order of trust and a candidate found
    twice keeps the more trusted of the two. Frequency is deliberately last: the
    most common noun phrase in this corpus is "the following example".
    """
    where = config.paths()
    corpus = _corpus(where.upstream)
    memory = Memory.load(where.memory) if where.memory.exists() else Memory()
    page = next((one for one in corpus if one.path.name == "glossary.po"), None)

    found = mine.merge(
        mine.from_human(memory),
        mine.from_term_page(page) if page is not None else [],
        mine.from_frequency(
            [entry.msgid for one in corpus for entry in one], limit=limit, minimum=minimum
        ),
        mine.from_machine(_machine(where.content)),
    )

    measured = mine.stats(found)
    table = Table(box=None)
    table.add_column("source")
    table.add_column("candidates", justify="right")
    for source, count in measured.by_source.items():
        table.add_row(source, f"{count:,}")
    table.add_row("[bold]total", f"[bold]{measured.total:,}")
    console.print(table)
    console.print(f"{measured.contested:,} of them rendered more than one way already")

    where.candidates.parent.mkdir(parents=True, exist_ok=True)
    where.candidates.write_text(mine.dumps(found), encoding="utf-8")
    console.print(f"wrote {where.candidates}")


def _machine(content: Path) -> list[Catalog]:
    """The already-translated tree, or nothing if it is not checked out yet."""
    return sync.read_corpus(content) if content.exists() else []


@glossary_app.command("curate")
def glossary_curate(
    size: Annotated[int, typer.Option("--batch", help="Candidates per call.")] = curate.BATCH_SIZE,
    take: Annotated[int, typer.Option(help="Curate only the first N candidates, 0 for all.")] = 0,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
) -> None:
    """Ask the fleet for a rendering of every candidate, and refuse most answers.

    Writes the accepted rows to the proposal file for a person to read. Nothing
    here touches ``glossary.yaml``: promoting is ``glossary bump`` and it is a
    separate command because a curated row is a suggestion until somebody has
    agreed with it.
    """
    where = config.paths()
    if not where.candidates.exists():
        console.print(f"[red]no candidates at {where.candidates}, run glossary mine[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    found = mine.loads(where.candidates.read_text(encoding="utf-8"))
    if take:
        found = found[:take]
    made = curate.batches(found, size=size)
    if not made:
        console.print("nothing to curate")
        return

    known = _routes()
    if not yes:
        typer.confirm(f"{len(made)} calls about {len(found):,} terms, continue?", abort=True)

    started = time.monotonic()
    replies = asyncio.run(_curate(known, made))
    outcome = curate.collect(replies)
    elapsed = time.monotonic() - started

    console.print(
        f"{len(outcome.accepted):,} accepted, {len(outcome.declined):,} declined, "
        f"{len(outcome.dropped):,} dropped, in {elapsed:.0f}s"
    )
    if outcome.by_rule:
        table = Table(box=None)
        table.add_column("rule")
        table.add_column("dropped", justify="right")
        for rule, count in outcome.by_rule.items():
            table.add_row(rule, f"{count:,}")
        console.print(table)

    proposed = glossary.Glossary(version=0).with_terms(
        glossary.match_order(outcome.accepted), version=0
    )
    where.proposal.parent.mkdir(parents=True, exist_ok=True)
    glossary.save(proposed, where.proposal)
    console.print(f"wrote {where.proposal}")

    where.reports.mkdir(parents=True, exist_ok=True)
    report = where.reports / "glossary-curation.md"
    report.write_text(curate.report(outcome, prompt=curate.prompt_id()), encoding="utf-8")
    console.print(f"wrote {report}")


async def _curate(known: list[routes.Route], made: list[curate.Batch]) -> list[curate.Reply]:
    done = 0

    def tick(reply: curate.Reply) -> None:
        nonlocal done
        done += 1
        console.print(
            f"[{done}/{len(made)}] batch {reply.index}: {len(reply.accepted)} accepted, "
            f"{len(reply.declined)} declined, {len(reply.dropped)} dropped"
        )

    async with Client() as client:
        return await curate.ask(client, known, made, on_reply=tick)


@glossary_app.command("check")
def glossary_check(
    path: Annotated[Path | None, typer.Option(help="Check a file other than the glossary.")] = None,
) -> None:
    """Run ``G-e`` and ``G-f`` over the list, and ``G05`` over the Markdown."""
    where = config.paths()
    target = path or where.glossary
    if not target.exists():
        console.print(f"[red]no glossary at {target}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    terms = glossary.load(target)
    problems = glossary.check(terms)
    if where.glossary_markdown.exists() and path is None:
        problems += glossary.agrees(where.glossary_markdown.read_text(encoding="utf-8"), terms)

    for problem in problems:
        console.print(f"[red]{problem.rule}[/red] {problem.en}: {problem.detail}")
    measured = glossary.stats(terms)
    console.print(
        f"{measured.terms:,} terms, {measured.kept:,} keep-English, "
        f"{measured.contextual:,} contextual, version {terms.version}"
    )
    if problems:
        console.print(f"[red]{len(problems)} problems[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)


@glossary_app.command("diff")
def glossary_diff(
    old: Annotated[int, typer.Argument(help="The older version.")],
    new: Annotated[int, typer.Argument(help="The newer version.")],
) -> None:
    """Print what moved between two versions.

    This is what makes terminology revisable. ``stale --glossary`` intersects
    these terms with the memory and re-queues the few hundred entries that
    contain one, rather than the corpus.
    """
    before, after = _version(old), _version(new)
    moved = glossary.diff(before, after)
    for term in moved.added:
        console.print(f"[green]+ {term.en}[/green] {term.rendering}")
    for term in moved.removed:
        console.print(f"[red]- {term.en}[/red] {term.rendering}")
    for was, now in moved.changed:
        console.print(f"[yellow]~ {now.en}[/yellow] {was.rendering} -> {now.rendering}")
    console.print(f"{len(moved.terms):,} terms moved between v{old} and v{new}")


def _version(number: int) -> glossary.Glossary:
    """One archived version, or an exit naming the file it wanted."""
    where = config.paths()
    if number == glossary.load(where.glossary).version and where.glossary.exists():
        return glossary.load(where.glossary)
    path = where.versions / f"v{number}.yaml"
    if not path.exists():
        console.print(f"[red]no archived version {number} at {path}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    return glossary.load(path)


@glossary_app.command("show")
def glossary_show(
    term: Annotated[str | None, typer.Argument(help="One term, or all of them.")] = None,
) -> None:
    """Print the glossary, or one row of it."""
    terms = glossary.load(config.paths().glossary)
    if term is not None:
        found = terms.get(term)
        if found is None:
            console.print(f"[red]no term {term!r}[/red]")
            raise typer.Exit(ExitCode.CHECK_FAILED)
        console.print(found)
        return
    console.print(glossary.table(terms))


@glossary_app.command("bump")
def glossary_bump(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
) -> None:
    """Promote the reviewed proposal, raise the version, regenerate the Markdown.

    The old version is archived first. Without that ``glossary diff 6 7`` has
    only one side and a version bump becomes a re-queue of everything, which is
    the failure that makes people stop bumping.
    """
    where = config.paths()
    if not where.proposal.exists():
        console.print(f"[red]no proposal at {where.proposal}, run glossary curate[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    current = (
        glossary.load(where.glossary) if where.glossary.exists() else glossary.Glossary(version=0)
    )
    proposed = glossary.load(where.proposal)
    merged = current.with_terms(
        glossary.match_order({term.en: term for term in [*current.terms, *proposed.terms]}.values())
    )
    if merged.version == current.version:
        console.print("nothing to promote")
        return

    problems = glossary.check(merged)
    for problem in problems:
        console.print(f"[red]{problem.rule}[/red] {problem.en}: {problem.detail}")
    if problems:
        raise typer.Exit(ExitCode.CHECK_FAILED)

    moved = glossary.diff(current, merged)
    if not yes:
        typer.confirm(
            f"v{current.version} -> v{merged.version}, {len(moved.terms):,} terms move, continue?",
            abort=True,
        )

    if where.glossary.exists():
        where.versions.mkdir(parents=True, exist_ok=True)
        glossary.save(current, where.versions / f"v{current.version}.yaml")
    where.glossary.parent.mkdir(parents=True, exist_ok=True)
    glossary.save(merged, where.glossary)
    console.print(f"wrote {where.glossary} at v{merged.version}")

    if where.glossary_markdown.exists():
        markdown = where.glossary_markdown.read_text(encoding="utf-8")
        where.glossary_markdown.write_text(glossary.render(markdown, merged), encoding="utf-8")
        console.print(f"regenerated the table in {where.glossary_markdown}")
    where.proposal.unlink()


def main() -> NoReturn:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(130)
    raise AssertionError("unreachable: typer exits via SystemExit")
