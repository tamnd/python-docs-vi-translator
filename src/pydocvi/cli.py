"""The ``pydocvi`` command line.

This module wires arguments to library functions and does nothing else. No other
module in the package imports it, and every command here should stay short
enough to read in one screen.
"""

import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from pydocvi import (
    __version__,
    apply,
    audit,
    batch,
    catalog,
    classify,
    config,
    curate,
    fleet,
    glossary,
    mine,
    queue,
    render,
    report,
    routes,
    sync,
    translate,
    worker,
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
prompt_app = typer.Typer(help="Show what goes over the wire.", no_args_is_help=True)
app.add_typer(prompt_app, name="prompt")
report_app = typer.Typer(help="Generate the committed reports.", no_args_is_help=True)
app.add_typer(report_app, name="report")

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
        loaded = sync.load_human(memory, catalogs)
        console.print(f"human segments stored: {loaded.stored:,}   dropped: {loaded.dropped:,}")

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
    write_report: Annotated[
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

    if write_report:
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


@app.command(name="translate")
def translate_command(
    tier: Annotated[int | None, typer.Option(help="Which tier, 1 to 6.")] = None,
    file: Annotated[
        Path | None, typer.Option("--file", help="One catalog, relative to upstream.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Stop after this many calls.")] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Work what is already queued and queue nothing new.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what it would cost and spend nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
) -> None:
    """Translate a tier, or a file, and write what came back to the memory.

    The expensive command. It spends a model call per batch and it is the only
    thing in this tool that does, so it dry-runs first, prints what the last
    ``fleet bench`` says the wall clock will be, and asks.

    Batches every entry of which the memory already holds are dropped before
    anything is queued, which is what makes a second pass over a tier cost
    almost nothing. What is queued is content-addressed on the file, the
    entries, the prompt and the glossary version, so a prompt change re-queues
    the corpus and an interrupted run picks up exactly where it stopped.
    """
    where = config.paths()
    known = _working_routes()
    batches = _selected(where.upstream, tier=tier, file=file)
    run = translate.Run(
        queue=queue.Queue(where.queue, queue.Stage.TRANSLATE),
        memory=Memory.load(where.memory),
        glossary=glossary.load(where.glossary),
        batches=batches,
        run=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
    )

    pending = len(run.queue)
    if resume:
        console.print(f"resuming: {pending:,} batches already queued, queueing nothing new")
    else:
        plan = run.plan(run.untranslated(batches), tier=tier, write=not dry_run)
        # str() rather than the object, because rich renders a dataclass as its
        # repr and would print the fields instead of the sentence.
        console.print(str(plan))
        # On a dry run nothing was written, so what the queue holds and what the
        # plan would have added are two separate numbers and the estimate wants
        # both. On a real run the second is already inside the first.
        pending = len(run.queue) + (plan.queued if dry_run else 0)
    console.print(_wall_clock(where, pending))

    if dry_run:
        return
    if not pending:
        console.print("nothing outstanding, every batch in the selection is already done")
        return
    if missing := routes.missing_keys(known):
        console.print(f"[red]{', '.join(missing)} is not set in this shell[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)
    if not yes:
        typer.confirm(f"{pending:,} batches to call, continue?", abort=True)

    _translated(run, asyncio.run(_translate(run, known, limit=limit)), where=where)


def _selected(upstream: Path, *, tier: int | None, file: Path | None) -> list[batch.Batch]:
    """The batches a person asked for, by tier or by file or all of them."""
    batches = batch.build(_catalogs(upstream, file), root=upstream)
    if tier is not None:
        try:
            batches = batch.of_tier(batches, tier)
        except batch.TierError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(ExitCode.USAGE) from error
    if not batches:
        console.print("[red]nothing to translate in that selection[/red]")
        raise typer.Exit(ExitCode.USAGE)
    return batches


def _catalogs(upstream: Path, file: Path | None) -> list[Catalog]:
    """The corpus, or the one file of it somebody named.

    Filtered after reading rather than read on its own, because a batch carries
    the path it came from and the two paths have to agree. Reading one file by a
    different route was how the first version of this produced batches whose
    ``file`` was absolute and whose traces could not be found.
    """
    corpus = _corpus(upstream)
    if file is None:
        return corpus
    wanted = (upstream / file).resolve()
    found = [one for one in corpus if one.path.resolve() == wanted]
    if not found:
        console.print(f"[red]no catalog at {upstream / file}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    return found


async def _translate(
    run: translate.Run, known: Sequence[routes.Route], *, limit: int | None
) -> worker.Progress:
    """Drain the translate queue through the fleet.

    The memory is written whatever happens, including a Ctrl-C, which is what
    the ``finally`` is for. Losing an hour of calls to a keystroke is the
    failure this whole stage is arranged to avoid.
    """
    async with Client() as client:
        pool = worker.Worker(
            queue=run.queue,
            router=routes.Router.of(known),
            client=client,
            build=run.build,
            handle=run.handle,
            limit=limit,
        )
        try:
            return await pool.run()
        finally:
            run.save(force=True)


def _translated(run: translate.Run, progress: worker.Progress, *, where: config.Paths) -> None:
    """The lines a person reads when a run stops, however it stopped.

    The tally is written to disk here as well as printed. A refusal that was
    fixed on rung 2 leaves no trace in the corpus, so this file is the only
    place ``report quality`` can get the number that says whether the prompt is
    working.
    """
    tally = run.tally
    tally.calls = progress.calls
    tally.seconds = progress.seconds
    tally.by_route = dict(progress.by_route)
    tally.save(where.tallies)
    console.print(
        f"{progress.done:,} calls in {progress.seconds / 3600:.1f}h "
        f"at {progress.calls_per_hour:.1f} calls/hour, "
        f"{progress.empty:,} empty, {progress.failed:,} failed"
    )
    console.print(
        f"{tally.accepted:,} entries accepted, {tally.refused:,} refused, "
        f"{tally.rejected:,} batches refused whole, {tally.dead:,} entries out of rungs"
    )
    if tally.by_rule:
        table = Table(title="refusals", box=None)
        table.add_column("rule")
        table.add_column("count", justify="right")
        for rule, count in sorted(tally.by_rule.items()):
            table.add_row(rule, f"{count:,}")
        console.print(table)
    for rung, count in sorted(tally.by_attempt.items()):
        console.print(f"rung {rung}: {count:,} refusals")


def _wall_clock(where: config.Paths, batches: int) -> str:
    """What the last bench says this will take.

    Named as an absence when there has not been one. A default rate would print
    a number that looks measured and is not, and the estimate is the thing the
    person is deciding on.
    """
    target = where.reports / "fleet-bench.json"
    if not target.exists():
        return "no fleet bench on record, so no estimate. Run pydocvi fleet bench first"
    measured = fleet.Measured.from_json(target.read_text(encoding="utf-8"))
    if measured.calls_per_hour <= 0:
        return "the last fleet bench measured no calls, so no estimate"
    hours = fleet.hours_for(batches, measured.calls_per_hour)
    return (
        f"about {hours:.1f}h at the measured {measured.calls_per_hour:.1f} calls/hour "
        f"over {', '.join(measured.routes)}, retries included"
    )


@app.command(name="apply")
def apply_command(
    branch: Annotated[str, typer.Option(help="Which version the catalogs are of.")] = (
        sync.DEFAULT_BRANCH
    ),
    check: Annotated[
        bool, typer.Option("--check", help="Compare the content repo against the memory.")
    ] = False,
    file: Annotated[
        Path | None, typer.Option("--file", help="One catalog, relative to upstream.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would change and write nothing.")
    ] = False,
    refuzzy: Annotated[
        bool,
        typer.Option(
            "--refuzzy",
            help="Treat an unmarked translation in the corpus as unreviewed, not as a reviewer's.",
        ),
    ] = False,
) -> None:
    """Write the memory onto the catalogs, or check that it already is.

    ``--check`` is the one CI runs. It renders every file from the same memory
    and compares bytes, so a hand edit on the content repo fails the build by
    name rather than being reverted by the next run and mentioned in a log.

    ``--refuzzy`` is for a corpus that came from somewhere else. An unfuzzied
    entry normally means a reviewer signed off, and it is protected for the days
    the Transifex round trip takes; this repository inherited 241 entries an
    earlier script wrote without marking, which look identical. The flag says to
    believe the memory over the file where the two disagree, and it never touches
    an entry the memory records as ``human``, so it is not the bulk unfuzzy
    command spec 02 §4 refuses to have. It is meant to be run once.
    """
    where = config.paths()
    memory = Memory.load(where.memory)
    sources = _sources(where.upstream, file)
    stamp = apply.Stamp(
        project=f"Python {branch}",
        run=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
    )
    run = apply.check if check else apply.plan
    result = run(
        sources, memory, root=where.upstream, into=where.content, stamp=stamp, refuzzy=refuzzy
    )

    counts = result.counts
    console.print(
        f"{counts.written:,} written   {counts.kept:,} left to the reviewer   "
        f"{counts.copied:,} copied through   {counts.untranslated:,} still English"
    )
    for one in result.changed:
        console.print(f"  {one.path.relative_to(where.content)}")

    if check:
        if result.clean:
            console.print("[green]the catalogs are what the memory says they should be[/green]")
            return
        console.print(f"[red]{len(result.changed):,} file(s) differ from the memory[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    if dry_run:
        console.print(f"[yellow]dry run, {len(result.changed):,} file(s) would change[/yellow]")
        return
    console.print(f"wrote {len(apply.write(result)):,} file(s) to {where.content}")


def _sources(upstream: Path, file: Path | None) -> list[Path]:
    """The catalogs to apply, all of them or the one somebody named."""
    if not upstream.exists():
        console.print(f"[red]no upstream checkout at {upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    if file is None:
        return catalog.walk(upstream)
    one = upstream / file
    if not one.exists():
        console.print(f"[red]no catalog at {one}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    return [one]


@prompt_app.command("show")
def prompt_show(
    which: Annotated[
        str | None, typer.Argument(help="A batch id, or a file path to take the first batch of.")
    ] = None,
    fingerprint: Annotated[
        bool, typer.Option("--hash", help="Print the prompt hash and nothing else.")
    ] = False,
) -> None:
    """Print exactly what one batch sends, system message and user message.

    This is the command that makes the hash in a provenance comment worth
    recording. A person who finds a wrong Vietnamese sentence can read the batch
    id in the comment beside it, run this, and see the same bytes the model saw,
    without a fleet, a key or a tunnel.

    The hash on its own is the fast path, and it takes no corpus: it is what CI
    compares against the manifest to catch a prompt edited without a version
    bump behind it.
    """
    if fingerprint:
        console.print(render.fingerprint())
        return

    where = config.paths()
    batches = batch.build(_corpus(where.upstream), root=where.upstream)
    found = _one_batch(batches, which)
    prompt = render.render(found, glossary.load(where.glossary))

    console.print(
        f"[dim]batch {found.id}, {len(found)} entries, prompt {prompt.fingerprint}[/dim]",
        soft_wrap=True,
    )
    _verbatim(prompt.system)
    console.print("[dim]" + "-" * 60 + "[/dim]")
    _verbatim(prompt.user)


def _verbatim(text: str) -> None:
    """Print text as it is, with nothing done to it.

    Rich wraps at the terminal width, reads square brackets as markup and
    highlights anything that looks like a number or a path. All three are wrong
    here: the point of this command is that what it prints is what was sent, and
    the first version of it reflowed a long entry into three lines that no
    ``msgid`` ever had.
    """
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def _one_batch(batches: list[batch.Batch], which: str | None) -> batch.Batch:
    """The batch a person named, by id or by file, or the first one there is."""
    if which is None:
        return batches[0]
    for one in batches:
        if one.id == which or one.path == which:
            return one
    console.print(f"[red]no batch {which!r}, and no file by that name[/red]")
    raise typer.Exit(ExitCode.USAGE)


def _corpus(upstream: Path) -> list[Catalog]:
    """Read the upstream corpus, or exit saying it is not there."""
    if not upstream.exists():
        console.print(f"[red]no upstream checkout at {upstream}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)
    return sync.read_corpus(upstream)


def _routes() -> list[routes.Route]:
    """Every route in the file, enabled or not, or an exit naming the path.

    For the two commands whose job is to show or fetch rather than to spend:
    ``status``, which should show a host that is switched off, and ``trace``,
    which is how somebody works out why it was switched off.
    """
    try:
        return routes.load()
    except routes.RouteError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE) from error


def _working_routes() -> list[routes.Route]:
    """The routes a command may tunnel to, probe, or spend calls on.

    ``enabled`` was honoured by ``Router`` and by nothing the command line
    reaches, so switching a host off in the route file changed nothing: it was
    still tunnelled to, still probed, still benched, and ``fleet up`` said it
    opened a tunnel for every enabled route while opening one for all of them.
    A host is switched off because somebody found out it does not work, and that
    finding should not need to be made twice.
    """
    known = [route for route in _routes() if route.enabled]
    if not known:
        console.print("[red]every route in the file is disabled[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)
    return known


@fleet_app.command("up")
def fleet_up() -> None:
    """Open a tunnel for every enabled route."""
    known = _working_routes()
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
    known = _working_routes()
    manager = fleet.Fleet(known)
    for route in known:
        closed = manager.down(route)
        console.print(f"{route.name}: {'closed' if closed else 'nothing to close'}")


@fleet_app.command("status")
def fleet_status() -> None:
    """What is forwarded where, including the hosts that are switched off.

    The one fleet command that reports every route in the file rather than the
    ones a run would use, because a host missing from this table looks like a
    host missing from the file.
    """
    known = _routes()
    switched_off = {route.name for route in known if not route.enabled}
    table = Table(box=None)
    table.add_column("route")
    table.add_column("host")
    table.add_column("forward")
    table.add_column("state")
    for tunnel in fleet.Fleet(known).status():
        state = "[green]up[/green]" if tunnel.up else "[red]down[/red]"
        if tunnel.route in switched_off:
            state = f"{state} [yellow](disabled)[/yellow]"
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
    known = _working_routes()
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
def doctor(
    quick: Annotated[
        bool, typer.Option("--quick", help="Health only. Do not spend a call per route.")
    ] = False,
) -> None:
    """Say whether a run can start, and exit 3 if it cannot.

    The command to put in front of anything expensive. Its exit code is what the
    runbook's ``set -e`` keys on, which is why it is a top-level command rather
    than a subcommand of ``fleet``.

    It asks every healthy route to complete one short call, which costs about a
    minute for the fleet. That is new, and it is here because the cheap version
    of this command passed a host that answers ``/v1/health`` with 200 and then
    never finishes a completion. Saying a run can start when it cannot is the
    one thing this command must not do, so the expensive question is the default
    and ``--quick`` is for the person who only wants to know about tunnels.
    """
    known = _working_routes()
    manager = fleet.Fleet(known)
    tunnels = [manager.probe(route) for route in known]
    for tunnel in tunnels:
        colour = "green" if tunnel.up else "red"
        console.print(f"[{colour}]{tunnel}[/{colour}]  {tunnel.detail}")

    missing = routes.missing_keys(known)
    healthy = [route for route in known if _named(tunnels, route.name)]
    answered: list[fleet.Liveness] = []
    if not quick and healthy and not missing:
        console.print(f"asking {len(healthy)} route(s) to complete a call, a minute or so")
        answered = asyncio.run(_alive(healthy, say=_alive_line))

    diagnosis = fleet.Diagnosis(
        tunnels=tunnels, missing_keys=missing, cooling=[], answered=answered
    )
    for name in diagnosis.missing_keys:
        console.print(f"[red]{name} is not set in this shell[/red]")
    for result in diagnosis.answered:
        if result.substituted:
            console.print(
                f"[yellow]{result.route} was asked for {result.asked} "
                f"and served {result.served}[/yellow]"
            )
    console.print(diagnosis.summary)
    if not diagnosis.healthy:
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)


def _named(tunnels: Sequence[fleet.Tunnel], name: str) -> bool:
    return any(tunnel.up for tunnel in tunnels if tunnel.route == name)


def _alive_line(result: fleet.Liveness) -> None:
    colour = "green" if result.up else "red"
    console.print(f"[{colour}]{result}[/{colour}]")


async def _alive(
    known: Sequence[routes.Route],
    *,
    say: Callable[[fleet.Liveness], None] = lambda _: None,
) -> list[fleet.Liveness]:
    """Ask every route for one completion, all of them at once.

    Concurrently, unlike ``bench``, because this measures nothing and a dead
    host is two minutes of waiting that the healthy ones should not be queued
    behind.
    """
    async with Client(retries=0) as client:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(fleet.alive(client, route)) for route in known]
        results = [task.result() for task in tasks]
    for result in results:
        say(result)
    return results


@fleet_app.command("bench")
def fleet_bench(
    calls: Annotated[int, typer.Option(help="Calls to make per route.")] = 20,
    prompt: Annotated[
        str, typer.Option(help="What to send.")
    ] = "Reply with the single word: ready.",
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation.")] = False,
    route: Annotated[
        list[str] | None, typer.Option(help="Measure only these routes. Repeatable.")
    ] = None,
) -> None:
    """Measure calls per hour and safe concurrency, and write the report.

    Every wall-clock estimate this tool prints comes from the report this
    writes. Without it the estimates are the design notes' aspirations, which
    were written before anything had run.

    The key check is here rather than left to ``doctor`` because the first real
    three-route run of this command was made in a shell that had no key in it.
    Eighteen calls came back 401 in a little over a second each, and the command
    exited 0 and wrote a report saying every route does 0.0 calls per hour. A
    report of zeros is worse than no report, because the next estimate is
    computed off it.

    ``--route`` exists because the second real run met a host that answers health
    and never finishes a call. At a 1 200 second timeout and two retries that is
    an hour a call and six hours for the route, and it stood between the run and
    the two hosts behind it in the list. A sick host should cost its own
    measurement and nobody else's.

    Anything in the route file that was not measured is named in the report,
    whether it was left out by ``--route`` or switched off in the file. Switching
    the sick host off made it drop out of the report silently, which is the thing
    the line was added to prevent.
    """
    every = _routes()
    known = _working_routes()
    if route:
        if unknown := sorted(set(route) - {one.name for one in known}):
            console.print(f"[red]no route named {', '.join(unknown)}[/red]")
            raise typer.Exit(ExitCode.USAGE)
        known = [one for one in known if one.name in route]
    absent = [one.name for one in every if one.name not in {other.name for other in known}]

    if missing := routes.missing_keys(known):
        console.print(f"[red]{', '.join(missing)} is not set in this shell[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)

    total = calls * len(known)
    if not yes:
        typer.confirm(f"{total} real calls across {len(known)} routes, continue?", abort=True)

    results = asyncio.run(_bench(known, calls=calls, prompt=prompt, say=_bench_line))

    if not any(result.successes for result in results):
        console.print("[red]no route completed a single call, nothing was measured[/red]")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)

    where = config.paths()
    where.reports.mkdir(parents=True, exist_ok=True)
    target = where.reports / "fleet-bench.md"
    batches = len(batch.build(sync.read_corpus(where.upstream), root=where.upstream))
    target.write_text(
        fleet.bench_markdown(results, batches=batches, absent=absent), encoding="utf-8"
    )
    # The same measurement twice, once for a person and once for `translate`,
    # which has to print what a tier costs before it spends anything. Reading the
    # number back out of the markdown would be a parser for this project's own
    # report and would break silently the first time a column moved.
    sidecar = where.reports / "fleet-bench.json"
    sidecar.write_text(fleet.Measured.of(results, batches=batches).as_json(), encoding="utf-8")
    console.print(f"wrote {target} and {sidecar}")


def _bench_line(result: fleet.Bench) -> None:
    """One route's measurement, printed the moment that route is done."""
    console.print(
        f"{result.route}: {result.successes}/{result.calls} in {result.seconds:.0f}s, "
        f"{result.calls_per_hour:.1f} calls/hour at concurrency {result.concurrency}"
    )


async def _bench(
    known: list[routes.Route],
    *,
    calls: int,
    prompt: str,
    say: Callable[[fleet.Bench], None] = lambda _: None,
) -> list[fleet.Bench]:
    """Measure every route in turn, announcing each one as it finishes.

    One route at a time, because two hosts measured at once share this machine's
    uplink and the number stops being about the host. Announced as it finishes
    rather than at the end, because a route here is a quarter of an hour and the
    first real three-route run was killed during the third one. It had measured
    two hosts by then and printed neither, which is a bad trade for one line of
    code.
    """
    async with Client() as client:
        measured = []
        for route in known:
            result = await fleet.bench(client, route, calls=calls, prompt=prompt)
            say(result)
            measured.append(result)
        return measured


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

    known = _working_routes()
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

    where.reports.mkdir(parents=True, exist_ok=True)
    report = where.reports / "glossary-curation.md"
    report.write_text(curate.report(outcome, prompt=curate.prompt_id()), encoding="utf-8")
    console.print(f"wrote {report}")

    # The report is written either way, because a run that failed is exactly the
    # one worth having a record of. The proposal is not, because a run that
    # accepted nothing has nothing to propose and would only overwrite the last
    # good one with an empty file. That is not hypothetical: the fleet went down
    # mid-run and a 873 term proposal became a 165 byte stub.
    if not outcome.accepted:
        console.print(
            f"[red]nothing was accepted, so {where.proposal.name} is left as it was.[/red]"
        )
        console.print("check the fleet with [bold]pydocvi doctor[/bold] and run this again")
        raise typer.Exit(ExitCode.FLEET_UNREACHABLE)

    proposed = glossary.Glossary(version=0).with_terms(
        glossary.match_order(outcome.accepted), version=0
    )
    where.proposal.parent.mkdir(parents=True, exist_ok=True)
    glossary.save(proposed, where.proposal)
    console.print(f"wrote {where.proposal}")


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
    fix: Annotated[bool, typer.Option(help="Regenerate the table G05 is unhappy about.")] = False,
) -> None:
    """Run ``G-e`` and ``G-f`` over the list, and ``G05`` over the Markdown.

    ``--fix`` is only for ``G05``, which is the one problem here with a
    mechanical answer: the table is generated, so a table that disagrees with
    the glossary is regenerated rather than argued with. ``G-e`` and ``G-f`` are
    about what the rows say and there is nothing to do about them but decide.
    """
    where = config.paths()
    target = path or where.glossary
    if not target.exists():
        console.print(f"[red]no glossary at {target}[/red]")
        raise typer.Exit(ExitCode.CHECK_FAILED)

    terms = glossary.load(target)
    problems = glossary.check(terms)
    if where.glossary_markdown.exists() and path is None:
        markdown = where.glossary_markdown.read_text(encoding="utf-8")
        stale = glossary.agrees(markdown, terms)
        if stale and fix:
            where.glossary_markdown.write_text(glossary.render(markdown, terms), encoding="utf-8")
            console.print(f"regenerated the table in {where.glossary_markdown}")
        else:
            problems += stale

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


@app.command(name="audit")
def audit_command(
    only: Annotated[
        str | None,
        typer.Option("--only", help="Check ids, group names or prefixes, comma-separated."),
    ] = None,
    skip: Annotated[
        str | None, typer.Option("--skip", help="The same, for checks to leave out.")
    ] = None,
    lang: Annotated[str, typer.Option("--lang", help="Which language to audit.")] = config.LANGUAGE,
    write_to: Annotated[
        Path | None, typer.Option("--report", help="Write the Markdown report here.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the finding list on stdout instead of a table.")
    ] = False,
    fail_soft: Annotated[
        bool, typer.Option("--fail-soft", help="Let a soft check fail the run too.")
    ] = False,
    branch: Annotated[str, typer.Option(help="Which version the catalogs are of.")] = (
        sync.DEFAULT_BRANCH
    ),
) -> None:
    """Run every check over the committed corpus.

    Exit 0 iff every hard check passes. This is expected to be red for a long
    time and that is the job: the audit is the executable form of every quality
    claim the project makes, so it goes green when the corpus is finished and
    not before.

    Nothing here calls a model, opens a socket or reads a key. That is what
    makes it fast enough to run on every push and what makes a green run worth
    believing without knowing which route happened to answer that day.
    """
    if lang != config.LANGUAGE:
        console.print(f"[red]this tool only translates into {config.LANGUAGE}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    where = config.paths()
    if not where.content.exists():
        console.print(f"[red]no content repo at {where.content}[/red]")
        raise typer.Exit(ExitCode.USAGE)

    started = time.monotonic()
    corpus = audit.assemble(where, branch=branch)
    try:
        result = audit.run(corpus, only=_names(only), skip=_names(skip), fail_soft=fail_soft)
    except audit.UnknownCheckError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(ExitCode.USAGE) from error

    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        write_to.write_text(audit.markdown(result), encoding="utf-8")
    if as_json:
        typer.echo(result.as_json(), nl=False)
    else:
        _audit_table(result, corpus, elapsed=time.monotonic() - started)
    if not result.ok:
        raise typer.Exit(ExitCode.CHECK_FAILED)


def _names(given: str | None) -> list[str]:
    """A comma-separated option as a list, with the empty pieces dropped."""
    if given is None:
        return []
    return [one.strip() for one in given.split(",") if one.strip()]


def _audit_table(result: audit.Report, corpus: audit.Corpus, *, elapsed: float) -> None:
    """One row per failing check, then the verdict.

    Only the failing ones. A table of forty-one rows where thirty-five say zero
    is a table whose useful part is below the fold, and the count of what passed
    is on the last line for anyone who wants it.
    """
    failing = result.failing()
    if failing:
        table = Table(title=None, box=None, pad_edge=False)
        table.add_column("check")
        table.add_column("hard")
        table.add_column("findings", justify="right")
        table.add_column("what it checks")
        for one in failing:
            colour = "red" if one.check.hard else "yellow"
            table.add_row(
                f"[{colour}]{one.check.id}[/{colour}]",
                "yes" if one.check.hard else "no",
                f"{len(one.findings):,}",
                one.check.title,
            )
        console.print(table)

    passed = len(result.results) - len(failing)
    console.print(
        f"{len(corpus.catalogs):,} catalogs, {audit.plural(len(result.results), 'check')}, "
        f"{passed} passed, {audit.plural(len(result.findings), 'finding')}, {elapsed:.1f}s"
    )
    if result.ok:
        console.print("[green]every hard check passes[/green]")
        return
    hard = result.failing(hard=True)
    console.print(f"[red]{audit.plural(len(hard), 'hard check')} failing[/red]")


@report_app.command("coverage")
def report_coverage(
    readme: Annotated[
        bool, typer.Option("--readme/--no-readme", help="Regenerate the README table too.")
    ] = True,
    branch: Annotated[str, typer.Option(help="Which version the catalogs are of.")] = (
        sync.DEFAULT_BRANCH
    ),
) -> None:
    """Write ``reports/coverage.md``, and the README table with it.

    Five columns, and only the human one may be called translated. The machine
    column is a corpus to review and the passthrough column is strings that
    never needed translating, and a project that adds the three together is how
    a README comes to claim a million translated words.
    """
    where = config.paths()
    if not where.content.exists():
        console.print(f"[red]no content repo at {where.content}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    corpus = audit.assemble(where, branch=branch)
    target = _written(where.published / "coverage.md", report.coverage(corpus))
    console.print(f"wrote {target}")
    if readme:
        _readme(where, corpus)
    _coverage_table(corpus)


def _readme(where: config.Paths, corpus: audit.Corpus) -> None:
    """Put the table back into the README, or say why it could not go there.

    A missing fence is a usage error rather than a crash, because the fix is one
    line in a Markdown file and the person running this is the person who can
    make it.
    """
    target = where.content / "README.md"
    if not target.exists():
        console.print(f"[yellow]no README at {target}, so nothing to regenerate[/yellow]")
        return
    try:
        rewritten = report.render(target.read_text(encoding="utf-8"), corpus)
    except report.ReportError as error:
        console.print(f"[red]{target}: {error}[/red]")
        raise typer.Exit(ExitCode.USAGE) from error
    target.write_text(rewritten, encoding="utf-8")
    console.print(f"wrote {target}")


def _coverage_table(corpus: audit.Corpus) -> None:
    """The same numbers on the terminal, so the command says what it did."""
    tiers = report.by_tier(corpus)
    table = Table(box=None)
    table.add_column("tier")
    table.add_column("entries", justify="right")
    for column in report.COLUMNS:
        table.add_column(str(column), justify="right")
    for number, count in sorted(tiers.items()):
        table.add_row(
            str(number),
            f"{count.total:,}",
            *(f"{count.entries[one]:,}" for one in report.COLUMNS),
        )
    console.print(table)


@report_app.command("quality")
def report_quality(
    limit: Annotated[int, typer.Option(help="Rows in each worst-first table.")] = 20,
    branch: Annotated[str, typer.Option(help="Which version the catalogs are of.")] = (
        sync.DEFAULT_BRANCH
    ),
) -> None:
    """Write ``reports/quality.md``: invariants, refusals, terminology, deaths.

    Everything in it is either a recount off the committed corpus or a number a
    run wrote down, and the two are labelled. Where there has been no run the
    section says so rather than printing zeros, because a refusal rate of 0.0 %
    reads as a perfect run and an absent one reads as absent.
    """
    where = config.paths()
    if not where.content.exists():
        console.print(f"[red]no content repo at {where.content}[/red]")
        raise typer.Exit(ExitCode.USAGE)
    corpus = audit.assemble(where, branch=branch)
    tallies = translate.Tally.read_all(where.tallies)
    target = _written(
        where.published / "quality.md", report.quality(corpus, tallies=tallies, limit=limit)
    )
    console.print(f"wrote {target}")
    if not tallies:
        console.print(
            "[yellow]no translation run on record, so the refusal rates are absent "
            "rather than zero[/yellow]"
        )


def _written(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def main() -> NoReturn:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(130)
    raise AssertionError("unreachable: typer exits via SystemExit")
