import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

import pytest
from typer.testing import CliRunner

import pydocvi
from conftest import FAKE_KEY, FakeRunner
from pydocvi import __version__, config, fleet, glossary, mine
from pydocvi.catalog import segment_id
from pydocvi.cli import ExitCode, app, main
from pydocvi.client import Answer
from pydocvi.fleet import Ran
from pydocvi.queue import Job, Queue, Stage, State
from pydocvi.routes import Route

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch upstream checkout and work directory, wired through the environment.

    A git repository rather than a plain directory because ``sync`` records the
    commit it ran against, and a pin with no commit in it is not a pin.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    shutil.copy(data_dir / "small.po", upstream / "bugs.po")
    for command in (
        ["git", "init", "-q", "-b", "3.15"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "corpus"],
    ):
        subprocess.run(command, cwd=upstream, check=True, capture_output=True)
    monkeypatch.setenv("PYDOCVI_UPSTREAM", str(upstream))
    monkeypatch.setenv("PYDOCVI_WORK", str(tmp_path / "work"))
    return tmp_path


def test_version_prints_the_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == ExitCode.OK
    assert result.stdout.strip() == __version__


def test_no_arguments_shows_help_rather_than_a_traceback() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == ExitCode.USAGE
    assert "Usage" in result.stdout


def test_unknown_command_is_a_usage_error() -> None:
    result = runner.invoke(app, ["nope"])
    assert result.exit_code == ExitCode.USAGE


def test_verbose_is_accepted_before_the_command() -> None:
    result = runner.invoke(app, ["--verbose", "version"])
    assert result.exit_code == ExitCode.OK


def test_module_entry_point_runs() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "pydocvi", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == __version__


def test_main_exits_through_typer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pydocvi", "version"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == ExitCode.OK


def test_ctrl_c_exits_130_rather_than_printing_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("pydocvi.cli.app", interrupt)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 130


class TestSync:
    def test_reports_the_counts_and_writes_the_pin(self, workspace: Path) -> None:
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == ExitCode.OK
        pin = (workspace / "work" / "manifests" / "upstream.yaml").read_text(encoding="utf-8")
        assert 'branch: "3.15"' in pin
        assert "entries: 4" in pin
        assert (workspace / "work" / "reports" / "sync.md").exists()
        assert (workspace / "work" / "memory.json").exists()

    def test_dry_run_writes_nothing(self, workspace: Path) -> None:
        result = runner.invoke(app, ["sync", "--dry-run"])
        assert result.exit_code == ExitCode.OK
        assert not (workspace / "work").exists()

    def test_human_loads_translated_entries_into_the_memory(self, workspace: Path) -> None:
        runner.invoke(app, ["sync", "--human"])
        result = runner.invoke(app, ["tm", "stats"])
        assert "human" in result.stdout
        assert "total" in result.stdout

    def test_a_missing_upstream_checkout_is_a_failed_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYDOCVI_UPSTREAM", str(tmp_path / "absent"))
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "no upstream checkout" in result.stdout


class TestTm:
    def test_stats_on_an_empty_memory_still_prints_a_table(self, workspace: Path) -> None:
        result = runner.invoke(app, ["tm", "stats"])
        assert result.exit_code == ExitCode.OK
        assert "machine" in result.stdout

    def test_show_finds_a_stored_segment(self, workspace: Path) -> None:
        runner.invoke(app, ["sync", "--human"])
        wanted = segment_id("Dealing with Bugs")
        result = runner.invoke(app, ["tm", "show", wanted])
        assert result.exit_code == ExitCode.OK
        assert wanted in result.stdout

    def test_show_of_an_unknown_id_is_a_failed_check(self, workspace: Path) -> None:
        result = runner.invoke(app, ["tm", "show", "0" * 16])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "no segment" in result.stdout


class TestClassify:
    def test_prints_a_kind_for_every_entry(self, workspace: Path) -> None:
        result = runner.invoke(app, ["classify"])
        assert result.exit_code == ExitCode.OK
        assert "prose" in result.stdout
        assert "never sent to a model" in result.stdout

    def test_report_writes_the_breakdown(self, workspace: Path) -> None:
        result = runner.invoke(app, ["classify", "--report"])
        assert result.exit_code == ExitCode.OK
        report = (workspace / "work" / "reports" / "classify.md").read_text(encoding="utf-8")
        assert report.startswith("# Classification")
        assert "| kind | entries | share |" in report

    def test_a_missing_upstream_checkout_is_a_failed_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYDOCVI_UPSTREAM", str(tmp_path / "absent"))
        assert runner.invoke(app, ["classify"]).exit_code == ExitCode.CHECK_FAILED


class TestBatch:
    def test_prints_the_batch_and_file_counts(self, workspace: Path) -> None:
        result = runner.invoke(app, ["batch"])
        assert result.exit_code == ExitCode.OK
        assert "batches over" in result.stdout

    def test_stats_lists_the_heaviest_files(self, workspace: Path) -> None:
        result = runner.invoke(app, ["batch", "--stats", "--top", "3"])
        assert result.exit_code == ExitCode.OK
        assert "entries per batch" in result.stdout
        assert "bugs.po" in result.stdout

    def test_a_missing_upstream_checkout_is_a_failed_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYDOCVI_UPSTREAM", str(tmp_path / "absent"))
        assert runner.invoke(app, ["batch"]).exit_code == ExitCode.CHECK_FAILED


@pytest.fixture
def route_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A route file with the shape of a real one and none of its addresses.

    Pointed at by the environment, because the real one lives in the user's
    config directory and is never part of a checkout.
    """
    target = tmp_path / "routes.json"
    target.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "name": "a",
                        "base_url": "http://127.0.0.1:8103/v1",
                        "model": "gpt-5",
                        "host": "a",
                        "local_port": 8103,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYDOCVI_ROUTES", str(target))
    monkeypatch.setenv("CHATGPT_PROXY_KEY", FAKE_KEY)
    return target


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    """Every subprocess the fleet would run, answered from a dictionary."""
    fake = FakeRunner({})
    monkeypatch.setattr(fleet, "Subprocess", lambda: fake)
    return fake


class TestFleetCommands:
    def test_status_lists_every_route(self, route_file: Path, commands: FakeRunner) -> None:
        result = runner.invoke(app, ["fleet", "status"])
        assert result.exit_code == ExitCode.OK
        assert "127.0.0.1:8103" in result.stdout

    def test_up_opens_a_forward(self, route_file: Path, commands: FakeRunner) -> None:
        commands.answers = {"lsof": Ran(code=1)}
        assert runner.invoke(app, ["fleet", "up"]).exit_code == ExitCode.OK
        assert commands.ran("ExitOnForwardFailure=yes")

    def test_up_failing_is_exit_three(self, route_file: Path, commands: FakeRunner) -> None:
        """Exit 3 is what the runbook keys on, so a tunnel that did not open has
        to stop the script rather than let a nine-hour run start."""
        commands.answers = {"lsof": Ran(code=1), "ssh": Ran(code=255, err="denied")}
        result = runner.invoke(app, ["fleet", "up"])
        assert result.exit_code == ExitCode.FLEET_UNREACHABLE
        assert "denied" in result.stdout

    def test_down_closes_what_is_open(self, route_file: Path, commands: FakeRunner) -> None:
        commands.answers = {"lsof": Ran(code=0, out="4242")}
        result = runner.invoke(app, ["fleet", "down"])
        assert result.exit_code == ExitCode.OK
        assert commands.ran("kill 4242")

    def test_probe_with_nothing_answering_is_exit_three(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        commands.answers = {"curl": Ran(code=0, out="000")}
        assert runner.invoke(app, ["fleet", "probe"]).exit_code == ExitCode.FLEET_UNREACHABLE

    def test_a_missing_route_file_names_the_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Named because the commonest cause is looking in the wrong place, and
        the path is written in the platform's own separators."""
        absent = Path("/nonexistent/routes.json")
        monkeypatch.setenv("PYDOCVI_ROUTES", str(absent))
        result = runner.invoke(app, ["fleet", "status"])
        assert result.exit_code == ExitCode.FLEET_UNREACHABLE
        assert str(absent) in result.stdout

    def test_trace_prints_the_prompt_and_reply(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        commands.answers = {
            "grep -rl": Ran(code=0, out="/traces/aa.json"),
            "cat ": Ran(code=0, out="the prompt and the reply"),
        }
        result = runner.invoke(app, ["fleet", "trace", "3f2a1c", "--day", "2026-08-15"])
        assert result.exit_code == ExitCode.OK
        assert "the prompt and the reply" in result.stdout

    def test_trace_for_a_batch_nobody_has(self, route_file: Path, commands: FakeRunner) -> None:
        commands.answers = {"grep -rl": Ran(code=0, out="")}
        result = runner.invoke(app, ["fleet", "trace", "3f2a1c", "--day", "2026-08-15"])
        assert result.exit_code == ExitCode.CHECK_FAILED

    def test_trace_on_a_route_that_does_not_exist(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        result = runner.invoke(
            app, ["fleet", "trace", "3f2a1c", "--day", "2026-08-15", "--route", "nope"]
        )
        assert result.exit_code == ExitCode.USAGE

    def test_bench_asks_before_spending_real_calls(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        """Twenty calls a route is most of an hour of a shared session, so the
        confirmation is not decoration."""
        result = runner.invoke(app, ["fleet", "bench"], input="n\n")
        assert result.exit_code != ExitCode.OK


class TestDoctor:
    def test_a_fleet_that_is_answering_is_exit_zero(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        commands.answers = {"curl": Ran(code=0, out="200")}
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == ExitCode.OK
        assert "1 of 1 routes answering" in result.stdout

    def test_a_fleet_that_is_down_is_exit_three(
        self, route_file: Path, commands: FakeRunner
    ) -> None:
        commands.answers = {"curl": Ran(code=7, err="Failed to connect")}
        assert runner.invoke(app, ["doctor"]).exit_code == ExitCode.FLEET_UNREACHABLE

    def test_a_key_that_is_not_in_the_shell_is_named(
        self, route_file: Path, commands: FakeRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The commonest way a run fails to start, and the one that used to look
        like a model problem."""
        monkeypatch.delenv("CHATGPT_PROXY_KEY")
        commands.answers = {"curl": Ran(code=0, out="200")}
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == ExitCode.FLEET_UNREACHABLE
        assert "CHATGPT_PROXY_KEY is not set" in result.stdout


class TestQueueCommands:
    def add_job(self, workspace: Path, **overrides: object) -> Queue:
        each = Queue(Path(os.environ["PYDOCVI_WORK"]) / "queue", Stage.TRANSLATE)
        each.add(Job(id="job001", stage=Stage.TRANSLATE, payload={"file": "bugs.po"}, **overrides))  # type: ignore[arg-type]
        return each

    def test_stats_on_a_queue_nobody_has_used(self, workspace: Path) -> None:
        result = runner.invoke(app, ["queue", "stats"])
        assert result.exit_code == ExitCode.OK
        assert "outstanding: 0" in result.stdout

    def test_stats_counts_what_is_waiting(self, workspace: Path) -> None:
        self.add_job(workspace)
        result = runner.invoke(app, ["queue", "stats"])
        assert "outstanding: 1" in result.stdout
        assert "translate" in result.stdout

    def test_reap_returns_an_expired_lease(self, workspace: Path) -> None:
        each = self.add_job(workspace)
        each.claim(now=0.0)
        result = runner.invoke(app, ["queue", "reap"])
        assert "reaped 1 expired lease(s)" in result.stdout
        assert each.count(State.PENDING) == 1

    def test_reap_leaves_a_lease_that_is_still_running(self, workspace: Path) -> None:
        each = self.add_job(workspace)
        each.claim(now=time.time())
        runner.invoke(app, ["queue", "reap"])
        assert each.count(State.LEASED) == 1

    def test_dead_lists_nothing_when_nothing_died(self, workspace: Path) -> None:
        result = runner.invoke(app, ["queue", "dead"])
        assert result.exit_code == ExitCode.OK
        assert "nothing dead" in result.stdout

    def test_dead_says_why_each_job_died(self, workspace: Path) -> None:
        each = self.add_job(workspace, attempts=3)
        claimed = each.claim(now=0.0)
        assert claimed is not None
        each.release(claimed, error="empty answer")
        result = runner.invoke(app, ["queue", "dead"])
        assert "empty answer" in result.stdout

    def test_retry_is_the_only_way_back_from_dead(self, workspace: Path) -> None:
        each = self.add_job(workspace, attempts=3)
        claimed = each.claim(now=0.0)
        assert claimed is not None
        each.release(claimed, error="down")
        result = runner.invoke(app, ["queue", "retry"])
        assert "1 job(s) back to pending" in result.stdout
        assert each.count(State.PENDING) == 1

    def test_drain_asks_first(self, workspace: Path) -> None:
        each = self.add_job(workspace)
        claimed = each.claim(now=0.0)
        assert claimed is not None
        each.finish(claimed)
        assert runner.invoke(app, ["queue", "drain"], input="n\n").exit_code != ExitCode.OK
        assert each.count(State.DONE) == 1

    def test_drain_removes_finished_work_only(self, workspace: Path) -> None:
        each = self.add_job(workspace)
        claimed = each.claim(now=0.0)
        assert claimed is not None
        each.finish(claimed)
        result = runner.invoke(app, ["queue", "drain", "--yes"])
        assert "removed 1 finished job(s)" in result.stdout


def test_version_falls_back_when_the_package_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source tree with no install still has a usable ``__version__``."""

    def not_installed(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    reloaded = importlib.reload(pydocvi)
    try:
        assert reloaded.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(pydocvi)


@pytest.fixture
def content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch content repo, which is where the glossary lives."""
    where = tmp_path / "content"
    where.mkdir()
    monkeypatch.setenv("PYDOCVI_CONTENT", str(where))
    return where


def write_glossary(content: Path, text: str) -> Path:
    target = content / "manifests" / "glossary.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


class TestGlossaryMine:
    def test_mining_writes_the_candidates_file(self, workspace: Path, content: Path) -> None:
        result = runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        assert result.exit_code == ExitCode.OK
        assert (content / "manifests" / "glossary-candidates.yaml").exists()

    def test_the_counts_are_printed_by_source(self, workspace: Path, content: Path) -> None:
        result = runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        assert "frequency" in result.stdout and "total" in result.stdout

    def test_what_it_writes_reads_back(self, workspace: Path, content: Path) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        text = (content / "manifests" / "glossary-candidates.yaml").read_text(encoding="utf-8")
        assert mine.loads(text)

    def test_no_upstream_is_a_failed_check_rather_than_a_traceback(
        self, content: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYDOCVI_UPSTREAM", "/nonexistent/upstream")
        assert runner.invoke(app, ["glossary", "mine"]).exit_code == ExitCode.CHECK_FAILED


class FakeCompletions:
    """A client that answers every batch from a function of its prompt."""

    def __init__(self, answer: Callable[[str], str]) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def complete(self, route: Route, prompt: str, *, system: str | None = None) -> Answer:
        self.prompts.append(prompt)
        return Answer(text=self.answer(prompt), route=route.name, model=route.model, seconds=1.0)


def echo_renderings(prompt: str) -> str:
    """Answer every numbered term with a plausible Vietnamese phrase."""
    lines = []
    for line in prompt.splitlines():
        match = re.match(r"^(\d+)\. (.+)$", line)
        if match:
            lines.append(f"{match.group(1)}. {match.group(2)} = bản dịch")
    return "\n".join(lines)


@pytest.fixture
def completions(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    fake = FakeCompletions(echo_renderings)
    monkeypatch.setattr("pydocvi.cli.Client", lambda: fake)
    return fake


class TestGlossaryCurate:
    def test_curating_with_no_candidates_says_which_command_to_run(
        self, workspace: Path, content: Path, route_file: Path
    ) -> None:
        result = runner.invoke(app, ["glossary", "curate", "--yes"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "glossary mine" in result.stdout.replace("\n", "")

    def test_a_run_writes_the_proposal_and_not_the_glossary(
        self,
        workspace: Path,
        content: Path,
        route_file: Path,
        completions: FakeCompletions,
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        result = runner.invoke(app, ["glossary", "curate", "--yes"])
        assert result.exit_code == ExitCode.OK
        assert (content / "manifests" / "glossary-proposed.yaml").exists()
        assert not (content / "manifests" / "glossary.yaml").exists()

    def test_the_proposal_is_at_version_zero_because_nobody_has_read_it(
        self,
        workspace: Path,
        content: Path,
        route_file: Path,
        completions: FakeCompletions,
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        runner.invoke(app, ["glossary", "curate", "--yes"])
        proposed = glossary.load(content / "manifests" / "glossary-proposed.yaml")
        assert proposed.version == 0

    def test_the_report_lands_under_work(
        self,
        workspace: Path,
        content: Path,
        route_file: Path,
        completions: FakeCompletions,
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        runner.invoke(app, ["glossary", "curate", "--yes"])
        assert (workspace / "work" / "reports" / "glossary-curation.md").exists()

    def test_take_limits_how_many_terms_are_asked_about(
        self,
        workspace: Path,
        content: Path,
        route_file: Path,
        completions: FakeCompletions,
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "20"])
        runner.invoke(app, ["glossary", "curate", "--yes", "--take", "2", "--batch", "1"])
        assert len(completions.prompts) == 2


class TestARunThatAcceptedNothing:
    """The fleet went down part way through a real run and every call failed.

    The proposal was written anyway, so 873 reviewed rows became an empty file
    and the only copy of that work was gone. A run with nothing in it must not
    be able to do that.
    """

    @pytest.fixture
    def failing(self, monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
        def refuse(prompt: str) -> str:
            raise RuntimeError("connection reset")

        fake = FakeCompletions(refuse)
        monkeypatch.setattr("pydocvi.cli.Client", lambda: fake)
        return fake

    def test_the_earlier_proposal_is_left_alone(
        self, workspace: Path, content: Path, route_file: Path, failing: FakeCompletions
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        earlier = content / "manifests" / "glossary-proposed.yaml"
        earlier.parent.mkdir(parents=True, exist_ok=True)
        earlier.write_text('version: 0\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        runner.invoke(app, ["glossary", "curate", "--yes"])
        assert glossary.load(earlier).terms[0].en == "iterable"

    def test_it_exits_as_an_unreachable_fleet_rather_than_a_success(
        self, workspace: Path, content: Path, route_file: Path, failing: FakeCompletions
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        result = runner.invoke(app, ["glossary", "curate", "--yes"])
        assert result.exit_code == ExitCode.FLEET_UNREACHABLE

    def test_it_says_the_proposal_was_left_as_it_was(
        self, workspace: Path, content: Path, route_file: Path, failing: FakeCompletions
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        result = runner.invoke(app, ["glossary", "curate", "--yes"])
        assert "left as it was" in result.stdout.replace("\n", "")

    def test_the_report_is_still_written_because_a_failure_is_worth_a_record(
        self, workspace: Path, content: Path, route_file: Path, failing: FakeCompletions
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        runner.invoke(app, ["glossary", "curate", "--yes"])
        assert (workspace / "work" / "reports" / "glossary-curation.md").exists()

    def test_no_proposal_is_created_where_there_was_none(
        self, workspace: Path, content: Path, route_file: Path, failing: FakeCompletions
    ) -> None:
        runner.invoke(app, ["glossary", "mine", "--minimum", "1", "--limit", "5"])
        runner.invoke(app, ["glossary", "curate", "--yes"])
        assert not (content / "manifests" / "glossary-proposed.yaml").exists()


class TestGlossaryCheck:
    def test_a_clean_glossary_passes(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        result = runner.invoke(app, ["glossary", "check"])
        assert result.exit_code == ExitCode.OK
        assert "1 terms" in result.stdout

    def test_a_collision_fails_and_names_the_rule(self, content: Path) -> None:
        write_glossary(
            content,
            'version: 1\nterms:\n  - en: "bug"\n    vi: "lỗi"\n  - en: "mistake"\n    vi: "lỗi"\n',
        )
        result = runner.invoke(app, ["glossary", "check"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "G-e" in result.stdout

    def test_a_markdown_table_that_disagrees_fails(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "GLOSSARY.md").write_text(
            f"# Terms\n\n{glossary.TABLE_OPEN}\n\n{glossary.TABLE_CLOSE}\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["glossary", "check"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "G05" in result.stdout

    def test_a_missing_glossary_names_the_path(self, content: Path) -> None:
        """Neither the separator nor the wrapping is part of the message.

        A temporary directory is long enough that the console wraps the path on
        some runners and not others, and Windows writes the separator the other
        way round. Both of those failed this assertion while the command was
        doing exactly the right thing.
        """
        result = runner.invoke(app, ["glossary", "check"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        printed = result.stdout.replace("\n", "").replace("\\", "/")
        assert str(config.paths().glossary).replace("\\", "/") in printed


class TestGlossaryShow:
    def test_the_whole_table_prints(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        assert "iterable" in runner.invoke(app, ["glossary", "show"]).stdout

    def test_one_term_prints(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        assert runner.invoke(app, ["glossary", "show", "iterable"]).exit_code == ExitCode.OK

    def test_a_term_that_is_not_there_is_a_failed_check(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        assert runner.invoke(app, ["glossary", "show", "nope"]).exit_code == ExitCode.CHECK_FAILED


class TestGlossaryBump:
    def test_promoting_raises_the_version_and_writes_the_glossary(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "decorator"\n    vi: "decorator"\n    keep_en: true\n',
            encoding="utf-8",
        )
        result = runner.invoke(app, ["glossary", "bump", "--yes"])
        assert result.exit_code == ExitCode.OK
        assert glossary.load(content / "manifests" / "glossary.yaml").version == 2

    def test_the_old_version_is_archived_so_diff_has_two_sides(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "decorator"\n    vi: "decorator"\n    keep_en: true\n',
            encoding="utf-8",
        )
        runner.invoke(app, ["glossary", "bump", "--yes"])
        assert (content / "manifests" / "glossary" / "v1.yaml").exists()

    def test_the_proposal_is_consumed(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "decorator"\n    vi: "decorator"\n    keep_en: true\n',
            encoding="utf-8",
        )
        runner.invoke(app, ["glossary", "bump", "--yes"])
        assert not (content / "manifests" / "glossary-proposed.yaml").exists()

    def test_the_markdown_table_is_regenerated(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "GLOSSARY.md").write_text(
            f"# Terms\n\n{glossary.TABLE_OPEN}\n\n{glossary.TABLE_CLOSE}\n", encoding="utf-8"
        )
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "decorator"\n    vi: "decorator"\n    keep_en: true\n',
            encoding="utf-8",
        )
        runner.invoke(app, ["glossary", "bump", "--yes"])
        assert "decorator" in (content / "GLOSSARY.md").read_text(encoding="utf-8")

    def test_a_proposal_that_collides_is_refused_before_anything_is_written(
        self, content: Path
    ) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "bug"\n    vi: "lỗi"\n')
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "mistake"\n    vi: "lỗi"\n', encoding="utf-8"
        )
        result = runner.invoke(app, ["glossary", "bump", "--yes"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert glossary.load(content / "manifests" / "glossary.yaml").version == 1

    def test_promoting_nothing_new_does_not_move_the_version(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        (content / "manifests" / "glossary-proposed.yaml").write_text(
            'version: 0\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n', encoding="utf-8"
        )
        result = runner.invoke(app, ["glossary", "bump", "--yes"])
        assert "nothing to promote" in result.stdout

    def test_no_proposal_says_which_command_to_run(self, content: Path) -> None:
        write_glossary(content, 'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        result = runner.invoke(app, ["glossary", "bump", "--yes"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "glossary curate" in result.stdout.replace("\n", "")


class TestGlossaryDiff:
    def test_what_moved_between_two_versions_prints(self, content: Path) -> None:
        write_glossary(content, 'version: 2\nterms:\n  - en: "iterable"\n    vi: "lặp được"\n')
        archive = content / "manifests" / "glossary"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "v1.yaml").write_text(
            'version: 1\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n', encoding="utf-8"
        )
        result = runner.invoke(app, ["glossary", "diff", "1", "2"])
        assert result.exit_code == ExitCode.OK
        assert "khả lặp -> lặp được" in result.stdout

    def test_a_version_nobody_archived_names_the_file(self, content: Path) -> None:
        write_glossary(content, 'version: 2\nterms:\n  - en: "iterable"\n    vi: "khả lặp"\n')
        result = runner.invoke(app, ["glossary", "diff", "1", "2"])
        assert result.exit_code == ExitCode.CHECK_FAILED
        assert "v1.yaml" in result.stdout.replace("\n", "")
