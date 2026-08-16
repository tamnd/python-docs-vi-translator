import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import pydocvi
from pydocvi import __version__
from pydocvi.catalog import segment_id
from pydocvi.cli import ExitCode, app, main

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
