import importlib
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import pydocvi
from pydocvi import __version__
from pydocvi.cli import ExitCode, app, main

runner = CliRunner()


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
