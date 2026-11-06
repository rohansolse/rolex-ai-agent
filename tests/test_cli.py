"""Phase 0 smoke tests: the package imports and the CLI runs."""

import pytest

from rolex import __version__
from rolex.core.errors import ConfigError, ProviderError, RolexError, ToolError
from rolex.interface.cli import main


def test_version_is_a_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_main_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "rolex" in capsys.readouterr().out


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_help_flag_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "usage: rolex" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["chta"], ["--versoin"], ["--version", "extra"]])
def test_unrecognised_arguments_fail(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert "unrecognised" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("error", [ConfigError, ProviderError, ToolError])
def test_every_error_inherits_from_rolex_error(error: type[RolexError]) -> None:
    assert issubclass(error, RolexError)
