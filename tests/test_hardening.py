"""Release diagnostics and recoverable failures through normal entry points."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from rolex.agent.loop import Agent
from rolex.core.config import Config, load_config
from rolex.core.errors import ConfigError, ProviderError
from rolex.core.logging import configure_logging
from rolex.interface.cli import main
from rolex.llm.base import Message, Reply, ToolCall
from rolex.llm.ollama import OllamaProvider
from rolex.tools.base import Tool
from test_chat import FakeProvider
from test_ollama_provider import FakeTransport


@pytest.fixture
def restore_logger() -> Iterator[None]:
    logger = logging.getLogger("rolex")
    state = (logger.handlers[:], logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = state


@pytest.mark.usefixtures("restore_logger")
def test_debug_records_rounds_and_tool_failure_without_private_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("DEBUG", structured=True)

    def fail(args: object) -> str:
        raise RuntimeError("private customer message")

    provider = FakeProvider(
        [
            Reply("", (ToolCall("read_info", {"text": "private argument"}),)),
            Reply("Recovered"),
        ]
    )
    agent = Agent(provider, tools=[Tool(name="read_info", description="Read info", run=fail)])
    list(agent.ask("private user prompt"))
    output = capsys.readouterr().err
    records = [json.loads(line) for line in output.splitlines()]
    assert any(row["event"] == "model_request" and row["round"] == 2 for row in records)
    assert any(
        row["event"] == "tool_failed" and row["error_type"] == "RuntimeError" for row in records
    )
    assert any("elapsed_ms" in row for row in records)
    assert "private" not in output


@pytest.mark.parametrize("argv", [["--debug", "chat"], ["chat", "--debug"]])
@pytest.mark.usefixtures("restore_logger")
def test_debug_flag_reaches_chat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    monkeypatch.setattr("rolex.interface.cli.load_config", lambda: Config())

    def chat(config: Config) -> int:
        logging.getLogger("rolex.interface.chat").debug("test_event")
        return 0

    monkeypatch.setattr("rolex.interface.cli.run_chat", chat)
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().err)["event"] == "test_event"


@pytest.mark.parametrize("error, status", [(PermissionError(), 1), (KeyboardInterrupt(), 130)])
@pytest.mark.usefixtures("restore_logger")
def test_cli_reports_device_failure_and_interrupt_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    status: int,
) -> None:
    monkeypatch.setattr("rolex.interface.cli.load_config", lambda: Config())

    def chat(config: Config) -> int:
        raise error

    monkeypatch.setattr("rolex.interface.cli.run_chat", chat)
    assert main(["chat"]) == status
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_timeout_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="request_timeout"):
        load_config(root=tmp_path, env={"ROLEX_REQUEST_TIMEOUT": value})


def test_malformed_tool_calls_report_provider_error() -> None:
    provider = OllamaProvider(
        Config(), FakeTransport(reply={"message": {"content": "", "tool_calls": 42}})
    )
    with pytest.raises(ProviderError, match="tool_calls"):
        provider.complete([Message("user", "hello")])
