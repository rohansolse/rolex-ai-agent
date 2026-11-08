"""Launch requests bypass the browser-only model planner, preserving consent."""

from collections.abc import Mapping
from typing import Any

import pytest

from rolex.agent.applications import matches_installed, requested_application
from rolex.agent.intent import CommandState, validate
from rolex.agent.workflow import Workflow
from rolex.core.config import Config
from rolex.core.errors import ProviderError, ToolError
from rolex.interface.chat import run_chat
from rolex.tools.base import Tool
from test_workflow import Planner, plan
from workflow_fixture import FixtureBrowser


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("hello open chrome", "Google Chrome"),
        ("i aked to open chrome", "Google Chrome"),
        ("I asked you to open Safari.", "Safari"),
        ("Hi, please launch Visual Studio Code", "Visual Studio Code"),
        ("Chrome open kara", "Google Chrome"),
        ("Chrome खोलो।", "Google Chrome"),
        ("Calculator उघड", "Calculator"),
    ],
)
def test_installed_application_request_runs_without_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    said: str,
    name: str,
) -> None:
    launched: list[Mapping[str, Any]] = []

    def launch(arguments: Mapping[str, Any]) -> str:
        launched.append(arguments)
        return f"Opened {arguments['name']}."

    monkeypatch.setattr(
        "rolex.interface.processor.installed_apps",
        lambda: ["Google Chrome", "Safari", "Visual Studio Code", "Calculator"],
    )
    tool = Tool("open_app", "Launch an app", launch, safety="mutating")
    lines = iter([said, "y", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert run_chat(Config(), provider=Planner([]), tools=[tool]) == 0
    assert launched == [{"name": name}]
    output = capsys.readouterr().out
    assert "Cancelled" not in output
    assert f"Opened {name}." in output


@pytest.mark.parametrize("denied", [True, False])
def test_app_launch_denial_and_failure_never_claim_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], denied: bool
) -> None:
    calls: list[str] = []

    def fail(arguments: Mapping[str, Any]) -> str:
        calls.append(str(arguments["name"]))
        raise ToolError("Chrome launch failed")

    monkeypatch.setattr("rolex.interface.processor.installed_apps", lambda: ["Google Chrome"])
    tool = Tool("open_app", "Launch", fail, safety="mutating")
    lines = iter(["hello open chrome", "n" if denied else "y", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert run_chat(Config(), provider=Planner([]), tools=[tool]) == 0
    assert calls == ([] if denied else ["Google Chrome"])
    output = capsys.readouterr().out
    assert "Opened chrome" not in output and "Cancelled" not in output
    assert "permission" in output if denied else "launch failed" in output


@pytest.mark.parametrize(
    "said",
    [
        "How do I open Chrome?",
        "Don't open Chrome",
        "open Chrome and send hi",
        "write 'open Chrome'",
    ],
)
def test_questions_negation_and_compounds_are_not_direct_launches(said: str) -> None:
    assert requested_application(said) is None


def test_chat_and_browser_targets_cannot_match_application_substrings() -> None:
    assert matches_installed("Chrome", ["Google Chrome"])
    assert not matches_installed("chat", ["ChatGPT"])
    assert not matches_installed("chrome tab", ["Google Chrome"])


@pytest.mark.parametrize("said", ["hello open chrome", "Open two tabs", "write 'cancel'"])
def test_model_cannot_invent_cancellation(said: str) -> None:
    from rolex.agent.intent import Step

    with pytest.raises(ProviderError, match="Cancellation was not requested"):
        validate(plan(Step("cancel")), said)


def test_unrelated_system_request_skips_browser_planner() -> None:
    from rolex.agent.consent import always_allow
    from rolex.agent.intent import Step

    provider = Planner([plan(Step("cancel"))])
    workflow = Workflow(provider, CommandState(), always_allow(), None, FixtureBrowser())
    assert workflow.process("How much battery is left?") is None
    assert len(provider.plans) == 1
