"""The policy that decides whether a tool may run."""

from collections.abc import Mapping
from typing import Any

import pytest

from rolex.agent.consent import Policy, always_allow
from rolex.tools.base import Safety, Tool


def tool(safety: Safety, name: str = "t") -> Tool:
    return Tool(name=name, description="d", run=lambda _: "", safety=safety)


def yes(_: Tool, __: Mapping[str, Any]) -> bool:
    return True


def no(_: Tool, __: Mapping[str, Any]) -> bool:
    return False


# --- the defaults ---------------------------------------------------------


def test_read_only_runs_without_asking() -> None:
    asked = False

    def ask(_: Tool, __: Mapping[str, Any]) -> bool:
        nonlocal asked
        asked = True
        return True

    assert Policy(ask=ask).decide(tool("read_only"), {}).allowed
    assert not asked


def test_mutating_asks_first() -> None:
    questions: list[str] = []

    def ask(t: Tool, _: Mapping[str, Any]) -> bool:
        questions.append(t.name)
        return True

    assert Policy(ask=ask).decide(tool("mutating", "open_app"), {}).allowed
    assert questions == ["open_app"]


def test_a_refusal_is_honoured() -> None:
    decision = Policy(ask=no).decide(tool("mutating"), {})
    assert not decision.allowed
    assert "refused permission" in decision.reason


def test_destructive_is_refused_even_if_the_user_would_say_yes() -> None:
    """Off by default: a confused model must not be one click from damage."""
    decision = Policy(ask=yes).decide(tool("destructive", "wipe"), {})
    assert not decision.allowed
    assert "ROLEX_ALLOW_DESTRUCTIVE" in decision.reason


def test_destructive_asks_once_it_is_allowed() -> None:
    policy = Policy(ask=yes, allow_destructive=True)
    assert policy.decide(tool("destructive"), {}).allowed


def test_destructive_still_respects_a_refusal_when_allowed() -> None:
    policy = Policy(ask=no, allow_destructive=True)
    assert not policy.decide(tool("destructive"), {}).allowed


# --- when nobody can be asked --------------------------------------------


def test_without_an_asker_anything_needing_consent_is_refused() -> None:
    """A piped or scheduled run has no one at the keyboard."""
    decision = Policy(ask=None).decide(tool("mutating"), {})
    assert not decision.allowed
    assert "no way to ask" in decision.reason


def test_without_an_asker_read_only_still_runs() -> None:
    assert Policy(ask=None).decide(tool("read_only"), {}).allowed


# --- the reasons are written for the model -------------------------------


@pytest.mark.parametrize(
    "policy",
    [Policy(ask=no), Policy(ask=None), Policy(ask=yes)],
)
def test_every_refusal_tells_the_model_it_did_not_happen(policy: Policy) -> None:
    """Otherwise the model reports success for something that never ran."""
    for safety in ("mutating", "destructive"):
        decision = policy.decide(tool(safety), {})
        if not decision.allowed:
            assert decision.reason.startswith("Not done:")


def test_the_arguments_are_shown_to_the_asker() -> None:
    seen: list[Mapping[str, Any]] = []

    def ask(_: Tool, arguments: Mapping[str, Any]) -> bool:
        seen.append(arguments)
        return True

    Policy(ask=ask).decide(tool("mutating"), {"name": "Safari"})

    assert seen == [{"name": "Safari"}]


def test_always_allow_permits_everything_unasked() -> None:
    policy = always_allow()
    for safety in ("read_only", "mutating", "destructive"):
        assert policy.decide(tool(safety), {}).allowed
