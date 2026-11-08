"""The fixed benchmark evaluates model output and never executes tools."""

import pytest

from rolex.core.config import Config
from rolex.interface.benchmark import CASES, evaluate
from rolex.llm.base import Reply
from test_chat import FakeProvider


def test_benchmark_scores_exact_calls_without_executing_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("benchmark must not execute a browser or system action")

    monkeypatch.setattr("subprocess.run", forbidden)
    answers: list[str | Reply | Exception] = [
        Reply("" if case.expected else "An answer", case.expected) for case in CASES
    ]
    result = evaluate(FakeProvider(answers), Config())
    assert result["passed"] == result["total"] == len(CASES)


def test_benchmark_reports_errors_and_wrong_calls() -> None:
    from rolex.core.errors import ProviderError

    answers: list[str | Reply | Exception] = [ProviderError("offline")]
    answers.extend(Reply("A guessed answer") for _ in CASES[1:])
    result = evaluate(FakeProvider(answers), Config())
    assert result["passed"] == 2
    assert result["cases"][0]["error"] == "offline"
