"""The chat REPL, driven with a fake provider and scripted input."""

from collections.abc import Iterator

import pytest

from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.interface.chat import HISTORY_TURNS, run_chat
from rolex.llm.base import Message, Reply, ToolSchema


class FakeProvider:
    """A provider that answers from a script and records what it was sent."""

    def __init__(
        self,
        answers: list[str | Exception],
        *,
        available: bool = True,
        listing: list[str] | Exception | None = None,
    ) -> None:
        self.answers = answers
        self.available = available
        self.listing = listing
        self.seen: list[list[Message]] = []

    def installed_models(self) -> list[str]:
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing or []

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return self.available

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        return Reply(content="".join(self.stream(messages, tools)))

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        self.seen.append(list(messages))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        yield from (part for part in answer.partition(" ") if part)


def _typed(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Script what the user types, then end the session."""
    queue = list(lines)

    def fake_input(_: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def test_answers_a_question(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider(["hello there"])

    assert run_chat(Config(), provider=provider) == 0
    assert "hello there" in capsys.readouterr().out


def test_history_grows_so_the_model_has_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "first", "second")
    provider = FakeProvider(["one", "two"])

    run_chat(Config(), provider=provider)

    first, second = provider.seen
    assert [m.role for m in first] == ["system", "user"]
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert second[2].content == "one"


@pytest.mark.parametrize("word", ["exit", "quit", "BYE", ":q"])
def test_quit_words_leave(monkeypatch: pytest.MonkeyPatch, word: str) -> None:
    _typed(monkeypatch, word, "never reached")
    provider = FakeProvider([])
    assert run_chat(Config(), provider=provider) == 0
    assert provider.seen == []


def test_blank_lines_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "", "   ", "hi")
    provider = FakeProvider(["ok"])
    run_chat(Config(), provider=provider)
    assert len(provider.seen) == 1


def test_eof_leaves_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch)
    assert run_chat(Config(), provider=FakeProvider([])) == 0


def test_a_failed_turn_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "boom", "again")
    provider = FakeProvider([ProviderError("server fell over"), "recovered"])

    assert run_chat(Config(), provider=provider) == 0
    captured = capsys.readouterr()
    assert "server fell over" in captured.err
    assert "recovered" in captured.out


def test_a_failed_turn_leaves_no_dangling_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unanswered question is dropped, or every later turn carries it."""
    _typed(monkeypatch, "boom", "again")
    provider = FakeProvider([ProviderError("nope"), "fine"])

    run_chat(Config(), provider=provider)

    _, retry = provider.seen
    assert [m.role for m in retry] == ["system", "user"]
    assert retry[1].content == "again"


def test_unavailable_provider_explains_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False)

    assert run_chat(Config(), provider=provider) == 1
    assert "not ready" in capsys.readouterr().err


def test_history_is_trimmed_but_keeps_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama truncates from the front, which would drop rolex's own name."""
    turns = 40
    _typed(monkeypatch, *[f"q{i}" for i in range(turns)])
    provider = FakeProvider([f"a{i}" for i in range(turns)])

    run_chat(Config(), provider=provider)

    last = provider.seen[-1]
    assert last[0].role == "system"
    assert "rolex" in last[0].content
    assert len(last) <= HISTORY_TURNS + 2
    assert last[-1].content == f"q{turns - 1}"


def test_a_down_server_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False, listing=UnreachableError("start it with X"))

    assert run_chat(Config(), provider=provider) == 1
    assert "start it with X" in capsys.readouterr().err


def test_a_reachable_server_is_not_blamed_for_being_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A garbled answer is a different problem from a dead server."""
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False, listing=ProviderError("garbled"))

    run_chat(Config(), provider=provider)
    err = capsys.readouterr().err
    assert "not usefully" in err
    assert "not running" not in err


def test_a_missing_model_is_reported_with_the_pull_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False, listing=["llama3.2:3b"])

    run_chat(Config(model="qwen2.5:3b"), provider=provider)
    err = capsys.readouterr().err
    assert "ollama pull qwen2.5:3b" in err
    assert "llama3.2:3b" in err
