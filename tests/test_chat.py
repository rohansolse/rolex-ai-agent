"""The chat REPL, driven with a fake provider and scripted input."""

from collections.abc import Iterator

import pytest

from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.interface.chat import HISTORY_TURNS, _trim, run_chat
from rolex.llm.base import Message, Reply, ToolCall, ToolSchema
from rolex.tools.base import Tool


class FakeProvider:
    """A provider that answers from a script and records what it was sent."""

    def __init__(
        self,
        answers: list[str | Reply | Exception],
        *,
        available: bool = True,
        listing: list[str] | Exception | None = None,
    ) -> None:
        self.answers = answers
        self.available = available
        self.listing = listing
        self.seen: list[list[Message]] = []
        self.offered: list[list[ToolSchema]] = []

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
        self.seen.append(list(messages))
        self.offered.append(list(tools or []))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, Reply) else Reply(content=answer)

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        yield self.complete(messages, tools).content


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

    assert run_chat(Config(), provider=provider, tools=[]) == 0
    assert "hello there" in capsys.readouterr().out


def test_history_grows_so_the_model_has_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "first", "second")
    provider = FakeProvider(["one", "two"])

    run_chat(Config(), provider=provider, tools=[])

    first, second = provider.seen
    assert [m.role for m in first] == ["system", "user"]
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert second[2].content == "one"


@pytest.mark.parametrize("word", ["exit", "quit", "BYE", ":q"])
def test_quit_words_leave(monkeypatch: pytest.MonkeyPatch, word: str) -> None:
    _typed(monkeypatch, word, "never reached")
    provider = FakeProvider([])
    assert run_chat(Config(), provider=provider, tools=[]) == 0
    assert provider.seen == []


def test_blank_lines_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "", "   ", "hi")
    provider = FakeProvider(["ok"])
    run_chat(Config(), provider=provider, tools=[])
    assert len(provider.seen) == 1


def test_eof_leaves_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch)
    assert run_chat(Config(), provider=FakeProvider([]), tools=[]) == 0


def test_a_failed_turn_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "boom", "again")
    provider = FakeProvider([ProviderError("server fell over"), "recovered"])

    assert run_chat(Config(), provider=provider, tools=[]) == 0
    captured = capsys.readouterr()
    assert "server fell over" in captured.err
    assert "recovered" in captured.out


def test_a_failed_turn_leaves_no_dangling_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unanswered question is dropped, or every later turn carries it."""
    _typed(monkeypatch, "boom", "again")
    provider = FakeProvider([ProviderError("nope"), "fine"])

    run_chat(Config(), provider=provider, tools=[])

    _, retry = provider.seen
    assert [m.role for m in retry] == ["system", "user"]
    assert retry[1].content == "again"


def test_unavailable_provider_explains_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False)

    assert run_chat(Config(), provider=provider, tools=[]) == 1
    assert "not ready" in capsys.readouterr().err


def test_history_is_trimmed_but_keeps_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama truncates from the front, which would drop rolex's own name."""
    turns = 40
    _typed(monkeypatch, *[f"q{i}" for i in range(turns)])
    provider = FakeProvider([f"a{i}" for i in range(turns)])

    run_chat(Config(), provider=provider, tools=[])

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

    assert run_chat(Config(), provider=provider, tools=[]) == 1
    assert "start it with X" in capsys.readouterr().err


def test_a_reachable_server_is_not_blamed_for_being_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A garbled answer is a different problem from a dead server."""
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False, listing=ProviderError("garbled"))

    run_chat(Config(), provider=provider, tools=[])
    err = capsys.readouterr().err
    assert "not usefully" in err
    assert "not running" not in err


def test_a_missing_model_is_reported_with_the_pull_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider([], available=False, listing=["llama3.2:3b"])

    run_chat(Config(model="qwen2.5:3b"), provider=provider, tools=[])
    err = capsys.readouterr().err
    assert "ollama pull qwen2.5:3b" in err
    assert "llama3.2:3b" in err


def test_tools_are_offered_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "hi")
    provider = FakeProvider(["ok"])
    clock = Tool(name="get_time", description="the time", run=lambda _: "now")

    run_chat(Config(), provider=provider, tools=[clock])

    assert [schema.name for schema in provider.offered[0]] == ["get_time"]


def test_the_banner_names_the_tools(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch)
    clock = Tool(name="get_time", description="d", run=lambda _: "now")

    run_chat(Config(), provider=FakeProvider([]), tools=[clock])

    assert "Tools: get_time" in capsys.readouterr().out


def test_tool_work_is_shown_to_the_user(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Seeing the step is how the user knows the answer was not invented."""
    _typed(monkeypatch, "what time is it?")
    wants_clock = Reply(content="", tool_calls=(ToolCall("get_time", {}),))
    provider = FakeProvider([wants_clock, "It is 4pm"])
    clock = Tool(name="get_time", description="d", run=lambda _: "16:00")

    run_chat(Config(), provider=provider, tools=[clock])

    out = capsys.readouterr().out
    assert "get_time()" in out
    assert "16:00" in out
    assert "It is 4pm" in out


def test_a_long_tool_result_is_shortened_on_screen(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _typed(monkeypatch, "go")
    provider = FakeProvider([Reply(content="", tool_calls=(ToolCall("chatty", {}),)), "done"])
    chatty = Tool(name="chatty", description="d", run=lambda _: "x" * 500)

    run_chat(Config(), provider=provider, tools=[chatty])

    out = capsys.readouterr().out
    assert "…" in out
    assert "x" * 200 not in out


def test_trimming_never_starts_the_history_with_a_tool_result() -> None:
    """A tool result whose request was trimmed away confuses the model."""
    history = [
        Message("system", "s"),
        *[Message("user", f"q{i}") for i in range(HISTORY_TURNS)],
        Message("tool", "leftover", name="t"),
        Message("assistant", "a"),
    ]

    trimmed = _trim(history)

    assert trimmed[0].role == "system"
    assert trimmed[1].role != "tool"
