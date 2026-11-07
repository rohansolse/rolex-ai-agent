"""Carrying out a recognised command, and when not to."""

from collections.abc import Iterator

import pytest

from rolex.agent.commands import Match
from rolex.agent.loop import Answer, Event, ToolFinished, ToolStarted
from rolex.core.config import Config
from rolex.core.errors import ToolError
from rolex.interface.handle import carry_out, recognised
from rolex.llm.base import Message, Reply, ToolSchema

CONVERSATION = "Conversation with a client:\nthem: when can I collect it?"


class FakeProvider:
    def __init__(self, answer: str = "Tomorrow evening.") -> None:
        self.answer = answer
        self.asked = 0

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        self.asked += 1
        return Reply(content=self.answer)

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        yield self.complete(messages, tools).content


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    placed: list[str] = []

    def remember(text: str) -> str:
        placed.append(text)
        return "ok"

    monkeypatch.setattr("rolex.agent.reply.read_chat", lambda: CONVERSATION)
    monkeypatch.setattr("rolex.agent.reply.open_conversation", lambda: "a client")
    monkeypatch.setattr("rolex.agent.reply.put_in_message_box", remember)
    monkeypatch.setattr("rolex.agent.reply.message_box_text", lambda: "")
    return placed


def run(said: str, provider: FakeProvider, config: Config | None = None) -> list[Event]:
    match = recognised(said)
    assert match is not None, f"{said!r} should have been recognised"
    return list(carry_out(match, said, provider, config or Config()))


def test_asking_for_a_reply_drafts_one(page: list[str]) -> None:
    events = run("reply to this", FakeProvider())

    assert page == ["Tomorrow evening."]
    assert isinstance(events[-1], Answer)
    assert "Tomorrow evening." in events[-1].content


def test_the_work_is_shown_the_same_way_the_loop_shows_it(page: list[str]) -> None:
    """The user should not have to care which path handled their sentence."""
    events = run("draft a reply", FakeProvider())

    assert any(isinstance(e, ToolStarted) for e in events)
    assert any(isinstance(e, ToolFinished) for e in events)
    assert isinstance(events[-1], Answer)


def test_the_answer_says_it_was_not_sent(page: list[str]) -> None:
    events = run("reply to this", FakeProvider())
    last = events[-1]
    assert isinstance(last, Answer)
    assert "not sent" in last.content.casefold()


def test_the_model_is_asked_once(page: list[str]) -> None:
    """No tool-choosing round trip: that is the point of routing it."""
    provider = FakeProvider()
    run("reply to this", provider)
    assert provider.asked == 1


def test_a_failure_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REPL has to survive it, the same as any other turn."""

    def no_chat() -> str:
        raise ToolError("no conversation is open. Open one first.")

    monkeypatch.setattr("rolex.agent.reply.read_chat", no_chat)
    monkeypatch.setattr("rolex.agent.reply.open_conversation", lambda: "a client")
    monkeypatch.setattr("rolex.agent.reply.message_box_text", lambda: "")

    events = run("reply to this", FakeProvider())

    assert any(isinstance(e, ToolFinished) and e.failed for e in events)
    last = events[-1]
    assert isinstance(last, Answer)
    assert "no conversation is open" in last.content


def test_an_unknown_command_says_so_rather_than_pretending(page: list[str]) -> None:
    events = list(carry_out(Match(name="dance"), "dance", FakeProvider(), Config()))
    last = events[-1]
    assert isinstance(last, Answer)
    assert "does not know how" in last.content


def test_ordinary_sentences_are_not_recognised() -> None:
    """Everything unmatched falls through to the agent loop as before."""
    for said in ("open chrome", "what time is it", "tell me a joke"):
        assert recognised(said) is None


def test_the_configured_language_is_used(page: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def spy(provider: object, config: object, **kwargs: object) -> object:
        asked.append(str(kwargs.get("language")))
        raise ToolError("stop here")

    monkeypatch.setattr("rolex.interface.handle.draft_reply", spy)

    run("reply to this", FakeProvider(), Config(reply_language="marathi"))

    assert asked == ["marathi"]


# --- a different model for writing ----------------------------------------


def test_one_model_by_default(page: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is built when the same model does both."""

    def refuse(config: object) -> object:
        raise AssertionError("a second provider should not be built")

    monkeypatch.setattr("rolex.interface.handle.build_provider", refuse)

    run("reply to this", FakeProvider(), Config(model="qwen2.5:3b"))
    assert page


def test_a_separate_writing_model_is_used(page: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A small model is right for chat and wrong for writing to a client:
    measured, qwen2.5:3b invented a price where qwen2.5:7b asked for it."""
    built: list[str] = []

    def build(config: Config) -> FakeProvider:
        built.append(config.model)
        return FakeProvider("Tomorrow evening.")

    monkeypatch.setattr("rolex.interface.handle.build_provider", build)

    run("reply to this", FakeProvider(), Config(model="qwen2.5:3b", reply_model="qwen2.5:7b"))

    assert built == ["qwen2.5:7b"]


def test_the_same_name_means_the_same_model(
    page: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(config: object) -> object:
        raise AssertionError("a second provider should not be built")

    monkeypatch.setattr("rolex.interface.handle.build_provider", refuse)

    run("reply to this", FakeProvider(), Config(model="m", reply_model="m"))
    assert page


def test_a_draft_that_was_not_placed_is_still_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing to overwrite should not mean losing the draft."""
    monkeypatch.setattr("rolex.agent.reply.read_chat", lambda: CONVERSATION)
    monkeypatch.setattr("rolex.agent.reply.open_conversation", lambda: "a client")
    monkeypatch.setattr("rolex.agent.reply.message_box_text", lambda: "half written")
    monkeypatch.setattr("rolex.agent.reply.put_in_message_box", lambda text: "ok")

    match = recognised("reply to this")
    assert match is not None
    events = list(
        carry_out(match, "reply to this", FakeProvider(), Config(), lambda old, new: False)
    )

    last = events[-1]
    assert isinstance(last, Answer)
    assert "Tomorrow evening." in last.content
    assert "already something there" in last.content


def test_a_bad_reply_model_does_not_end_the_session(
    page: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_provider raises ConfigError, which run_chat does not catch."""
    from rolex.core.errors import ConfigError

    def refuse(config: Config) -> object:
        raise ConfigError("the gemini provider arrives later")

    monkeypatch.setattr("rolex.interface.handle.build_provider", refuse)

    events = run("reply to this", FakeProvider(), Config(model="a", reply_model="b"))

    last = events[-1]
    assert isinstance(last, Answer)
    assert "gemini provider" in last.content


def test_the_exchange_is_remembered(page: list[str]) -> None:
    """A routed command skips the agent loop, so without this a follow-up
    like "make it shorter" has nothing to work from."""
    from rolex.llm.base import Message

    history: list[Message] = [Message("system", "be rolex")]
    match = recognised("reply to this")
    assert match is not None

    list(carry_out(match, "reply to this", FakeProvider(), Config(), None, history))

    assert [m.role for m in history] == ["system", "user", "assistant"]
    assert history[1].content == "reply to this"
    assert "Tomorrow evening." in history[2].content


def test_no_history_is_fine(page: list[str]) -> None:
    match = recognised("reply to this")
    assert match is not None
    assert list(carry_out(match, "reply to this", FakeProvider(), Config()))
