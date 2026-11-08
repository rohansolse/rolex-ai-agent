"""Carrying out a recognised command, and when not to."""

from collections.abc import Iterator

import pytest

from rolex.agent.commands import Match
from rolex.agent.loop import Answer, Event, ToolFinished, ToolRefused, ToolStarted
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


# --- opening a chat --------------------------------------------------------


def test_opening_a_chat_searches_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def opener(name: str) -> str:
        asked.append(name)
        return f"Opened the WhatsApp chat with {name}."

    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", opener)

    events = run("open the testing ai group", FakeProvider())

    assert asked == ["testing ai"]
    last = events[-1]
    assert isinstance(last, Answer)
    assert "testing ai" in last.content


def test_the_model_is_not_asked_how_to_open_a_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", lambda name: "Opened.")
    provider = FakeProvider()

    run("open testing ai group", provider)

    assert provider.asked == 0


def test_a_chat_that_cannot_be_opened_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(name: str) -> str:
        raise ToolError("no WhatsApp chat called 'ghost' came up in the search")

    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", refuse)

    events = run("open the ghost group", FakeProvider())

    assert any(isinstance(e, ToolFinished) and e.failed for e in events)
    last = events[-1]
    assert isinstance(last, Answer)
    assert "came up in the search" in last.content


# --- sending ---------------------------------------------------------------


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Fake the WhatsApp page; returns what happened to it."""
    state: dict[str, object] = {"box": "", "open": "testing ai", "sent": [], "opened": []}

    def put(text: str) -> str:
        state["box"] = text
        return f"Put this in the message box, unsent, for you to check:\n{text}"

    def send() -> str:
        sent = state["sent"]
        assert isinstance(sent, list)
        sent.append((state["open"], state["box"]))
        state["box"] = ""
        return f"Sent to {state['open']}: {sent[-1][1]}"

    def open_chat(name: str) -> str:
        opened = state["opened"]
        assert isinstance(opened, list)
        opened.append(name)
        state["open"] = name
        return f"Opened the WhatsApp chat with {name}."

    monkeypatch.setattr("rolex.interface.handle.put_in_message_box", put)
    monkeypatch.setattr("rolex.interface.handle.send_message", send)
    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", open_chat)
    monkeypatch.setattr("rolex.interface.handle.message_box_text", lambda: state["box"])
    monkeypatch.setattr("rolex.interface.handle.open_conversation", lambda: state["open"])
    return state


def send(said: str, chat: dict[str, object], approve: object = True) -> list[Event]:
    match = recognised(said)
    assert match is not None, f"{said!r} should have been recognised"
    asker = None if approve is None else (lambda about, text: bool(approve))
    return list(carry_out(match, said, FakeProvider(), Config(), asker))


def test_a_message_is_typed_and_sent(chat: dict[str, object]) -> None:
    events = send("send hi to rohan solse", chat)

    assert chat["opened"] == ["rohan solse"]
    assert chat["sent"] == [("rohan solse", "hi")]
    last = events[-1]
    assert isinstance(last, Answer)
    assert "Sent to rohan solse" in last.content


def test_nothing_is_sent_without_a_yes(chat: dict[str, object]) -> None:
    """A message cannot be taken back: a moment later it is on someone
    else's phone."""
    events = send("send hi to rohan solse", chat, approve=False)

    assert chat["sent"] == []
    assert chat["box"] == "hi", "it stays in the box for the user to deal with"
    last = events[-1]
    assert isinstance(last, Answer)
    assert "Not sent" in last.content


def test_with_nobody_to_ask_nothing_is_sent(chat: dict[str, object]) -> None:
    """A piped or scheduled run has no one at the keyboard, and silence is
    not permission to message a client."""
    send("send hi to rohan solse", chat, approve=None)
    assert chat["sent"] == []


def test_the_question_names_the_recipient_and_the_words(
    chat: dict[str, object],
) -> None:
    asked: list[tuple[str, str]] = []

    def record(about: str, text: str) -> bool:
        asked.append((about, text))
        return False

    match = recognised("send hi to rohan solse")
    assert match is not None
    list(carry_out(match, "send hi to rohan solse", FakeProvider(), Config(), record))

    assert asked == [("send:rohan solse", "hi")]


def test_send_it_sends_what_is_already_there(chat: dict[str, object]) -> None:
    chat["box"] = "already typed"
    send("send it", chat)
    assert chat["sent"] == [("testing ai", "already typed")]
    assert chat["opened"] == [], "no chat is opened; the one in front is used"


def test_send_it_with_an_empty_box_says_so(chat: dict[str, object]) -> None:
    events = send("send it", chat)
    assert chat["sent"] == []
    last = events[-1]
    assert isinstance(last, Answer)
    assert "nothing in the message box" in last.content


def test_a_chat_that_will_not_open_stops_before_typing(
    chat: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str) -> str:
        raise ToolError("no WhatsApp chat called 'ghost' came up in the search")

    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", refuse)

    events = send("send hi to ghost", chat)

    assert chat["box"] == "", "nothing typed into whatever was open"
    assert chat["sent"] == []
    last = events[-1]
    assert isinstance(last, Answer)
    assert "came up in the search" in last.content


def test_sending_does_not_type_over_a_half_written_message(
    chat: dict[str, object],
) -> None:
    """Filling the box selects everything and types over it."""
    chat["box"] = "I was writing thi"
    events = send("send hi to rohan solse", chat, approve=False)

    assert chat["box"] == "I was writing thi"
    assert chat["sent"] == []
    last = events[-1]
    assert isinstance(last, Answer)
    assert "already has" in last.content


def test_an_approved_replacement_goes_ahead(chat: dict[str, object]) -> None:
    chat["box"] = "half written"
    send("send hi to rohan solse", chat, approve=True)
    assert chat["sent"] == [("rohan solse", "hi")]


def test_every_started_step_is_finished(
    chat: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpaired ToolStarted leaves the user looking at a step that never
    ended."""

    def refuse(name: str) -> str:
        raise ToolError("no such chat")

    monkeypatch.setattr("rolex.interface.handle.open_whatsapp_chat", refuse)

    events = send("send hi to ghost", chat)

    started = [e.name for e in events if isinstance(e, ToolStarted)]
    ended = [e.name for e in events if isinstance(e, (ToolFinished, ToolRefused))]
    for name in started:
        assert name in ended, f"{name} was started and never finished"
