"""The listen loop, with hearing and speaking faked.

No microphone, no model, no sound. The loop's decisions are what is tested:
what it does with what it heard, and what it does when it heard nothing.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from rolex.core.config import Config
from rolex.core.errors import ProviderError
from rolex.interface.listen import Ears, run_listen
from rolex.llm.base import Message, Reply, ToolSchema
from rolex.tools.base import Tool
from rolex.voice.record import AudioError


class FakeEars:
    """Returns a scripted sequence of things "heard"."""

    def __init__(self, utterances: Sequence[str | Exception]) -> None:
        self.utterances = list(utterances)
        self.windows: list[float | None] = []

    def listen(self, *, seconds: float | None = None) -> str:
        self.windows.append(seconds)
        if not self.utterances:
            raise KeyboardInterrupt
        heard = self.utterances.pop(0)
        if isinstance(heard, Exception):
            raise heard
        return heard


class FakeSpeaker:
    """Records what would have been said aloud."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.said: list[str] = []

    def say(self, text: str) -> bool:
        if not self.available or not text.strip():
            return False
        self.said.append(text)
        return True


class FakeProvider:
    """Answers from a script."""

    def __init__(self, answers: Sequence[str | Reply | Exception]) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        self.asked.append(messages[-1].content)
        if not self.answers:
            raise AssertionError("the loop asked for more answers than the script has")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, Reply) else Reply(content=answer)

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        yield self.complete(messages, tools).content


def listen(
    heard: Sequence[str | Exception],
    answers: Sequence[str | Reply | Exception],
    *,
    push_to_talk: bool = False,
    tools: Sequence[Tool] = (),
    speaker: FakeSpeaker | None = None,
    config: Config | None = None,
) -> tuple[int, FakeProvider, FakeEars, FakeSpeaker]:
    """Run the loop over a script and return what everyone saw."""
    ears = FakeEars(heard)
    voice = FakeSpeaker() if speaker is None else speaker
    provider = FakeProvider(answers)

    code = run_listen(
        config or Config(),
        provider=provider,
        tools=tools,
        ears=ears,  # type: ignore[arg-type]
        speaker=voice,  # type: ignore[arg-type]
        push_to_talk=push_to_talk,
    )
    return code, provider, ears, voice


# --- hearing a command ----------------------------------------------------


def test_a_woken_command_reaches_the_model() -> None:
    _, provider, _, _ = listen(["rolex what time is it"], ["it is four o'clock"])
    assert provider.asked == ["what time is it"]


def test_the_answer_is_read_aloud() -> None:
    _, _, _, voice = listen(["rolex hello"], ["hello to you"])
    assert voice.said == ["hello to you"]


def test_the_answer_is_shown_even_when_it_cannot_be_spoken(
    capsys: pytest.CaptureFixture[str],
) -> None:
    listen(["rolex hello"], ["hello to you"], speaker=FakeSpeaker(available=False))
    assert "hello to you" in capsys.readouterr().out


def test_what_was_heard_is_shown_so_a_mishearing_is_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    listen(["rolex open chrome"], ["opening"])
    assert "open chrome" in capsys.readouterr().out


# --- not being spoken to --------------------------------------------------


def test_speech_without_the_wake_word_is_ignored() -> None:
    _, provider, _, _ = listen(["someone talking in the room", "rolex hello"], ["hi"])
    assert provider.asked == ["hello"]


def test_silence_is_ignored() -> None:
    _, provider, _, _ = listen(["", "   ", "rolex hello"], ["hi"])
    assert provider.asked == ["hello"]


def test_the_wake_word_alone_waits_for_the_command() -> None:
    """The user has rolex's attention and has not asked anything yet."""
    _, provider, ears, _ = listen(["rolex", "what time is it"], ["four o'clock"])
    assert provider.asked == ["what time is it"]
    assert ears.windows[1] is None, "the second listen should use the full window"


def test_the_wake_window_isshort() -> None:
    """Long clips make rolex slow to react; short ones catch the name."""
    _, _, ears, _ = listen(["rolex hello"], ["hi"])
    assert ears.windows[0] == pytest.approx(3.0)


# --- leaving ---------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["stop listening", "Stop listening.", "goodbye", "exit"])
def test_it_can_be_told_to_stop(phrase: str) -> None:
    code, provider, _, _ = listen([f"rolex {phrase}"], [])
    assert code == 0
    assert provider.asked == []


def test_ctrl_c_leaves_cleanly() -> None:
    code, _, _, _ = listen([], [])
    assert code == 0


# --- when something is wrong ---------------------------------------------


def test_a_microphone_failure_stops_with_an_explanation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retrying a blocked microphone forever would spin without telling anyone."""
    code, _, _, _ = listen([AudioError("the microphone could not be opened")], [])
    assert code == 1
    assert "microphone" in capsys.readouterr().err


def test_a_model_failure_does_not_end_the_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, provider, _, _ = listen(
        ["rolex first", "rolex second"],
        [ProviderError("server died"), "recovered"],
    )
    assert provider.asked == ["first", "second"]
    assert "server died" in capsys.readouterr().err


def test_an_unavailable_model_stops_before_listening(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Unavailable(FakeProvider):
        def is_available(self) -> bool:
            return False

    ears = FakeEars(["rolex hello"])
    code = run_listen(
        Config(),
        provider=Unavailable([]),
        tools=[],
        ears=ears,  # type: ignore[arg-type]
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
    )

    assert code == 1
    assert ears.windows == [], "it should not listen when it cannot answer"


# --- push to talk ---------------------------------------------------------


def test_push_to_talk_needs_no_wake_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _="": "")
    _, provider, _, _ = listen(["what time is it"], ["four o'clock"], push_to_talk=True)
    assert provider.asked == ["what time is it"]


def test_push_to_talk_always_uses_the_full_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no wake word to catch, so there is no short clip."""
    monkeypatch.setattr("builtins.input", lambda _="": "")
    _, _, ears, _ = listen(["hello"], ["hi"], push_to_talk=True)
    assert ears.windows
    assert all(window is None for window in ears.windows)


# --- recordings are not kept ----------------------------------------------


def test_a_recording_is_deleted_even_if_transcribing_fails(tmp_path: Any) -> None:
    """A recording of someone's room must not be left in /tmp."""
    recording = tmp_path / "clip.wav"
    recording.write_text("audio")

    class Recorder:
        def record(self, *, seconds: float | None = None) -> Any:
            return recording

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            raise RuntimeError("whisper fell over")

    ears = Ears(Recorder(), Transcriber())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        ears.listen()

    assert not recording.exists()


# --- the long haul --------------------------------------------------------


def test_history_is_trimmed_across_a_long_session() -> None:
    """A long voice session would otherwise overflow the context window and
    Ollama would drop the system prompt from the front."""
    turns = 40
    code, provider, _, _ = listen(
        [f"rolex q{i}" for i in range(turns)],
        [f"a{i}" for i in range(turns)],
    )

    assert code == 0
    assert provider.asked == [f"q{i}" for i in range(turns)]


def test_ctrl_d_in_push_to_talk_leaves_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rolex talk < /dev/null` used to end in a traceback."""

    def eof(_: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    code, _, _, _ = listen(["never reached"], [], push_to_talk=True)
    assert code == 0


def test_a_long_tool_result_is_shortened_on_screen(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rolex.llm.base import ToolCall

    chatty = Tool(name="chatty", description="d", run=lambda _: "x" * 500)
    wants = Reply(content="", tool_calls=(ToolCall("chatty", {}),))

    listen(["rolex go"], [wants, "done"], tools=[chatty])

    out = capsys.readouterr().out
    assert "…" in out
    assert "x" * 200 not in out


def test_a_multi_line_tool_result_is_shown_whole(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """machine_info returns two lines; only the first used to be seen."""
    from rolex.llm.base import ToolCall

    two_lines = Tool(name="info", description="d", run=lambda _: "Mac: arm64\nMemory: 8 GB")
    wants = Reply(content="", tool_calls=(ToolCall("info", {}),))

    listen(["rolex go"], [wants, "done"], tools=[two_lines])

    out = capsys.readouterr().out
    assert "Mac: arm64" in out
    assert "Memory: 8 GB" in out
