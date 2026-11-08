"""The listen loop, with hearing and speaking faked.

No microphone, no model, no sound. The loop's decisions are what is tested:
what it does with what it heard, and what it does when it heard nothing.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from rolex.core.config import Config
from rolex.core.errors import ProviderError
from rolex.interface.listen import (
    FOLLOW_UP_SECONDS,
    Ears,
    SpokenConsent,
    _hearing,
    _question,
    run_listen,
)
from rolex.llm.base import Message, Reply, ToolSchema
from rolex.tools.base import Tool
from rolex.voice.record import AudioError, Recording


class FakeEars:
    """Returns a scripted sequence of things "heard"."""

    def __init__(self, utterances: Sequence[str | Exception]) -> None:
        self.utterances = list(utterances)
        self.windows: list[float | None] = []
        self.discarded = 0

    def listen(self, *, seconds: float | None = None) -> str:
        self.windows.append(seconds)
        if not self.utterances:
            raise KeyboardInterrupt
        heard = self.utterances.pop(0)
        if isinstance(heard, Exception):
            raise heard
        return heard

    def next_clip(self, *, seconds: float) -> str:
        return self.listen(seconds=seconds)

    def discard(self) -> None:
        self.discarded += 1


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
        ears=ears,
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
    _, provider, _, _ = listen(["rolex", "what time is it"], ["four o'clock"])
    assert provider.asked == ["what time is it"]


def test_the_wait_after_a_bare_wake_word_is_bounded() -> None:
    """Waiting for ever would take a sentence spoken minutes later, to
    someone else, as the command."""
    _, _, ears, _ = listen(["rolex", "what time is it"], ["four o'clock"])
    assert ears.windows[:2] == [pytest.approx(3.0), pytest.approx(FOLLOW_UP_SECONDS)]


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
        ears=ears,
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
        def record(self, *, seconds: float | None = None) -> Recording:
            return Recording(path=recording, ended_on_silence=True)

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


# --- permission, asked out loud -------------------------------------------


class AskEars:
    """Ears that answer the permission question from a script."""

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.windows: list[float | None] = []

    def listen(self, *, seconds: float | None = None) -> str:
        self.windows.append(seconds)
        return self.answers.pop(0) if self.answers else ""

    def next_clip(self, *, seconds: float) -> str:
        return self.listen(seconds=seconds)

    def discard(self) -> None:
        return None


def consent(answers: Sequence[str]) -> tuple[bool, AskEars, FakeSpeaker]:
    ears = AskEars(answers)
    voice = FakeSpeaker()
    tool = Tool(name="open_app", description="d", run=lambda _: "", safety="mutating")
    allowed = SpokenConsent(ears=ears, voice=voice).ask(tool, {"name": "Google Chrome"})  # type: ignore[arg-type]
    return allowed, ears, voice


def test_the_question_is_asked_out_loud() -> None:
    """The user is speaking to rolex. A typed [y/N] goes unanswered."""
    _, _, voice = consent(["yes"])
    assert voice.said[0] == "Open Google Chrome?"


def test_a_spoken_yes_allows_it() -> None:
    allowed, _, _ = consent(["yes please"])
    assert allowed


def test_a_spoken_no_refuses_it() -> None:
    allowed, _, _ = consent(["no"])
    assert not allowed


def test_an_unclear_answer_is_asked_again() -> None:
    allowed, ears, voice = consent(["hmm", "yes"])
    assert allowed
    assert len(ears.windows) == 2
    assert any("yes or a no" in said for said in voice.said)


def test_silence_is_taken_as_no_after_asking_twice() -> None:
    allowed, ears, voice = consent(["", ""])
    assert not allowed
    assert len(ears.windows) == 2
    assert any("not done it" in said for said in voice.said)


def test_a_microphone_failure_while_asking_is_a_no() -> None:
    class Broken:
        def listen(self, *, seconds: float | None = None) -> str:
            raise AudioError("microphone gone")

    tool = Tool(name="open_app", description="d", run=lambda _: "", safety="mutating")
    assert not SpokenConsent(ears=Broken(), voice=FakeSpeaker()).ask(tool, {})  # type: ignore[arg-type]


def test_the_consent_window_is_short() -> None:
    """A yes or no is a word, not a speech."""
    _, ears, _ = consent(["yes"])
    assert ears.windows == [pytest.approx(4.0)]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        ("open_app", {"name": "Safari"}, "Open Safari?"),
        ("open_url", {"url": "https://example.com"}, "Open https://example.com in the browser?"),
        ("search_web", {"query": "pune weather"}, "Search the web for pune weather?"),
        ("some_tool", {}, "Run some tool?"),
        ("some_tool", {"x": "1"}, "Run some tool with 1?"),
    ],
)
def test_the_question_is_phrased_for_listening(
    tool_name: str, arguments: dict[str, str], expected: str
) -> None:
    tool = Tool(name=tool_name, description="d", run=lambda _: "", safety="mutating")
    assert _question(tool, arguments) == expected


def test_a_mutating_tool_in_a_voice_session_is_asked_about_aloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole path: heard a command, asked out loud, acted on the answer."""
    from rolex.llm.base import ToolCall

    monkeypatch.setattr("rolex.interface.processor.installed_apps", lambda: ["Google Chrome"])
    opened: list[str] = []
    app = Tool(
        name="open_app",
        description="d",
        run=lambda args: opened.append(str(args.get("name"))) or "opened",  # type: ignore[func-returns-value]
        safety="mutating",
    )
    wants = Reply(content="", tool_calls=(ToolCall("open_app", {"name": "Google Chrome"}),))

    _, _, _, voice = listen(
        ["rolex open chrome", "yes"],
        [wants, "Chrome is open"],
        tools=[app],
    )

    assert opened == ["Google Chrome"]
    assert any("Open Google Chrome?" in said for said in voice.said)


# --- the microphone is never left idle -------------------------------------


def test_the_next_clip_starts_before_the_last_is_transcribed(tmp_path: Any) -> None:
    """Otherwise the microphone is shut for however long whisper takes, and a
    wake word spoken in that gap is never heard."""

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            return "rolex hello"

    class CountingRecorder:
        def __init__(self) -> None:
            self.made = 0

        def record(self, *, seconds: float | None = None) -> Recording:
            self.made += 1
            clip = tmp_path / f"clip{self.made}.wav"
            clip.write_text("audio")
            return Recording(path=clip, ended_on_silence=True)

    recorder = CountingRecorder()
    ears = Ears(recorder, Transcriber())  # type: ignore[arg-type]

    assert ears.next_clip(seconds=3.0) == "rolex hello"
    assert ears.pending is not None, "the next clip should already be recording"

    ears.discard()

    # Two recordings for one transcript: the second was started before the
    # first was transcribed, so the microphone was never shut.
    assert recorder.made == 2


def test_a_transcribed_clip_is_deleted(tmp_path: Any) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_text("audio")

    class Recorder:
        def record(self, *, seconds: float | None = None) -> Recording:
            return Recording(path=clip, ended_on_silence=True)

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            return "hello"

    assert Ears(Recorder(), Transcriber()).listen() == "hello"  # type: ignore[arg-type]
    assert not clip.exists()


def test_a_discarded_clip_is_deleted(tmp_path: Any) -> None:
    """It holds rolex's own answer coming out of the speaker, not a command."""
    made: list[Any] = []

    class Recorder:
        def record(self, *, seconds: float | None = None) -> Recording:
            clip = tmp_path / f"clip{len(made)}.wav"
            clip.write_text("audio")
            made.append(clip)
            return Recording(path=clip, ended_on_silence=True)

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            return "rolex hello"

    ears = Ears(Recorder(), Transcriber())  # type: ignore[arg-type]
    ears.next_clip(seconds=3.0)
    ears.discard()

    assert ears.pending is None
    assert not any(clip.exists() for clip in made)


def test_discarding_nothing_is_harmless() -> None:
    ears = Ears(None, None)  # type: ignore[arg-type]
    ears.discard()
    assert ears.pending is None


def test_what_was_recorded_while_answering_is_thrown_away() -> None:
    """Twice: once after the answer, once on the way out."""
    _, _, ears, _ = listen(["rolex hello"], ["hi there"])
    assert ears.discarded == 2


def test_push_to_talk_discards_nothing_between_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing records between turns, so only the exit sweep runs."""
    monkeypatch.setattr("builtins.input", lambda _="": "")
    _, _, ears, _ = listen(["hello"], ["hi"], push_to_talk=True)
    assert ears.discarded == 1


@pytest.mark.parametrize(
    ("heard", "expected_code"),
    [
        ([], 0),
        (["rolex stop listening"], 0),
        ([AudioError("microphone gone")], 1),
    ],
)
def test_no_recording_is_left_running_on_the_way_out(
    heard: list[str | Exception], expected_code: int
) -> None:
    """However the session ends, the clip in flight must not survive it."""
    code, _, ears, _ = listen(heard, [])
    assert code == expected_code
    assert ears.discarded >= 1


def test_a_failed_recording_does_not_poison_every_later_one(tmp_path: Any) -> None:
    """Left in `pending`, a failed future re-raises instantly for ever after,
    so the microphone is never tried again."""
    attempts: list[int] = []

    class FlakyRecorder:
        def record(self, *, seconds: float | None = None) -> Recording:
            attempts.append(1)
            if len(attempts) == 1:
                raise AudioError("microphone busy")
            clip = tmp_path / f"clip{len(attempts)}.wav"
            clip.write_text("audio")
            return Recording(path=clip, ended_on_silence=True)

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            return "rolex hello"

    ears = Ears(FlakyRecorder(), Transcriber())  # type: ignore[arg-type]

    with pytest.raises(AudioError):
        ears.next_clip(seconds=3.0)

    assert ears.pending is None
    assert ears.next_clip(seconds=3.0) == "rolex hello"
    ears.discard()


def test_listening_drops_a_clip_already_in_flight(tmp_path: Any) -> None:
    """Two ffmpegs on one microphone split the utterance between them."""
    live: list[str] = []

    class OneAtATimeRecorder:
        def __init__(self) -> None:
            self.made = 0

        def record(self, *, seconds: float | None = None) -> Recording:
            if live:
                raise AssertionError("two recordings at once")
            live.append("recording")
            try:
                self.made += 1
                clip = tmp_path / f"clip{self.made}.wav"
                clip.write_text("audio")
                return Recording(path=clip, ended_on_silence=True)
            finally:
                live.pop()

    class Transcriber:
        def transcribe(self, path: Any) -> str:
            return "yes"

    ears = Ears(OneAtATimeRecorder(), Transcriber())  # type: ignore[arg-type]
    ears.next_clip(seconds=3.0)

    assert ears.pending is not None
    assert ears.listen(seconds=4.0) == "yes"
    assert ears.pending is None


def test_asking_permission_does_not_reuse_the_clip_of_its_own_question() -> None:
    """What was recording while rolex spoke holds rolex's voice, not a yes."""
    allowed, ears, _ = consent(["yes"])
    assert allowed
    assert ears.windows == [pytest.approx(4.0)]


# --- choosing how to hear --------------------------------------------------


def test_push_to_talk_records_clips_rather_than_streaming() -> None:
    """Nothing to stream: it records only when asked."""
    hearing = _hearing(Config(), None, push_to_talk=True)
    assert isinstance(hearing, Ears)


def test_streaming_can_be_turned_off() -> None:
    hearing = _hearing(Config(stream=False), None, push_to_talk=False)
    assert isinstance(hearing, Ears)


def test_a_missing_whisper_stream_falls_back_to_clips(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Clips lose words across boundaries, but they work. Refusing to start
    would be worse for someone who has whisper-cli and not whisper-stream."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    hearing = _hearing(Config(stream=True), None, push_to_talk=False)

    assert isinstance(hearing, Ears)
    assert "listening in clips instead" in capsys.readouterr().err
