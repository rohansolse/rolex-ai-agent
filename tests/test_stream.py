"""Reading whisper-stream's output.

The fixtures below are its real output, captured from a run. The parsing is
the part that goes wrong, and this way it can be checked without the
microphone or the model.
"""

import pytest

from rolex.core.config import Config
from rolex.core.errors import StreamError
from rolex.interface.listen import StreamEars
from rolex.voice.stream import BUFFER_MS, Stream, Transcript

FIRST = """\
### Transcription 0 START | t0 = 0 ms | t1 = 8627 ms

[00:00:00.000 --> 00:00:08.080]   Hello, Rolex, what is the time today?

### Transcription 0 END
"""

SECOND = """\
### Transcription 1 START | t0 = 0 ms | t1 = 14207 ms

[00:00:00.000 --> 00:00:11.840]   Hello, Rolex, what is the time today?
[00:00:11.840 --> 00:00:13.520]   Rolex Open Google Chrome

### Transcription 1 END
"""


def feed(transcript: Transcript, block: str) -> list[str]:
    """Everything new across a whole block of output."""
    said: list[str] = []
    for line in block.splitlines():
        said.extend(transcript.feed(line))
    return said


def test_an_utterance_is_reported() -> None:
    assert feed(Transcript(), FIRST) == ["Hello, Rolex, what is the time today?"]


def test_a_repeated_utterance_is_not_reported_twice() -> None:
    """Every block re-transcribes the rolling buffer, so each one contains
    all the ones before it. Reported again, rolex would answer twice."""
    transcript = Transcript()

    assert feed(transcript, FIRST) == ["Hello, Rolex, what is the time today?"]
    assert feed(transcript, SECOND) == ["Rolex Open Google Chrome"]


def test_the_buffer_emptying_is_not_mistaken_for_a_repeat() -> None:
    """When the buffer is cleared the timestamps start again from zero, and
    nothing would ever be reported again if that were read as old audio."""
    transcript = Transcript()
    feed(transcript, SECOND)

    restarted = (
        "### Transcription 2 START | t0 = 0 ms | t1 = 1800 ms\n"
        "[00:00:00.000 --> 00:00:01.800]   What time is it\n"
    )

    assert feed(transcript, restarted) == ["What time is it"]


def test_blank_segments_are_ignored() -> None:
    transcript = Transcript()
    block = (
        "### Transcription 0 START | t0 = 0 ms | t1 = 2000 ms\n[00:00:00.000 --> 00:00:02.000]   \n"
    )
    assert feed(transcript, block) == []


@pytest.mark.parametrize(
    "line",
    [
        "",
        "[Start speaking]",
        "main: using VAD, will transcribe on speech activity",
        "### Transcription 0 END",
        "whisper_init_state: compute buffer (decode) = 97.29 MB",
    ],
)
def test_everything_that_is_not_speech_is_ignored(line: str) -> None:
    assert Transcript().feed(line) == []


def test_a_merged_re_transcription_yields_only_the_new_part() -> None:
    """whisper sometimes reports a new sentence joined onto one already
    heard. Reported whole, rolex answers the old question again; dropped
    whole, the new command is lost."""
    transcript = Transcript()
    transcript.feed("### Transcription 0 START | t0 = 0 ms | t1 = 2000 ms")
    assert transcript.feed("[00:00:00.000 --> 00:00:02.000]   What time is it") == [
        "What time is it"
    ]

    transcript.feed("### Transcription 1 START | t0 = 0 ms | t1 = 5000 ms")
    merged = "[00:00:00.000 --> 00:00:05.000]   What time is it Rolex open chrome"
    assert transcript.feed(merged) == ["Rolex open chrome"]


def test_the_same_sentence_with_a_later_end_time_is_still_a_repeat() -> None:
    """A re-transcription has had more audio, so it reports the same words
    ending later. A timestamp alone cannot tell that from new speech."""
    transcript = Transcript()
    transcript.feed("### Transcription 0 START | t0 = 0 ms | t1 = 2000 ms")
    transcript.feed("[00:00:00.000 --> 00:00:02.000]   What time is it")

    transcript.feed("### Transcription 1 START | t0 = 0 ms | t1 = 9000 ms")
    assert transcript.feed("[00:00:00.000 --> 00:00:08.000]   What time is it") == []


# --- the command ----------------------------------------------------------


def test_the_command_uses_voice_activity_not_a_clock() -> None:
    """step 0 is whisper.cpp's VAD mode: sentences end when the speaker
    stops. A fixed step is what cut "today" off the end of a question."""
    argv = Stream(Config()).argv("/whisper-stream")
    assert argv[argv.index("--step") + 1] == "0"


def test_the_buffer_is_bounded() -> None:
    argv = Stream(Config()).argv("/whisper-stream")
    assert argv[argv.index("--length") + 1] == str(BUFFER_MS)


def test_the_model_is_passed() -> None:
    config = Config()
    argv = Stream(config).argv("/whisper-stream")
    assert argv[argv.index("--model") + 1] == str(config.whisper_model)


def test_no_capture_device_is_named_by_default() -> None:
    assert "--capture" not in Stream(Config()).argv("/whisper-stream")


def test_a_chosen_microphone_is_passed() -> None:
    argv = Stream(Config(stream_device="1")).argv("/whisper-stream")
    assert argv[argv.index("--capture") + 1] == "1"


def test_the_ffmpeg_device_is_not_reused_here() -> None:
    """ffmpeg's avfoundation index and SDL's capture index number different
    lists; using one for the other opens the wrong microphone."""
    assert "--capture" not in Stream(Config(audio_device="2")).argv("/whisper-stream")


def test_a_missing_whisper_stream_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(StreamError, match="brew install whisper-cpp"):
        Stream(Config()).start()


def test_a_missing_model_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-stream")
    stream = Stream(Config(whisper_model=Config().whisper_model.parent / "absent.bin"))
    with pytest.raises(StreamError, match="make whisper-model"):
        stream.start()


def test_the_microphone_stopping_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead whisper-stream must not look like a quiet room for ever."""
    stream = Stream(Config())
    stream.lines.put(None)
    with pytest.raises(StreamError, match="Privacy & Security"):
        stream.listen(timeout=0.1)


def test_nothing_said_is_not_an_error() -> None:
    assert Stream(Config()).listen(timeout=0.01) is None


def test_draining_leaves_the_queue_empty_and_nothing_reported_twice() -> None:
    stream = Stream(Config())
    for line in SECOND.splitlines():
        stream.lines.put(line)

    stream.drain()

    assert stream.lines.empty()
    assert "Rolex Open Google Chrome" in stream.transcript.said


# --- the loop's view of a stream ------------------------------------------


class FakeStream:
    """A Stream that yields scripted utterances and records being drained."""

    def __init__(self, said: list[str | Exception]) -> None:
        self.said = list(said)
        self.drained = 0
        self.stopped = False

    def listen(self, *, timeout: float | None = None) -> str | None:
        if not self.said:
            return None
        nxt = self.said.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt or None

    def drain(self) -> None:
        self.drained += 1

    def stop(self) -> None:
        self.stopped = True


def ears(said: list[str | Exception]) -> tuple[StreamEars, FakeStream]:
    stream = FakeStream(said)
    return StreamEars(stream), stream  # type: ignore[arg-type]


def test_an_utterance_is_passed_through() -> None:
    hearing, _ = ears(["open chrome"])
    assert hearing.listen(seconds=1.0) == "open chrome"


def test_silence_within_the_limit_gives_nothing() -> None:
    hearing, _ = ears([])
    assert hearing.listen(seconds=0.05) == ""


def test_listening_drops_what_was_heard_while_rolex_spoke() -> None:
    """The microphone stays open while rolex asks a question out loud, so
    its own voice is sitting in the queue when it starts listening."""
    hearing, stream = ears(["yes"])
    hearing.listen(seconds=1.0)
    assert stream.drained == 1


def test_waiting_for_the_wake_word_has_no_deadline() -> None:
    """Nothing has been said yet; there is nothing to give up on."""
    hearing, _ = ears(["", "", "rolex open chrome"])
    assert hearing.next_clip(seconds=3.0) == "rolex open chrome"


def test_a_dead_microphone_is_raised_rather_than_waited_on() -> None:
    hearing, _ = ears([StreamError("the microphone stopped")])
    with pytest.raises(StreamError):
        hearing.next_clip(seconds=3.0)


def test_closing_stops_the_stream() -> None:
    hearing, stream = ears([])
    hearing.close()
    assert stream.stopped
