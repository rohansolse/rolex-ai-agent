"""Recording, transcribing and speaking, with the subprocesses faked.

No microphone is opened and no sound is made, so the suite is silent and runs
anywhere. What is tested is the command we build and how we read what comes
back, which is where the mistakes are.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rolex.core.config import Config
from rolex.voice.record import AudioError, Listener, Recorder, explain
from rolex.voice.speak import Speaker
from rolex.voice.transcribe import Transcriber, TranscriptionError, clean


@dataclass
class FakeRun:
    """Stands in for subprocess.run, recording the argv it was handed."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    raises: Exception | None = None
    writes: str | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str], **_: Any) -> "FakeRun":
        self.calls.append(list(argv))
        if self.raises:
            raise self.raises
        if self.writes is not None:
            Path(argv[-1]).write_text(self.writes)
        return self

    @property
    def argv(self) -> list[str]:
        assert len(self.calls) == 1, f"expected one call, got {self.calls}"
        return self.calls[0]


# --- recording ------------------------------------------------------------


@dataclass
class FakePopen:
    """Stands in for ffmpeg: emits scripted log lines and notes being stopped."""

    lines: list[str] = field(default_factory=list)
    returncode: int = 0
    writes: str | None = "pretend this is audio"
    destination: Path | None = None
    stopped: bool = False

    argv: list[str] = field(default_factory=list)
    stdin: Any = None
    stdout: Any = None
    stderr: Any = None

    def __call__(self, argv: list[str], **_: Any) -> "FakePopen":
        self.argv = list(argv)
        self.destination = Path(argv[-1])
        if self.writes is not None:
            self.destination.write_text(self.writes)
        self.stdin = _StopRecorder(self)
        self.stderr = iter(self.lines)
        return self

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        self.stopped = True


@dataclass
class _StopRecorder:
    """ffmpeg's stdin: 'q' written to it means "stop now"."""

    process: "FakePopen"
    closed: bool = False

    def write(self, data: str) -> int:
        if "q" in data:
            self.process.stopped = True
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakePopen:
    fake = FakePopen()
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")
    monkeypatch.setattr("subprocess.Popen", fake)
    return fake


def test_a_recording_is_written_and_returned(ffmpeg: FakePopen) -> None:
    path = Recorder(Config()).record()
    try:
        assert path.exists()
        assert path.suffix == ".wav"
    finally:
        path.unlink(missing_ok=True)


def test_the_command_records_mono_at_the_rate_whisper_wants() -> None:
    argv = Recorder(Config()).argv("/ffmpeg", Path("/tmp/x.wav"), 8.0)
    assert "-ac" in argv and argv[argv.index("-ac") + 1] == "1"
    assert "-ar" in argv and argv[argv.index("-ar") + 1] == "16000"


def test_the_log_level_lets_silencedetect_through() -> None:
    """At 'error' the detector's output never appears and nothing can stop
    early — which is how the first version of this silently did nothing."""
    argv = Recorder(Config()).argv("/ffmpeg", Path("/tmp/x.wav"), 8.0)
    assert argv[argv.index("-loglevel") + 1] == "info"


def test_the_silence_filter_uses_the_configured_duration() -> None:
    argv = Recorder(Config(silence_seconds=0.8)).argv("/ffmpeg", Path("/tmp/x.wav"), 8.0)
    filters = argv[argv.index("-af") + 1]
    assert "silencedetect" in filters
    assert "d=0.8" in filters


def test_the_time_limit_is_passed() -> None:
    argv = Recorder(Config()).argv("/ffmpeg", Path("/tmp/x.wav"), 3.0)
    assert argv[argv.index("-t") + 1] == "3.0"


def test_the_default_microphone_is_used_when_none_is_named() -> None:
    argv = Recorder(Config()).argv("/ffmpeg", Path("/tmp/x.wav"), 3.0)
    assert argv[argv.index("-i") + 1] == ":default"


def test_a_named_microphone_is_used() -> None:
    argv = Recorder(Config(audio_device="1")).argv("/ffmpeg", Path("/tmp/x.wav"), 3.0)
    assert argv[argv.index("-i") + 1] == ":1"


# --- stopping when the speaker stops -------------------------------------


# These lines are ffmpeg's real output, copied from a run. The first version
# of this module was written against an imagined format and did nothing.

SPEECH_THEN_SILENCE = "[Parsed_silencedetect_0 @ 0x79710] silence_start: 1.475563\n"
SILENCE_AT_THE_START = "[Parsed_silencedetect_0 @ 0x79710] silence_start: 0\n"
SPEECH_BEGINS = "[Parsed_silencedetect_0 @ 0x79710] silence_end: 1.1 | silence_duration: 1.1\n"
PROGRESS = "size=      30KiB time=00:00:01.02 bitrate= 240.6kbits/s speed=2.01x\n"


def test_silence_after_speech_ends_the_recording() -> None:
    """A clip that starts with talking never produces a silence_end at all,
    so waiting for one means never stopping. That was the original bug."""
    assert Listener().should_stop(SPEECH_THEN_SILENCE)


def test_silence_before_anyone_speaks_does_not_end_it() -> None:
    """Otherwise the recording ends before the user has said a word."""
    assert not Listener().should_stop(SILENCE_AT_THE_START)


def test_silence_after_a_pause_then_speech_ends_the_recording() -> None:
    listener = Listener()
    assert not listener.should_stop(SILENCE_AT_THE_START)
    assert not listener.should_stop(SPEECH_BEGINS)
    assert listener.should_stop("silence_start: 3.2")


def test_several_events_on_one_line_are_all_read() -> None:
    """ffmpeg's progress output uses carriage returns, so a whole run's
    events can arrive in a single line."""
    listener = Listener()
    combined = PROGRESS + "silence_start: 0 " + PROGRESS + "silence_end: 1.1 | duration: 1.1"
    assert not listener.should_stop(combined)
    assert listener.spoke


def test_ordinary_ffmpeg_chatter_is_ignored() -> None:
    listener = Listener()
    listener.spoke = True
    assert not listener.should_stop(PROGRESS)
    assert not listener.should_stop("Stream #0:0: Audio: pcm_s16le, 16000 Hz, mono")


def test_ffmpeg_is_asked_to_stop_once_the_sentence_ends(ffmpeg: FakePopen) -> None:
    """A real stop, not just a log line read: this is what the first version
    of this module claimed to do and did not."""
    ffmpeg.lines = [PROGRESS, SPEECH_THEN_SILENCE]

    path = Recorder(Config()).record()
    try:
        assert ffmpeg.stopped
    finally:
        path.unlink(missing_ok=True)


def test_a_clean_exit_from_an_early_stop_is_not_an_error(ffmpeg: FakePopen) -> None:
    """ffmpeg returns 255 when told to quit, which is a finished sentence."""
    ffmpeg.returncode = 255
    path = Recorder(Config()).record()
    path.unlink(missing_ok=True)


# --- when recording fails -------------------------------------------------


def test_a_missing_ffmpeg_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(AudioError, match="brew install ffmpeg"):
        Recorder(Config()).record()


def test_a_failed_recording_leaves_no_file_behind(ffmpeg: FakePopen) -> None:
    ffmpeg.returncode = 1
    ffmpeg.lines = ["Input/output error\n"]

    with pytest.raises(AudioError):
        Recorder(Config()).record()

    assert ffmpeg.destination is not None
    assert not ffmpeg.destination.exists()


def test_an_interrupted_recording_leaves_no_file_behind(ffmpeg: FakePopen) -> None:
    """Ctrl-C is the documented way to quit, and usually lands mid-recording."""

    def interrupt(timeout: float | None = None) -> int:
        raise KeyboardInterrupt

    ffmpeg.wait = interrupt  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt):
        Recorder(Config()).record()

    assert ffmpeg.destination is not None
    assert not ffmpeg.destination.exists()


def test_an_empty_recording_is_an_error(ffmpeg: FakePopen) -> None:
    ffmpeg.writes = ""
    with pytest.raises(AudioError, match="nothing was recorded"):
        Recorder(Config()).record()


def test_a_hanging_recording_is_an_error(ffmpeg: FakePopen) -> None:
    def hang(timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 1)

    ffmpeg.wait = hang  # type: ignore[method-assign]

    with pytest.raises(AudioError, match="did not finish"):
        Recorder(Config()).record()


def test_a_blocked_microphone_explains_the_privacy_setting() -> None:
    """This is the first thing that goes wrong on a new Mac."""
    assert "Privacy & Security" in explain("Input/output error")
    assert "Privacy & Security" in explain("Operation not permitted")


def test_an_unknown_device_says_how_to_list_them() -> None:
    assert "list_devices" in explain("No such device")


def test_only_the_last_line_of_a_long_complaint_is_shown() -> None:
    assert explain("lots\nof\nnoise\nthe actual problem") == "the actual problem"


def test_an_empty_complaint_still_says_something() -> None:
    assert explain("") != ""


# --- transcribing ---------------------------------------------------------


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "ggml-base.en.bin"
    path.write_text("pretend weights")
    return path


def test_what_was_said_is_returned(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    fake = FakeRun(stdout=" what time is it \n")
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    monkeypatch.setattr("subprocess.run", fake)

    assert Transcriber(Config(whisper_model=model)).transcribe(Path("/tmp/x.wav")) == (
        "what time is it"
    )


def test_the_model_and_file_are_passed(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    fake = FakeRun(stdout="hello")
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    monkeypatch.setattr("subprocess.run", fake)

    Transcriber(Config(whisper_model=model)).transcribe(Path("/tmp/x.wav"))

    assert fake.argv[fake.argv.index("--model") + 1] == str(model)
    assert fake.argv[fake.argv.index("--file") + 1] == "/tmp/x.wav"
    assert "--no-timestamps" in fake.argv


def test_it_leaves_a_core_free(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    """A long transcription must not make the rest of the machine unusable."""
    fake = FakeRun(stdout="hello")
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    monkeypatch.setattr("subprocess.run", fake)
    monkeypatch.setattr("os.cpu_count", lambda: 8)

    Transcriber(Config(whisper_model=model)).transcribe(Path("/tmp/x.wav"))

    assert fake.argv[fake.argv.index("--threads") + 1] == "7"


def test_a_missing_model_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    with pytest.raises(TranscriptionError, match="make whisper-model"):
        Transcriber(Config(whisper_model=Path("/nowhere/model.bin"))).transcribe(Path("/x.wav"))


def test_a_missing_whisper_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(TranscriptionError, match="brew install whisper-cpp"):
        Transcriber(Config()).transcribe(Path("/x.wav"))


def test_a_failed_run_reports_what_whisper_said(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    monkeypatch.setattr("subprocess.run", FakeRun(returncode=1, stderr="bad model file"))
    with pytest.raises(TranscriptionError, match="bad model file"):
        Transcriber(Config(whisper_model=model)).transcribe(Path("/x.wav"))


@pytest.mark.parametrize(
    "output",
    ["[BLANK_AUDIO]", "(silence)", "  ", ".", "...", "[ Silence ]", "[music]"],
)
def test_whispers_transcript_of_nothing_is_nothing(output: str) -> None:
    """Left in, these reach the model as if the user had said them."""
    assert clean(output) == ""


def test_annotations_are_removed_but_the_words_are_kept() -> None:
    assert clean("[BLANK_AUDIO] open chrome (coughs)") == "open chrome"


def test_whitespace_is_collapsed() -> None:
    assert clean("  what   time\n  is it  \n") == "what time is it"


# --- speaking -------------------------------------------------------------


def test_an_answer_is_read_aloud(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", fake)

    assert Speaker(Config()).say("it is four o'clock")
    assert fake.argv == ["/usr/bin/say", "--", "it is four o'clock"]


def test_a_chosen_voice_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", fake)

    Speaker(Config(voice="Daniel")).say("hello")

    assert fake.argv[1:3] == ["-v", "Daniel"]


def test_an_answer_starting_with_a_dash_is_still_spoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`say` reads a leading "- " as a flag and refuses the whole thing, so a
    bulleted answer — which the model does produce — went unspoken."""
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", fake)

    assert Speaker(Config()).say("- first item")
    assert fake.argv == ["/usr/bin/say", "--", "- first item"]


def test_nothing_is_said_when_speaking_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", fake)

    assert not Speaker(Config(speak=False)).say("hello")
    assert fake.calls == []


def test_nothing_is_said_when_there_is_nothing_to_say(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", fake)

    assert not Speaker(Config()).say("   ")
    assert fake.calls == []


def test_a_system_without_say_is_simply_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    speaker = Speaker(Config())
    assert not speaker.available
    assert not speaker.say("hello")


@pytest.mark.parametrize(
    "failure",
    [OSError("no audio device"), subprocess.TimeoutExpired(cmd="say", timeout=1)],
)
def test_a_failure_to_speak_never_raises(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """The answer is already on screen. A silent rolex beats a crashed one."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("subprocess.run", FakeRun(raises=failure))

    assert not Speaker(Config()).say("hello")


def test_a_stalled_whisper_does_not_hang_the_loop(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    def hang(*_: Any, **__: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="whisper-cli", timeout=120)

    monkeypatch.setattr("shutil.which", lambda _: "/whisper-cli")
    monkeypatch.setattr("subprocess.run", hang)

    with pytest.raises(TranscriptionError, match="did not finish"):
        Transcriber(Config(whisper_model=model)).transcribe(Path("/x.wav"))
