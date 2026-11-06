"""Listening continuously, instead of in fixed-length clips.

Recording three seconds at a time loses whatever falls across a clip
boundary: "rolex, what is the time today" arrives as "rolex what is the time"
with "today" in the next clip, where it is ignored for having no wake word.
No model, however large, can reassemble a sentence that was cut in half
before it reached it.

So the microphone is left open and ``whisper-stream`` decides where sentences
end, using voice activity rather than a clock. It re-transcribes a rolling
buffer, which means every utterance it reports also contains the ones before
it, so :class:`Transcript` tracks what has already been seen and yields only
what is new.
"""

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue

from rolex.core.config import Config
from rolex.core.errors import StreamError

BUFFER_MS = 10_000
"""How much audio whisper-stream keeps and re-transcribes.

Ten seconds: long enough for any spoken command, short enough that the work
of re-transcribing it stays under a second or so.
"""

VAD_THRESHOLD = 0.6
"""How loud counts as speech. whisper.cpp's own default."""

#: "### Transcription 7 START | t0 = 0 ms | t1 = 14207 ms"
_HEADER = re.compile(r"### Transcription \d+ START \| t0 = (\d+) ms \| t1 = (\d+) ms")

#: "[00:00:11.840 --> 00:00:13.520]   Rolex Open Google Chrome"
_SEGMENT = re.compile(r"\[[\d:.]+ --> [\d:.]+\]\s*(.*)")


@dataclass(slots=True)
class Transcript:
    """Reads whisper-stream's output and reports only what is newly spoken.

    Pure: no subprocess, no microphone. The parsing is where the mistakes
    are, and this way they can be made visible without saying anything out
    loud.
    """

    said: list[str] = field(default_factory=list)
    """What has already been reported from the current buffer."""

    buffer_ms: int = 0
    """How much audio the last block covered. A drop means the rolling buffer
    was emptied and everything in it is about to be transcribed afresh."""

    def feed(self, line: str) -> list[str]:
        """What is newly spoken in this line of output, if anything.

        Matched on the words rather than the timestamps. A re-transcription
        reports the same sentence again with a *later* end time — it has had
        more audio to work with — so a timestamp alone cannot tell a repeat
        from something new.
        """
        header = _HEADER.search(line)
        if header:
            covered = int(header.group(2))
            if covered < self.buffer_ms:
                # The buffer was emptied; nothing in it has been heard before.
                self.said.clear()
            self.buffer_ms = covered
            return []

        found = _SEGMENT.search(line)
        if not found:
            return []

        spoken = found.group(1).strip()
        if not spoken:
            return []

        fresh = self._new_part(spoken)
        if not fresh:
            return []

        self.said.append(spoken)
        return [fresh]

    def _new_part(self, spoken: str) -> str:
        """The part of `spoken` not already reported.

        A re-transcription sometimes merges a new sentence onto the end of
        one already heard. Reported whole, rolex would answer the old
        question again; dropped whole, the new command would be lost.
        """
        for earlier in reversed(self.said):
            if spoken == earlier:
                return ""
            if spoken.startswith(earlier):
                return spoken[len(earlier) :].strip()
        return spoken


@dataclass(slots=True)
class Stream:
    """whisper-stream, running for as long as rolex is listening."""

    config: Config
    process: subprocess.Popen[str] | None = None
    lines: "Queue[str | None]" = field(default_factory=Queue)
    reader: threading.Thread | None = None
    transcript: Transcript = field(default_factory=Transcript)

    ended: bool = False
    """Whisper-stream has stopped. Latched: the end-of-output sentinel comes
    once, and without remembering it the loop would poll an empty queue for
    ever instead of saying the microphone had died."""

    def argv(self, executable: str) -> list[str]:
        """The command used. Separate so it can be checked without a microphone."""
        argv = [
            executable,
            "--model",
            str(self.config.whisper_model),
            # step 0 is whisper.cpp's voice-activity mode: it waits for a
            # pause rather than cutting at a fixed length.
            "--step",
            "0",
            "--length",
            str(BUFFER_MS),
            "--vad-thold",
            str(VAD_THRESHOLD),
            "--language",
            "en",
            "--threads",
            str(self.config.threads),
        ]
        if self.config.stream_device:
            argv += ["--capture", self.config.stream_device]
        return argv

    def start(self) -> None:
        """Open the microphone and keep it open.

        Raises:
            StreamError: whisper-stream is missing or would not start.
        """
        executable = shutil.which("whisper-stream")
        if executable is None:
            raise StreamError(
                "whisper-stream is not installed. Install it with:  brew install whisper-cpp"
            )
        if not self.config.whisper_model.is_file():
            raise StreamError(
                f"the speech model is not at {self.config.whisper_model}. "
                f"Download it with:\n  make whisper-model"
            )

        try:
            self.process = subprocess.Popen(
                self.argv(executable),
                stdout=subprocess.PIPE,
                # Discarded: whisper.cpp writes pages of backend and model
                # detail to stderr before it says anything useful.
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise StreamError(f"could not start whisper-stream: {exc}") from exc

        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        """Move whisper-stream's output onto the queue, off the main thread."""
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def listen(self, *, timeout: float | None = None) -> str | None:
        """The next newly spoken utterance, or None if nothing was said in time.

        Raises:
            StreamError: the microphone stopped.
        """
        if self.ended:
            raise self._stopped()

        try:
            line = self.lines.get(timeout=timeout)
        except Empty:
            return None

        if line is None:
            self.ended = True
            raise self._stopped()

        spoken = self.transcript.feed(line)
        return spoken[0] if spoken else None

    def _stopped(self) -> StreamError:
        return StreamError(
            "the microphone stopped. If macOS has not been given permission, "
            "allow it in System Settings -> Privacy & Security -> Microphone."
        )

    def drain(self) -> None:
        """Throw away anything heard while rolex was talking.

        The microphone stays open while rolex answers, so it hears its own
        voice. Those are not commands.
        """
        while True:
            try:
                line = self.lines.get_nowait()
            except Empty:
                return
            if line is None:
                self.ended = True
                return
            self.transcript.feed(line)

    def stop(self) -> None:
        """Close the microphone."""
        process, self.process = self.process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> "Stream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
