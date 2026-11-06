"""Recording from the microphone.

Uses ``ffmpeg`` rather than a Python audio binding, for two reasons: ffmpeg is
already installed, and binding to PortAudio would be this project's first
compiled dependency.

Recording stops when the speaker does. ffmpeg's ``silencedetect`` filter only
*reports* silence — it does not end the recording — so its output is read as
it arrives and ffmpeg is asked to stop at the first silence that follows
speech. Without that, every clip would run the full time limit and a
three-word command would leave the user waiting.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rolex.core.config import Config
from rolex.core.errors import RolexError

SAMPLE_RATE = 16_000
"""What whisper.cpp wants. Anything else has to be resampled anyway."""

SILENCE_THRESHOLD = "-35dB"
"""How quiet counts as quiet. Below this a room's own hum ends the sentence;
above it a softly spoken word does not register."""

GRACE = 10.0
"""Seconds beyond the limit before ffmpeg is considered stuck."""


class AudioError(RolexError):
    """The microphone could not be recorded from."""


#: silencedetect's two events and their timestamps, e.g.
#: "[Parsed_silencedetect_0 @ 0x...] silence_start: 1.475563".
_EVENT = re.compile(r"silence_(start|end): ([0-9.]+)")

MIN_LEAD = 0.3
"""Seconds of sound before a silence that make it count as "after speech".

A clip that begins with talking produces a ``silence_start`` and no
``silence_end`` at all — the first version of this waited for an end that
never came, and so never stopped early. A ``silence_start`` at 0 means the
user has not begun yet; one at 1.5 means they have finished.
"""


@dataclass(slots=True)
class Listener:
    """Reads ffmpeg's silencedetect output and decides when to stop.

    Separated from the subprocess so the decision can be tested without a
    microphone — which is the part that was wrong the first time, and was only
    caught by running it against real ffmpeg.
    """

    spoke: bool = field(default=False)
    """Whether any speech has been heard. Silence before the user starts
    talking must not end the recording."""

    def should_stop(self, line: str) -> bool:
        """Whether this line of ffmpeg output means the utterance is over.

        One line can carry several events, because ffmpeg's progress output
        uses carriage returns and the whole lot arrives together.
        """
        for kind, timestamp in _EVENT.findall(line):
            if kind == "end":
                # Silence ended, so speech began.
                self.spoke = True
                continue
            if self.spoke or float(timestamp) >= MIN_LEAD:
                return True
        return False


@dataclass(frozen=True, slots=True)
class Recording:
    """A captured clip, and how it came to an end."""

    path: Path

    ended_on_silence: bool
    """True when the speaker stopped, False when the clock ran out.

    The difference matters: a clip that ran out of time probably cut the user
    off mid-sentence, and the rest of what they said is about to arrive in
    the next one.
    """


@dataclass(frozen=True, slots=True)
class Recorder:
    """Captures one utterance at a time."""

    config: Config

    def record(self, *, seconds: float | None = None) -> Recording:
        """Record until the speaker stops, or the time limit.

        The caller owns the file and should delete it.

        Raises:
            AudioError: ffmpeg is missing, or the microphone could not be read.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioError("ffmpeg is not installed. Install it with:  brew install ffmpeg")

        limit = self.config.listen_seconds if seconds is None else seconds
        # A name, not an open handle: ffmpeg writes the file, not this process.
        descriptor, path = tempfile.mkstemp(suffix=".wav", prefix="rolex-")
        os.close(descriptor)
        destination = Path(path)

        try:
            ended_on_silence = self._capture(ffmpeg, destination, limit)
        except BaseException:
            # BaseException, not Exception: Ctrl-C is the documented way to
            # quit and usually lands mid-recording. A wav of someone's room
            # must not survive it.
            destination.unlink(missing_ok=True)
            raise
        return Recording(path=destination, ended_on_silence=ended_on_silence)

    def argv(self, ffmpeg: str, destination: Path, limit: float) -> list[str]:
        """The ffmpeg command used to record. Separate so it can be tested."""
        device = self.config.audio_device or "default"
        return [
            ffmpeg,
            "-hide_banner",
            # info, not error: silencedetect reports through the log, and at
            # 'error' its output never appears, so nothing could stop early.
            "-loglevel",
            "info",
            "-f",
            "avfoundation",
            "-i",
            f":{device}",
            "-t",
            f"{limit:.1f}",
            "-af",
            f"silencedetect=noise={SILENCE_THRESHOLD}:d={self.config.silence_seconds:.1f}",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-y",
            str(destination),
        ]

    def _capture(self, ffmpeg: str, destination: Path, limit: float) -> bool:
        """Run ffmpeg. Returns whether the speaker stopped before the clock did."""
        argv = self.argv(ffmpeg, destination, limit)
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        ended_on_silence, errors = self._watch(process, limit)

        if process.returncode not in {0, 255}:
            # 255 is what ffmpeg returns when asked to quit early, which is
            # exactly what a finished sentence looks like.
            raise AudioError(explain(errors))

        if not destination.exists() or destination.stat().st_size == 0:
            raise AudioError("nothing was recorded")

        return ended_on_silence

    def _watch(self, process: subprocess.Popen[str], limit: float) -> tuple[bool, str]:
        """Follow ffmpeg's output, stopping it once the speaker stops.

        Returns whether it was the speaker stopping that ended the clip, and
        everything ffmpeg said.
        """
        listener = Listener()
        lines: list[str] = []
        ended_on_silence = False

        try:
            if process.stderr is not None:
                for line in process.stderr:
                    lines.append(line)
                    if listener.should_stop(line):
                        ended_on_silence = True
                        break
            _finish(process, timeout=limit + GRACE)
        except subprocess.TimeoutExpired as exc:
            _kill(process)
            raise AudioError("the recording did not finish") from exc
        finally:
            # Whatever happened — including Ctrl-C — ffmpeg must not be left
            # holding the microphone.
            if process.poll() is None:
                _kill(process)

        return ended_on_silence, "".join(lines).strip()


def _finish(process: subprocess.Popen[str], *, timeout: float) -> None:
    """Ask ffmpeg to stop, and wait for it.

    'q' on stdin rather than a signal: ffmpeg then writes the wav header it
    has been holding back, so the file is playable. A signal leaves it
    truncated.

    Raises:
        subprocess.TimeoutExpired: it did not stop in time.
    """
    try:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.write("q")
            process.stdin.flush()
            process.stdin.close()
    except (OSError, ValueError):
        process.terminate()

    process.wait(timeout=timeout)


def _kill(process: subprocess.Popen[str]) -> None:
    """Stop ffmpeg without raising, whatever state it is in.

    Called from cleanup paths, where raising would mask the error that got us
    there — which it did, until this was separated out.
    """
    try:
        process.kill()
        process.wait(timeout=5)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass


def explain(stderr: str) -> str:
    """Turn ffmpeg's complaint into something a user can act on."""
    lowered = stderr.lower()

    # macOS reports a blocked microphone as an I/O error, not as a permission
    # problem, which is why the obvious wording is not enough on its own.
    blocked = ("input/output error", "permission", "not permitted", "not authorized")
    if any(phrase in lowered for phrase in blocked):
        return (
            "the microphone could not be opened. macOS needs to be told this is "
            "allowed: System Settings -> Privacy & Security -> Microphone, and "
            "turn on the terminal you are running rolex in."
        )
    if "no such" in lowered or "unknown input" in lowered:
        return (
            "that microphone was not found. List them with:  "
            'ffmpeg -f avfoundation -list_devices true -i ""'
        )
    return stderr.splitlines()[-1] if stderr else "the recording failed for an unknown reason"
