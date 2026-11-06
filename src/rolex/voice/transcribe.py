"""Turning recorded speech into text.

Runs ``whisper-cli`` from whisper.cpp as a subprocess. A Python binding would
pull in ctranslate2 and friends — several hundred megabytes on a machine with
13 GB free — for a job a command-line call does perfectly well.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rolex.core.config import Config
from rolex.core.errors import RolexError

#: Whisper emits these for silence, music, or a cough. They are not speech.
_NOISE = re.compile(r"[\[(][^\])]*[\])]")

#: Whisper's own transcript of nothing at all.
_EMPTY_MARKERS = frozenset({"", ".", "...", "[blank_audio]", "(silence)", "you"})


TIMEOUT = 120.0
"""Seconds to wait. Generous — a long clip on CPU is slow — but bounded, so a
stuck whisper cannot hang the listen loop forever."""


class TranscriptionError(RolexError):
    """The recording could not be turned into text."""


@dataclass(frozen=True, slots=True)
class Transcriber:
    """Speech to text, locally."""

    config: Config

    def transcribe(self, recording: Path) -> str:
        """What was said in `recording`, or an empty string for silence.

        Raises:
            TranscriptionError: whisper.cpp is missing, the model is missing,
                or the run failed.
        """
        executable = self._executable()
        model = self.config.whisper_model

        if not model.is_file():
            raise TranscriptionError(
                f"the speech model is not at {model}. Download it with:\n  make whisper-model"
            )

        argv = [
            executable,
            "--model",
            str(model),
            "--file",
            str(recording),
            "--no-timestamps",
            "--no-prints",
            "--language",
            "en",
            # One thread fewer than the machine has, so a long transcription
            # does not make the rest of the system unusable.
            "--threads",
            str(self.config.threads),
        ]

        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=TIMEOUT, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError(
                f"whisper did not finish within {TIMEOUT:.0f} seconds"
            ) from exc
        except OSError as exc:
            raise TranscriptionError(f"could not run {executable}: {exc}") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise TranscriptionError(detail or f"whisper exited with {result.returncode}")

        return clean(result.stdout)

    def _executable(self) -> str:
        found = shutil.which("whisper-cli")
        if found is None:
            raise TranscriptionError(
                "whisper.cpp is not installed. Install it with:  brew install whisper-cpp"
            )
        return found


def clean(output: str) -> str:
    """The spoken words, with whisper's annotations and padding removed.

    Whisper writes ``[BLANK_AUDIO]`` for silence and ``(coughs)`` for noise.
    Left in, those reach the model as if the user had said them.
    """
    text = _NOISE.sub(" ", output)
    text = " ".join(text.split())
    return "" if text.casefold().strip(".") in _EMPTY_MARKERS else text
