"""Saying things out loud.

macOS ships ``say``, which needs no model, no download and no dependency. It
sounds like a computer, which is a fair trade for Phase 4.
"""

import shutil
import subprocess
from dataclasses import dataclass

from rolex.core.config import Config

TIMEOUT = 120.0
"""Long answers take a while to read. Still bounded, so a stuck `say` cannot
hold the conversation open forever."""


@dataclass(frozen=True, slots=True)
class Speaker:
    """Reads text aloud, if speaking is switched on."""

    config: Config

    @property
    def available(self) -> bool:
        """Whether anything can be said at all."""
        return self.config.speak and shutil.which("say") is not None

    def say(self, text: str) -> bool:
        """Read `text` aloud. Returns whether it was actually spoken.

        Never raises. A silent rolex is a nuisance; a crashed one is worse,
        and the answer is already on screen either way.
        """
        spoken = text.strip()
        if not spoken or not self.available:
            return False

        executable = shutil.which("say")
        if executable is None:
            return False

        argv = [executable]
        if self.config.voice:
            argv += ["-v", self.config.voice]
        # Everything after `--` is text, never a flag. Without it an answer
        # beginning with "- " — a bulleted list, which the model does produce
        # — makes `say` exit with "invalid option" and nothing is spoken.
        argv += ["--", spoken]

        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=TIMEOUT, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
