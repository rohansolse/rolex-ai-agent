"""The ``rolex listen`` loop: speak, and rolex acts.

Two modes, because always-on listening is the nice one and push-to-talk is the
one that works when the room is noisy or whisper keeps mishearing:

- **wake word** — rolex records short clips, and acts when one starts with its
  name.
- **push to talk** — rolex records only when the user presses Enter.

Everything is local. Nothing is sent anywhere, and nothing is kept: each
recording is deleted as soon as it has been transcribed.
"""

import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rolex.agent.consent import Policy
from rolex.agent.loop import Agent, Answer, Event, ToolFinished, ToolRefused, ToolStarted
from rolex.core.config import Config
from rolex.core.errors import ProviderError, RolexError
from rolex.interface.chat import (
    STEP,
    ask_at_terminal,
    explain_unavailable,
    short,
    system_prompt,
    trim,
)
from rolex.llm.base import LLMProvider, Message
from rolex.llm.factory import build_provider
from rolex.tools import registered
from rolex.tools.base import Tool
from rolex.voice.record import Recorder
from rolex.voice.speak import Speaker
from rolex.voice.transcribe import Transcriber
from rolex.voice.wake import WAKE_WORD, strip

LISTEN_WINDOW = 3.0
"""Seconds per clip while waiting for the wake word. Long enough to catch
"rolex" plus the start of a sentence, short enough that the delay before rolex
reacts is not irritating."""

QUIT_PHRASES = frozenset({"stop listening", "goodbye", "exit", "quit", "stop"})

HEARD = "heard > "


@dataclass(slots=True)
class Ears:
    """Everything needed to hear a command. Grouped so tests can fake it."""

    recorder: Recorder
    transcriber: Transcriber

    def listen(self, *, seconds: float | None = None) -> str:
        """Record and transcribe one utterance. Empty means nothing was said."""
        recording = self.recorder.record(seconds=seconds)
        try:
            return self.transcriber.transcribe(recording)
        finally:
            # Deleted immediately: a recording of someone's room is not
            # something to leave lying in /tmp.
            Path(recording).unlink(missing_ok=True)


def run_listen(
    config: Config,
    *,
    provider: LLMProvider | None = None,
    tools: Sequence[Tool] | None = None,
    ears: Ears | None = None,
    speaker: Speaker | None = None,
    push_to_talk: bool = False,
) -> int:
    """Listen until told to stop. Returns the process exit code."""
    backend = build_provider(config) if provider is None else provider

    if not backend.is_available():
        explain_unavailable(config, backend)
        return 1

    hearing = Ears(Recorder(config), Transcriber(config)) if ears is None else ears
    voice = Speaker(config) if speaker is None else speaker

    agent = Agent(
        provider=backend,
        tools=registered() if tools is None else tools,
        policy=Policy(ask=ask_at_terminal, allow_destructive=config.allow_destructive),
        history=[Message("system", system_prompt(config))],
    )

    _greet(config, voice, push_to_talk=push_to_talk)

    while True:
        try:
            said = _next_command(hearing, push_to_talk=push_to_talk)
        except KeyboardInterrupt:
            print("\nStopped listening.")
            return 0
        except RolexError as exc:
            print(f"\nrolex: {exc}\n", file=sys.stderr)
            return 1

        if said is None:
            continue
        if said.casefold().strip(" .!?") in QUIT_PHRASES:
            print("Stopped listening.")
            voice.say("Goodbye.")
            return 0

        print(f"{HEARD}{said}")

        try:
            _answer(agent.ask(said), voice)
        except ProviderError as exc:
            print(f"\n[{exc}]\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[interrupted]\n")

        # Same reason as in chat: a long session would otherwise overflow the
        # context window and Ollama would drop the system prompt from the front.
        agent.history = trim(agent.history)


def _greet(config: Config, voice: Speaker, *, push_to_talk: bool) -> None:
    """Say what mode this is, and whether rolex will answer aloud."""
    how = "press Enter to talk" if push_to_talk else f'say "{WAKE_WORD}" to start'
    aloud = "speaking" if voice.available else "silent (no voice)"
    print(f"rolex listen — {how}. Answers {aloud}. Ctrl-C to stop.\n")
    if not config.whisper_model.is_file():
        print(f"{STEP}warning: no speech model at {config.whisper_model}\n", file=sys.stderr)


def _next_command(ears: Ears, *, push_to_talk: bool) -> str | None:
    """One command from the user, or None if nothing was said to rolex."""
    if push_to_talk:
        try:
            input(f"{STEP}press Enter, then speak… ")
        except EOFError:
            # `rolex talk < /dev/null`, or Ctrl-D. Nothing more is coming.
            raise KeyboardInterrupt from None
        return ears.listen() or None

    heard_text = ears.listen(seconds=LISTEN_WINDOW)
    if not heard_text:
        return None

    rest = strip(heard_text)
    if rest is None:
        return None

    if rest:
        return rest

    # Woken, but nothing asked yet. Listen again for the actual command.
    print(f"{STEP}listening…", flush=True)
    return ears.listen() or None


def _answer(events: Iterable[Event], voice: Speaker) -> None:
    """Print the agent's work, and read its answer aloud."""
    for event in events:
        _show(event, voice)


def _show(event: Event, voice: Speaker) -> None:
    """One step of the agent's work, on screen and, for an answer, aloud."""
    if isinstance(event, ToolStarted):
        shown = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
        print(f"{STEP}{event.name}({shown})", flush=True)
    elif isinstance(event, ToolFinished):
        mark = "failed: " if event.failed else ""
        print(f"{STEP}  {mark}{short(event.result)}", flush=True)
    elif isinstance(event, ToolRefused):
        print(f"{STEP}  refused", flush=True)
    elif isinstance(event, Answer):
        if event.gave_up:
            print(f"{STEP}(gave up asking for tools)", flush=True)
        print(f"rolex > {event.content}\n", flush=True)
        voice.say(event.content)
