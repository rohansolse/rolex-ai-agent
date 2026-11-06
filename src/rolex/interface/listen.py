"""The ``rolex listen`` loop: speak, and rolex acts.

Two modes, because always-on listening is the nice one and push-to-talk is the
one that works when the room is noisy or whisper keeps mishearing:

- **wake word** — rolex records short clips, and acts when one starts with its
  name.
- **push to talk** — rolex records only when the user presses Enter.

Everything is local. Nothing is sent anywhere, and nothing is kept: each
recording is deleted as soon as it has been transcribed.
"""

import contextlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rolex.agent.consent import Policy
from rolex.agent.loop import Agent, Answer, Event, ToolFinished, ToolRefused, ToolStarted
from rolex.core.config import Config
from rolex.core.errors import ProviderError, RolexError
from rolex.interface.chat import (
    STEP,
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
from rolex.voice.yesno import understand

LISTEN_WINDOW = 3.0
"""Seconds per clip while waiting for the wake word. Long enough to catch
"rolex" plus the start of a sentence, short enough that the delay before rolex
reacts is not irritating."""

QUIT_PHRASES = frozenset({"stop listening", "goodbye", "exit", "quit", "stop"})

CONSENT_WINDOW = 4.0
"""Seconds to wait for a spoken yes or no. A sentence, not a speech."""

CONSENT_TRIES = 2
"""How many times to ask before taking the silence as a no."""

HEARD = "heard > "

#: One worker: only ever one recording is in flight, and it must not outlive
#: the process, so the threads are daemons.
#: One worker, so only ever one ffmpeg holds the microphone. Its threads are
#: not daemons — Python joins them at exit — which is another reason every
#: exit path calls `Ears.discard`: it waits for the clip in flight and deletes
#: it, rather than leaving the interpreter to block on a recording nobody
#: wants.
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rolex-record")


RECORD_GRACE = 30.0
"""Seconds to wait for an abandoned recording before giving up on deleting it."""


@dataclass(slots=True)
class Ears:
    """Everything needed to hear. Grouped so tests can fake it."""

    recorder: Recorder
    transcriber: Transcriber

    pending: "Future[Path] | None" = None
    """A clip already being recorded while the last one is transcribed."""

    def listen(self, *, seconds: float | None = None) -> str:
        """Record and transcribe one utterance. Empty means nothing was said.

        Any clip already in flight is dropped first. Two recordings at once
        means two ffmpegs fighting over one microphone: the utterance ends up
        split across both files, and on a device that cannot be opened twice
        the second one simply fails.
        """
        self.discard()
        return self._read(self.recorder.record(seconds=seconds))

    def next_clip(self, *, seconds: float) -> str:
        """The next clip, with the microphone never left idle.

        The clip after this one starts recording before this one is
        transcribed. Done the obvious way round, the microphone is shut for
        however long whisper takes — roughly a second in every four — and a
        wake word spoken in that gap is never heard at all. That was the
        commonest reason rolex appeared to ignore someone.
        """
        if self.pending is None:
            self.pending = self._start(seconds)

        try:
            recording = self.pending.result()
        except Exception:
            # A failed recording must not stay in `pending`, or every later
            # call re-raises it instantly instead of trying the microphone
            # again. Ctrl-C is deliberately not caught: the future is still
            # running, and `discard` needs the handle to clean it up.
            self.pending = None
            raise

        self.pending = self._start(seconds)
        return self._read(recording)

    def discard(self) -> None:
        """Throw away a clip recorded while rolex was busy answering.

        It holds whatever the room was doing during the last answer — often
        rolex's own voice — which is not a command.
        """
        pending, self.pending = self.pending, None
        if pending is None:
            return
        # A failed recording leaves nothing to delete, which is fine here.
        with contextlib.suppress(Exception):
            pending.result(timeout=RECORD_GRACE).unlink(missing_ok=True)

    def _start(self, seconds: float) -> "Future[Path]":
        return _POOL.submit(self.recorder.record, seconds=seconds)

    def _read(self, recording: Path) -> str:
        """Transcribe a recording and delete it.

        Deleted whatever happens: a recording of someone's room is not
        something to leave lying in /tmp.
        """
        try:
            return self.transcriber.transcribe(recording)
        finally:
            Path(recording).unlink(missing_ok=True)


@dataclass(slots=True)
class SpokenConsent:
    """Asks permission out loud and listens for the answer."""

    ears: "Ears"
    voice: Speaker
    tries: int = CONSENT_TRIES

    def ask(self, tool: Tool, arguments: Mapping[str, Any]) -> bool:
        """Put the question to the user and wait for a spoken yes or no.

        Anything that is not clearly a yes is a no. Mishearing a refusal as
        consent opens something the user did not want; mishearing consent as
        unclear only costs them saying it again.
        """
        question = _question(tool, arguments)

        for attempt in range(self.tries):
            print(f"{STEP}{question}", flush=True)
            self.voice.say(question)

            try:
                # `listen`, not `next_clip`: whatever was recording while the
                # question was being spoken holds rolex's own voice, not an
                # answer. It is dropped and the answer recorded fresh.
                said = self.ears.listen(seconds=CONSENT_WINDOW)
            except RolexError as exc:
                print(f"{STEP}  could not hear an answer: {exc}", flush=True)
                return False
            except KeyboardInterrupt:
                print()
                return False

            answer = understand(said)
            if answer is not None:
                print(f"{STEP}  heard: {said.strip() or '(nothing)'}", flush=True)
                return answer

            if attempt + 1 < self.tries:
                self.voice.say("Sorry, was that a yes or a no?")

        print(f"{STEP}  no clear answer — taking that as no", flush=True)
        self.voice.say("I did not catch that, so I have not done it.")
        return False


def _question(tool: Tool, arguments: Mapping[str, Any]) -> str:
    """The permission question, phrased for someone listening rather than reading."""
    if tool.name == "open_app" and "name" in arguments:
        return f"Open {arguments['name']}?"
    if tool.name == "open_url" and "url" in arguments:
        return f"Open {arguments['url']} in the browser?"
    if tool.name == "search_web" and "query" in arguments:
        return f"Search the web for {arguments['query']}?"

    spoken = ", ".join(str(value) for value in arguments.values())
    name = tool.name.replace("_", " ")
    return f"Run {name}{f' with {spoken}' if spoken else ''}?"


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

    # Asked out loud, not at the keyboard. The user is talking to rolex; a
    # typed [y/N] goes unanswered, and in push-to-talk the Enter meant for the
    # next recording was swallowed by the prompt and read as a refusal.
    asker = SpokenConsent(ears=hearing, voice=voice)

    agent = Agent(
        provider=backend,
        tools=registered() if tools is None else tools,
        policy=Policy(ask=asker.ask, allow_destructive=config.allow_destructive),
        history=[Message("system", system_prompt(config))],
    )

    _greet(config, voice, push_to_talk=push_to_talk)

    while True:
        try:
            said = _next_command(hearing, push_to_talk=push_to_talk)
        except KeyboardInterrupt:
            print("\nStopped listening.")
            hearing.discard()
            return 0
        except RolexError as exc:
            print(f"\nrolex: {exc}\n", file=sys.stderr)
            hearing.discard()
            return 1

        if said is None:
            continue
        if said.casefold().strip(" .!?") in QUIT_PHRASES:
            print("Stopped listening.")
            voice.say("Goodbye.")
            hearing.discard()
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

        # Whatever was recorded while rolex was thinking and talking is not a
        # command — it is mostly rolex's own answer coming out of the speaker.
        if not push_to_talk:
            hearing.discard()


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

    heard_text = ears.next_clip(seconds=LISTEN_WINDOW)
    if not heard_text:
        return None

    rest = strip(heard_text)
    if rest is None:
        return None

    if rest:
        return rest

    # Woken, but nothing asked yet. The next clip is already recording — it
    # began the moment this one finished — so it holds the command the user
    # is speaking right now. Taking it is both faster and avoids opening a
    # second microphone alongside the one already running.
    print(f"{STEP}listening…", flush=True)
    return ears.next_clip(seconds=LISTEN_WINDOW) or None


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
