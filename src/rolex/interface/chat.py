"""The ``rolex chat`` REPL.

A plain typed conversation with the configured model. No tools, no voice —
those are Phases 3 and 4. This exists to prove the model is reachable and
answering.
"""

import sys
from collections.abc import Iterable

from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.llm.base import LLMProvider, Message
from rolex.llm.factory import build_provider

SYSTEM_PROMPT = """Your name is rolex. You are an assistant running locally on \
the user's Mac. When asked who you are, say you are rolex.

Keep answers short and direct — a sentence or two unless more is asked for.

The user writes in Marathi, Hindi, English, or a mix, and often types Marathi \
and Hindi in Latin letters ("tu kon ahes", "browser open kar", "kaise ho"). \
Understand all of it. Reply in simple English."""

QUIT_WORDS = frozenset({"exit", "quit", "bye", ":q"})

#: How many user/assistant messages to keep. A 3B model's context window is
#: small, and Ollama silently truncates from the front when it overflows —
#: which would drop the system prompt and rolex would forget its own name.
#: Trimming here instead keeps the system prompt and makes the loss visible.
HISTORY_TURNS = 20

PROMPT = "you   > "
ANSWER = "rolex > "


def run_chat(config: Config, *, provider: LLMProvider | None = None) -> int:
    """Run the REPL until the user leaves. Returns the process exit code."""
    backend = build_provider(config) if provider is None else provider

    if not backend.is_available():
        explain_unavailable(config, backend)
        return 1

    print(f"rolex chat — {backend.name}. Type 'exit' to leave.\n")

    history = [Message("system", SYSTEM_PROMPT)]

    while True:
        try:
            line = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line.lower() in QUIT_WORDS:
            return 0

        history.append(Message("user", line))

        try:
            answer = _say(backend.stream(history))
        except ProviderError as exc:
            # The conversation survives one bad turn; drop the unanswered
            # question so the next request is not sent with a dangling user
            # message the model never replied to.
            history.pop()
            print(f"\n[{exc}]\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            history.pop()
            print("\n[interrupted]\n")
            continue

        history.append(Message("assistant", answer))
        history = _trim(history)


def _trim(history: list[Message]) -> list[Message]:
    """Keep the system prompt and the most recent turns, drop the middle."""
    system, rest = history[0], history[1:]
    return [system, *rest[-HISTORY_TURNS:]]


def _say(chunks: Iterable[str]) -> str:
    """Print chunks as they arrive and return the whole answer."""
    print(ANSWER, end="", flush=True)
    parts: list[str] = []
    for chunk in chunks:
        parts.append(chunk)
        print(chunk, end="", flush=True)
    print("\n")
    return "".join(parts)


def explain_unavailable(config: Config, backend: LLMProvider) -> None:
    """Say what is actually wrong, and the command that fixes it."""
    print(f"rolex: {backend.name} is not ready.\n", file=sys.stderr)

    lister = getattr(backend, "installed_models", None)
    if lister is None:
        return

    try:
        installed: list[str] = lister()
    except UnreachableError as exc:
        print(f"{exc}\n", file=sys.stderr)
        return
    except ProviderError as exc:
        # Reachable, but it answered with something unusable. Saying "not
        # running" here would send the user chasing the wrong problem.
        print(f"The server answered, but not usefully: {exc}\n", file=sys.stderr)
        return

    print(f"The server is up but '{config.model}' is not pulled. Get it with:\n", file=sys.stderr)
    print(
        f"  OLLAMA_MODELS={config.ollama_models_dir} ollama pull {config.model}\n", file=sys.stderr
    )
    if installed:
        print(f"Models already on disk: {', '.join(sorted(set(installed)))}", file=sys.stderr)
