"""The ``rolex chat`` REPL.

A typed conversation with the configured model, running through the agent
loop so the model can use tools. Voice arrives in Phase 4.
"""

import sys
from collections.abc import Iterable, Sequence

from rolex.agent.loop import Agent, Answer, Event, ToolFinished, ToolStarted
from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.llm.base import LLMProvider, Message
from rolex.llm.factory import build_provider
from rolex.tools.base import Tool
from rolex.tools.clock import get_time

DEFAULT_TOOLS: tuple[Tool, ...] = (get_time,)
"""What rolex can do. One tool in Phase 2; the registry fills this in Phase 3."""

IDENTITY = """Your name is rolex. You are an assistant running on the user's \
Mac. When asked who you are, say you are rolex."""

OFFLINE = """You run entirely on this computer, with no internet connection. \
You cannot browse the web, fetch anything, or look anything up. If asked, say \
so plainly rather than claiming you are online."""

MANNER = """Keep answers short and direct — a sentence or two unless more is \
asked for.

You have tools, but most questions do not need one. Answer from what you \
already know. Reach for a tool only when the question is specifically about \
something the tool provides and you would otherwise have to guess. A greeting \
is not such a question. Once a tool has given you its result, answer in words \
— do not call it again.

The user writes in Marathi, Hindi, English, or a mix, and often types Marathi \
and Hindi in Latin letters ("tu kon ahes", "browser open kar", "kaise ho"). \
Understand all of it. Reply in simple English."""


def system_prompt(config: Config) -> str:
    """The instructions rolex opens every conversation with.

    The offline paragraph is only included for the local provider, because for
    a hosted one it would be a lie — and a model told something false about
    itself will repeat it confidently.
    """
    parts = [IDENTITY, OFFLINE, MANNER] if config.provider == "local" else [IDENTITY, MANNER]
    return "\n\n".join(parts)


QUIT_WORDS = frozenset({"exit", "quit", "bye", ":q"})

#: How many user/assistant messages to keep. A 3B model's context window is
#: small, and Ollama silently truncates from the front when it overflows —
#: which would drop the system prompt and rolex would forget its own name.
#: Trimming here instead keeps the system prompt and makes the loss visible.
HISTORY_TURNS = 20

PROMPT = "you   > "
ANSWER = "rolex > "
STEP = "       · "
"""Prefix for the work rolex does on the way to an answer."""


def run_chat(
    config: Config,
    *,
    provider: LLMProvider | None = None,
    tools: Sequence[Tool] | None = None,
) -> int:
    """Run the REPL until the user leaves. Returns the process exit code."""
    backend = build_provider(config) if provider is None else provider

    if not backend.is_available():
        explain_unavailable(config, backend)
        return 1

    agent = Agent(
        provider=backend,
        tools=DEFAULT_TOOLS if tools is None else tools,
        history=[Message("system", system_prompt(config))],
    )

    names = ", ".join(tool.name for tool in agent.tools) or "none"
    print(f"rolex chat — {backend.name}. Tools: {names}. Type 'exit' to leave.\n")

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

        try:
            _show(agent.ask(line))
        except ProviderError as exc:
            # One bad turn must not end the session. Agent.ask only commits a
            # turn to history once it finishes, so there is nothing to undo.
            print(f"\n[{exc}]\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n[interrupted]\n")
            continue

        agent.history = _trim(agent.history)


def _trim(history: list[Message]) -> list[Message]:
    """Keep the system prompt and the most recent turns, drop the middle.

    Trimming stops partway if it would start the kept history with a tool
    result: a ``tool`` message whose matching request has been dropped
    confuses the model more than the extra length costs.
    """
    system, rest = history[0], history[1:]
    kept = rest[-HISTORY_TURNS:]
    while kept and kept[0].role == "tool":
        kept = kept[1:]
    return [system, *kept]


def _show(events: Iterable[Event]) -> None:
    """Print each step of the agent's work, then its answer."""
    for event in events:
        if isinstance(event, ToolStarted):
            shown = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
            print(f"{STEP}{event.name}({shown})", flush=True)
        elif isinstance(event, ToolFinished):
            mark = "failed: " if event.failed else ""
            print(f"{STEP}  {mark}{_short(event.result)}", flush=True)
        elif isinstance(event, Answer):
            if event.gave_up:
                print(f"{STEP}(gave up asking for tools)", flush=True)
            print(f"{ANSWER}{event.content}\n", flush=True)


def _short(text: str, limit: int = 120) -> str:
    """One line, trimmed, so a chatty tool does not flood the terminal."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


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
