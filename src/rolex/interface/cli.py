"""Command line entrypoint.

Voice arrives in Phase 4; until then everything is typed.
"""

import shutil
import sys
from pathlib import Path

from rolex import __version__
from rolex.core.config import Config, load_config
from rolex.core.errors import RolexError
from rolex.core.logging import configure_logging
from rolex.interface.chat import explain_unavailable as run_chat_help
from rolex.interface.chat import run_chat
from rolex.interface.listen import run_listen
from rolex.llm.factory import build_provider
from rolex.tools import registered

USAGE = """usage: rolex <command>

rolex is a voice-driven assistant that runs on a local LLM.

commands:
  chat            type to rolex
  listen          speak to rolex; say "rolex" to wake it
  talk            speak to rolex, pressing Enter each time instead
  doctor          check that everything rolex needs is in place

  -V, --version   print the version and exit
  -h, --help      print this message and exit

rolex uses tools to do things. `rolex doctor` lists them. Anything that
changes the machine asks first. See PROJECT_ROADMAP.md."""


def main(argv: list[str] | None = None) -> int:
    """Run the rolex CLI. Returns the process exit code."""
    args = sys.argv[1:] if argv is None else argv

    if not args:
        print(USAGE)
        return 0

    command, rest = args[0], args[1:]

    if command in {"-V", "--version", "version"} and not rest:
        print(f"rolex {__version__}")
        return 0

    if command in {"-h", "--help", "help"} and not rest:
        print(USAGE)
        return 0

    if command in {"chat", "listen", "talk", "doctor"} and not rest:
        return _run(command)

    print(f"rolex: unrecognised arguments: {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def _run(command: str) -> int:
    """Run a command, turning our own errors into a message rather than a traceback."""
    try:
        config = load_config()
        configure_logging(config.log_level)
        if command == "chat":
            return run_chat(config)
        if command in {"listen", "talk"}:
            return run_listen(config, push_to_talk=command == "talk")
        return _doctor(config)
    except RolexError as exc:
        print(f"rolex: {exc}", file=sys.stderr)
        return 1


def _doctor(config: Config) -> int:
    """Report what is configured and whether the model can be reached."""
    provider = build_provider(config)

    print(f"provider     {config.provider} ({provider.name})")
    print(f"model        {config.model}")
    print(f"reply model  {config.drafting_model}")
    print(f"reply in     {config.reply_language}")
    print(f"ollama host  {config.ollama_host}")
    print(f"models dir   {config.ollama_models_dir}")
    print(f"destructive  {'allowed' if config.allow_destructive else 'refused'}")
    print(f"tools        {', '.join(tool.name for tool in registered()) or 'none'}")

    # A warning, not a failure: rolex never opens this directory itself. The
    # server decides where models live, and it may well be somewhere else —
    # this setting only matters for the `ollama serve` command rolex prints.
    if not config.ollama_models_dir.is_dir():
        print("             not there — fine unless you meant to use the external drive")
    print("             (the running server decides; this is only for the start command)")

    print()
    for label, detail in _voice_report(config):
        print(f"{label:12} {detail}")
    print()

    ready = provider.is_available()
    print(f"reachable    {'yes' if ready else 'no'}")
    if ready:
        return 0

    sys.stdout.flush()
    print(file=sys.stderr)
    run_chat_help(config, provider)
    return 1


def _voice_report(config: Config) -> list[tuple[str, str]]:
    """What voice needs, and whether it is there."""
    model = config.whisper_model
    return [
        ("ffmpeg", _found("ffmpeg", "brew install ffmpeg")),
        ("whisper", _found("whisper-cli", "brew install whisper-cpp")),
        ("streaming", _streaming(config)),
        ("say", _found("say", "not available on this system")),
        ("speech model", _model_state(model)),
        ("speaking", "on" if config.speak else "off (ROLEX_SPEAK=true to enable)"),
    ]


MIN_MODEL_BYTES = 10_000_000
"""Smallest the real weights could be. Below this it is a failed download,
which is worse than a missing file: everything reports it as present and
whisper fails on it later with something unhelpful."""


def _model_state(model: Path) -> str:
    """Whether the speech model is there, and whether it looks whole."""
    if not model.is_file():
        return f"MISSING at {model} — run: make whisper-model"

    size = model.stat().st_size
    if size < MIN_MODEL_BYTES:
        return f"TRUNCATED at {model} ({size} bytes) — delete it, then: make whisper-model"
    return f"{model} ({size / 1_000_000:.0f} MB)"


def _streaming(config: Config) -> str:
    """Whether the microphone is kept open, and whether it can be."""
    if not config.stream:
        return "off — listening in clips (ROLEX_STREAM=true to keep the mic open)"
    found = shutil.which("whisper-stream")
    return found if found else "MISSING whisper-stream — brew install whisper-cpp"


def _found(executable: str, remedy: str) -> str:
    path = shutil.which(executable)
    return path if path else f"MISSING — {remedy}"


if __name__ == "__main__":
    raise SystemExit(main())
