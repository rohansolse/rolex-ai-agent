"""Command line entrypoint.

Voice arrives in Phase 4; until then everything is typed.
"""

import sys

from rolex import __version__
from rolex.core.config import Config, load_config
from rolex.core.errors import RolexError
from rolex.core.logging import configure_logging
from rolex.interface.chat import explain_unavailable as run_chat_help
from rolex.interface.chat import run_chat
from rolex.llm.factory import build_provider

USAGE = """usage: rolex <command>

rolex is a voice-driven assistant that runs on a local LLM.

commands:
  chat            talk to the model (no tools yet)
  doctor          check that everything rolex needs is in place

  -V, --version   print the version and exit
  -h, --help      print this message and exit

Phase 1: chat only. Tools land in Phase 3, voice in Phase 4.
See PROJECT_ROADMAP.md."""


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

    if command in {"chat", "doctor"} and not rest:
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
        return _doctor(config)
    except RolexError as exc:
        print(f"rolex: {exc}", file=sys.stderr)
        return 1


def _doctor(config: Config) -> int:
    """Report what is configured and whether the model can be reached."""
    provider = build_provider(config)

    print(f"provider     {config.provider} ({provider.name})")
    print(f"model        {config.model}")
    print(f"ollama host  {config.ollama_host}")
    print(f"models dir   {config.ollama_models_dir}")

    # A warning, not a failure: rolex never opens this directory itself, and a
    # default ~/.ollama setup works perfectly well without it.
    if not config.ollama_models_dir.is_dir():
        print("             not there — fine unless you meant to use the external drive")

    ready = provider.is_available()
    print(f"reachable    {'yes' if ready else 'no'}")
    if ready:
        return 0

    sys.stdout.flush()
    print(file=sys.stderr)
    run_chat_help(config, provider)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
