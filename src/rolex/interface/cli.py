"""Command line entrypoint.

Phase 0 only proves the package is installed and wired up. The real REPL
arrives in Phase 1 and voice in Phase 4.
"""

import sys

from rolex import __version__

USAGE = """usage: rolex [--version | --help]

rolex is a voice-driven assistant that runs on a local LLM.

  -V, --version   print the version and exit
  -h, --help      print this message and exit

Phase 0 is scaffold only; there is no agent yet. The chat REPL lands in
Phase 1. See PROJECT_ROADMAP.md."""


def main(argv: list[str] | None = None) -> int:
    """Run the rolex CLI. Returns the process exit code."""
    args = sys.argv[1:] if argv is None else argv

    if not args:
        print(f"rolex {__version__} — scaffold only (Phase 0).")
        print("No agent yet. The chat REPL lands in Phase 1. Try `rolex --help`.")
        return 0

    if args[0] in {"-V", "--version", "version"} and len(args) == 1:
        print(f"rolex {__version__}")
        return 0

    if args[0] in {"-h", "--help", "help"} and len(args) == 1:
        print(USAGE)
        return 0

    print(f"rolex: unrecognised arguments: {' '.join(args)}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
