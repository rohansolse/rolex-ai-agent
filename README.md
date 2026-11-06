# rolex

A voice-driven assistant that runs on your own Mac. You say "rolex, browser open
kar" and it does it.

It runs on a **local LLM by default** — no API key, no per-token cost, no data
leaving the machine. A free-tier Google Gemini fallback is planned for when the
local model is not enough.

> **Status: Phase 0.** Scaffold and tooling only. There is no agent yet.
> See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for what lands when, and
> [ARCHITECTURE.md](ARCHITECTURE.md) for how it is put together.

## Requirements

- macOS on Apple silicon
- Python 3.12 (installed for you by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — `brew install uv`
- Ollama, from Phase 1 onward — `brew install --cask ollama`

## Setup

```bash
git clone <this repo>
cd rolex-ai-agent
make install
cp .env.example .env    # then edit it
```

## Checks

```bash
make lint        # ruff: lint + format check
make typecheck   # mypy --strict
make test        # pytest
make check       # all three, the same set CI runs
```

`make format` fixes what is fixable.

## Try it

```bash
make run
uv run rolex --version
```

In Phase 0 this only confirms the package is installed and wired up correctly.

## If the repo is on an external drive

Two things to know, both caused by the same thing — macOS writes AppleDouble
(`._name`) sidecar files on filesystems that cannot store extended attributes.

**The virtualenv has to live elsewhere.** Those sidecars corrupt wheel
installs, so `make` puts the venv at `~/.venvs/rolex` instead of `./.venv`.
`make` handles it, but a bare `uv` command does not — export the same values
first, or just use `make`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/rolex"
export UV_LINK_MODE=copy
```

**Model weights have to live on the external drive.** The internal disk is
nearly full, so `.env.example` points `OLLAMA_MODELS` at `/Volumes/DEVHAN`.
Keep it that way unless you have freed up room.

## Layout

```
src/rolex/
  core/       config, logging, errors — depends on nothing else
  llm/        LLM providers behind one protocol
  tools/      things the agent can do
  agent/      the think → act → observe loop
  interface/  CLI today, voice from Phase 4
tests/
```

The dependency rules between those packages are enforced by
[tests/test_architecture.py](tests/test_architecture.py), not just documented.
