# rolex

A voice-driven assistant that runs on your own Mac. You say "rolex, browser open
kar" and it does it.

It runs on a **local LLM by default** — no API key, no per-token cost, no data
leaving the machine. A free-tier Google Gemini fallback is planned for when the
local model is not enough.

> **Status: Phase 2.** rolex can use tools, so it does things rather than only
> talking — but it has exactly one tool so far. Real tools land in Phase 3,
> voice in Phase 4.
> See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for what lands when, and
> [ARCHITECTURE.md](ARCHITECTURE.md) for how it is put together.

## Requirements

- macOS on Apple silicon
- Python 3.12 (installed for you by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — `brew install uv`
- Ollama — `brew install ollama`

## Setup

```bash
git clone <this repo>
cd rolex-ai-agent
make install
cp .env.example .env    # then edit it

# Download the model. ~1.9 GB, onto the external drive, once.
OLLAMA_MODELS=/Volumes/DEVHAN/ollama-models ollama pull qwen2.5:3b
```

## Checks

```bash
make lint        # ruff: lint + format check
make typecheck   # mypy --strict
make test        # pytest
make check       # all three, the same set CI runs
```

`make format` fixes what is fixable.

## Talk to it

The model runs in a server that has to be up. Leave it running in one terminal:

```bash
make ollama
```

Then in another:

```bash
make doctor   # is everything in place?
make chat     # talk
```

```
you   > who are you?
rolex > I am rolex.

you   > what time is it?
       · get_time()
       ·   Friday 6 November 2026, 4:30 PM
rolex > It is Friday, 6 November 2026, at 4:30 PM.

you   > exit
```

The `·` lines are rolex doing something rather than guessing. A model cannot
know the time, so an answer with no `get_time` line above it would be invented.

### What it can and cannot do yet

It answers questions, and it can tell the time by actually looking. It cannot
open an app, search the web, or touch a file — those tools arrive in Phase 3,
and voice in Phase 4.

`qwen2.5:3b` is a small model, chosen because 8 GB of RAM will not hold a
larger one. **Ask it in English.** In testing it handled English reliably —
answering, and choosing correctly whether to use a tool, 9 times out of 9
across repeated runs. Romanised Marathi it handles unpredictably: it
understood "aaj kaay var ahe?" but not "kasa ahes" or "vel kiti zala", where
it either invents a translation or answers with something copied out of its
own instructions. That is a limit of the model's size, not of the prompt, and
the fix is the free-tier Gemini provider in Phase 5.

## If the repo is on an external drive

Two things to know, both caused by the same thing — macOS writes AppleDouble
(`._name`) sidecar files on filesystems that cannot store extended attributes.

**The virtualenv has to live elsewhere.** Those sidecars corrupt wheel
installs, so `make install` creates the venv at `~/.venvs/rolex` and points
`./.venv` at it as a symlink. Plain `uv run` then works here as it does
anywhere else. Run `make install` first; a bare `uv sync` would build a real
`./.venv` on the volume and fail partway through.

**Model weights have to live on the external drive.** The internal disk is
nearly full, so `.env.example` points `OLLAMA_MODELS` at `/Volumes/DEVHAN`.
Keep it that way unless you have freed up room.

## Configuration

Settings come from three places, each beating the one before it: the defaults
in `src/rolex/core/config.py`, then an optional `rolex.toml`, then the
environment (including `.env`). Every key is listed in
[.env.example](.env.example).

```bash
ROLEX_MODEL=llama3.2:3b make chat     # one run, different model
ROLEX_TEMPERATURE=0.8 make chat       # chattier, but worse at choosing tools
```

## Layout

```
src/rolex/
  core/       config, logging, errors — depends on nothing else
  llm/        LLM providers behind one protocol
  tools/      things the agent can do — one so far
  agent/      the think → act → observe loop
  interface/  CLI today, voice from Phase 4
tests/
```

The dependency rules between those packages are enforced by
[tests/test_architecture.py](tests/test_architecture.py), not just documented.
