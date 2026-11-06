# rolex

A voice-driven assistant that runs on your own Mac. You say "rolex, browser open
kar" and it does it.

It runs on a **local LLM by default** — no API key, no per-token cost, no data
leaving the machine. A free-tier Google Gemini fallback is planned for when the
local model is not enough.

> **Status: Phase 4.** You can speak to it. Say "rolex, open chrome" and it
> does, then answers out loud. Everything — the language model, the speech
> recognition, the voice — runs on this Mac.
> See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for what lands when, and
> [ARCHITECTURE.md](ARCHITECTURE.md) for how it is put together.

## Requirements

- macOS on Apple silicon
- Python 3.12 (installed for you by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — `brew install uv`
- Ollama — `brew install ollama`
- For voice: `brew install whisper-cpp ffmpeg`

## Setup

```bash
git clone <this repo>
cd rolex-ai-agent
make install
cp .env.example .env    # then edit it

# Download the models, onto the external drive, once.
OLLAMA_MODELS=/Volumes/DEVHAN/ollama-models ollama pull qwen2.5:3b   # 1.9 GB
make whisper-model                                                   # 141 MB
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
make chat     # type
make listen   # speak
```

```
you   > what time is it?
       · get_time()
       ·   Friday 6 November 2026, 4:30 PM
rolex > It is Friday, 6 November 2026, at 4:30 PM.

you   > open safari
       · allow open_app(name='Safari')? [y/N] y
       · open_app(name='Safari')
       ·   Opened Safari.
rolex > Safari is open.

you   > exit
```

The `·` lines are rolex doing something rather than guessing. A model cannot
know the time, so an answer with no `get_time` line above it would be invented.

### Tools

| Tool | What it does | Asks first? |
| --- | --- | --- |
| `get_time` | The date and time | no |
| `battery_level` | Battery charge | no |
| `disk_space` | Free disk space | no |
| `machine_info` | Memory, processor, macOS version | no |
| `open_app` | Opens an application | **yes** |
| `open_url` | Opens an http(s) page | **yes** |
| `search_web` | Searches the web in the browser | **yes** |

Anything that only looks runs straight away. Anything that makes a window
appear asks first, and anything that could delete or send something is
refused outright unless `ROLEX_ALLOW_DESTRUCTIVE=true` — there are no such
tools yet, but the rule is in place before there are.

## Speaking to it

```bash
make listen
```

Say "rolex", then what you want. It answers out loud. "hello rolex" and "hey
rolex" work too.

The microphone stays open and sentences end when you stop talking, not after
a fixed number of seconds, so a long command does not get cut off halfway.

```
you say: "rolex, how much battery do I have"
heard  > how much battery do I have
       · battery_level()
       ·   Battery: 58% (discharging)
rolex  > You have 58% battery left.          ← also spoken

you say: "rolex, open chrome"
heard  > open chrome
       · Open Google Chrome?                 ← asked out loud
you say: "yes"
       ·   heard: yes
       · open_app(name='Google Chrome')
rolex  > Chrome is open.
```

When rolex needs permission it asks out loud and waits for you to say yes or
no. If it cannot make out your answer it asks once more, and then does
nothing.

If the room is noisy or it keeps mishearing the wake word:

```bash
make talk     # press Enter, then speak
```

### The first time

macOS will ask whether your terminal may use the microphone — say yes. If it
was refused once, turn it on in **System Settings → Privacy & Security →
Microphone**. Without it, every recording fails with an I/O error, and rolex
says so.

### What to expect

- An answer lands a few seconds after you stop speaking: whisper takes under a
  second to several, and the model one to eight.
- `base.en` is the smallest useful speech model, chosen because this Mac has
  8 GB and is already holding a 2 GB language model. It mishears names.
- **English only.** The speech model is the English build, and the language
  model does not handle Marathi reliably either.
- Nothing is recorded to disk beyond the moment: each clip is deleted as soon
  as it has been transcribed, and nothing is sent anywhere.

Turn the voice off with `ROLEX_SPEAK=false`, or pick another with
`ROLEX_VOICE=Daniel` (`say -v '?'` lists them).

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
ROLEX_MODEL=llama3.2:3b make chat       # one run, different model
ROLEX_TEMPERATURE=0.8 make chat         # chattier, but worse at choosing tools
ROLEX_LOG_LEVEL=DEBUG make chat         # show what the loop is doing
```

## Layout

```
src/rolex/
  core/       config, logging, errors — depends on nothing else
  llm/        LLM providers behind one protocol
  tools/      things the agent can do, and the registry that declares them
  agent/      the think → act → observe loop
  voice/      microphone, speech to text, text to speech, wake word
  interface/  chat, listen
tests/
```

The dependency rules between those packages are enforced by
[tests/test_architecture.py](tests/test_architecture.py), not just documented.
