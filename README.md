# rolex

A voice-driven assistant that runs on your own Mac. You say "rolex, browser open
kar" and it does it.

It runs on a **local LLM by default** — no API key, no per-token cost, no data
leaving the machine. A free-tier Google Gemini fallback is planned for when the
local model is not enough.

> **Status: Phase 6.** Say "rolex, reply to this" with a client's chat open
> and it reads the conversation, writes an answer and puts it in the message
> box for you to check. It can also act on the page — scroll, click, type.
> Everything — the language model, the speech recognition, the voice — runs
> on this Mac.
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
| `close_app` | Quits an application | **yes** |
| `open_url` | Opens an http(s) page | **yes** |
| `search_web` | Searches the web in the browser | **yes** |
| `read_chat` | Reads the open chat conversation | no |
| `put_in_message_box` | Puts a draft in the box, unsent | **yes** |
| `read_page` | Reads the page open in Chrome | no |
| `page_buttons` | Lists what can be clicked, numbered | no |
| `scroll_page` | Scrolls the page | no |
| `click_text` | Clicks something by its text | **yes** |
| `click_item` | Clicks a numbered item from `page_buttons` | **yes** |
| `list_tabs` | Lists Chrome's tabs, numbered | no |
| `close_tab` / `switch_tab` | Closes or switches to a tab | **yes** |
| `type_text` | Types into the focused box | **yes** |
| `press_key` | Presses Enter, Escape, Tab… | **yes** |

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

## Answering a client

With a conversation open in Chrome:

```
you say: "rolex, reply to this"
       · read_chat()
       ·   Conversation with Priya Deshmukh
       · put_in_message_box()
rolex  > Draft is in the message box, not sent. Read it, change it if you
         like, then press Enter.

         I don't have the exact pricing to hand — could you tell me whether
         you need the haldi covered as well? I'll send you a quote today.
```

You can steer it: *"reply saying Friday after 5"*, *"answer this saying
yes"*.

**It never sends.** The draft sits in the box until you press Enter. That is
the whole safety of it.

Two more things it will not do: type over a message you had already started
without asking, and type into a different conversation from the one it read,
if you switched tabs while it was writing.

### Which model writes

Set `ROLEX_REPLY_MODEL` to something larger than the one you chat with. It
matters more than it sounds: asked about wedding charges with no price in the
conversation, `qwen2.5:3b` replied *"I charge $500"* — a figure it invented,
about to be sent to a client. `qwen2.5:7b` said it did not have the pricing
and asked what was needed.

```bash
ROLEX_MODEL=qwen2.5:3b ROLEX_REPLY_MODEL=qwen2.5:7b make listen
```

`ROLEX_REPLY_LANGUAGE` is `same` (follow the conversation), `english` or
`marathi`.

## Acting on a web page

rolex can see the page open in Chrome and act on it:

```
you   > what is on the page?
       · read_page()
rolex > It is WhatsApp Web, with 3 unread chats.

you   > open the MBRJ group
       · allow click_text(text='MBRJ self group')? [y/N] y
       · click_text(text='MBRJ self group')
rolex > Opened MBRJ self group.
```

**Chrome needs to be told this is allowed, once:** View → Developer → **Allow
JavaScript from Apple Events**. macOS will also ask the first time, under
System Settings → Privacy & Security → Automation. rolex says so plainly if
either is missing.

Reading and scrolling happen without asking. Clicking and typing always ask —
one wrong click sends a message to the wrong person.

**Clicks are real clicks.** macOS has to allow them: System Settings →
Privacy & Security → **Accessibility**, for whatever terminal runs rolex.
Without it every click is thrown away in silence, so rolex checks first and
says so rather than reporting success.

On a simple page this works well. On a busy one like WhatsApp Web there can
be hundreds of clickable things, and text alone is ambiguous — "Rohan Solse"
appears both in the chat list and inside the open conversation. Ask for
`page_buttons` and then click by number. Each line says where the thing is —
`[list/item]` is a row in the sidebar, `[conversation/item]` is inside the
open chat — and anything that appeared since the last look is marked `*new`.

**The model is the weak part, not the tools.** `qwen2.5:3b` will sometimes
invent a web address instead of clicking what is already on screen, however
plainly its instructions say otherwise. The tools themselves were verified by
hand on WhatsApp Web: listing, clicking by number, and switching between
conversations all work.

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
