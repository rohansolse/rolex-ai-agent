# rolex — Architecture

## What rolex is

A voice-driven assistant that runs on the user's Mac. The user speaks a command
("rolex, browser open kar") and rolex interprets it with an LLM and performs the
action on the machine.

Design goals, in priority order:

1. **Free to run.** Default to a local LLM. No paid API required.
2. **Runs on 8 GB RAM / Apple M3.** Every default must fit that budget.
3. **Swappable LLM.** Local today, hosted tomorrow, without touching agent logic.
4. **Safe.** Anything that changes the machine goes through an explicit,
   auditable tool with a confirmation policy.

## Hardware and storage constraints

These are facts about the target machine and they shape the defaults:

| Constraint | Value | Consequence |
| --- | --- | --- |
| RAM | 8 GB | Local models capped at ~3B–4B parameters, Q4 quantized |
| Internal disk free | ~14 GB | Model weights must NOT live on the internal disk |
| External disk free | ~425 GB (`/Volumes/DEVHAN`) | `OLLAMA_MODELS` points here |
| CPU/GPU | Apple M3 | Metal acceleration via Ollama |

## Layered design

```
                   ┌───────────────────────────┐
   voice / text →  │  interface layer          │  chat, listen, talk
                   │  (src/rolex/interface/)   │
                   └─────────────┬─────────────┘
                                 │  uses rolex.voice for the microphone,
                                 │  whisper and the spoken reply
                                 │  user utterance (str)
                   ┌─────────────▼─────────────┐
                   │  agent loop               │  think → call tool → observe
                   │  (src/rolex/agent/)       │  → repeat → answer
                   └──────┬──────────────┬─────┘
                          │              │
          ┌───────────────▼───┐   ┌──────▼────────────────┐
          │ llm providers     │   │ tool registry         │
          │ (src/rolex/llm/)  │   │ (src/rolex/tools/)    │
          │  • base.Protocol  │   │  • open_app           │
          │  • ollama (local) │   │  • open_url           │
          │  • gemini (free)  │   │  • ... (later phases) │
          │  • transport      │   │                       │
          │  • factory        │   │                       │
          └───────────────────┘   └───────────────────────┘
                          │              │
                   ┌──────▼──────────────▼─────┐
                   │  config + logging         │
                   │  (src/rolex/core/)        │
                   └───────────────────────────┘
```

### Module boundaries

- `rolex.core` — config loading, logging, shared types and errors. Depends on
  nothing else in the package. Every other module may depend on it.
- `rolex.llm` — provider protocol plus one module per backend. Knows how to turn
  a message list + tool schemas into a reply. Knows nothing about *which* tools
  exist or what they do. `transport` holds the JSON-over-HTTP plumbing, apart
  from any one provider, so Gemini can reuse it and so a provider's parsing can
  be tested against a fake rather than a patched private method.
- `rolex.tools` — each tool is a function declared with `@register`; its JSON
  Schema is derived from the signature, so the description the model sees and
  the code that runs cannot drift apart. Knows nothing about the LLM. Importing
  the package is what declares every tool, so a new module has to be added to
  its `__init__`.
- `rolex.agent` — owns the loop. The only module that imports both `llm` and
  `tools`, and the one that translates a `tools.Tool` into the `llm.ToolSchema`
  a provider wants. It yields the loop's steps as events rather than printing
  them, so the CLI and, later, the voice interface can show the same work
  differently.
- `rolex.interface` — entrypoints. Owns I/O and nothing else.
- `rolex.voice` — the microphone, speech to text, text to speech, and the wake
  word. Devices rather than interface: each can fail on its own and is tested
  on its own. Depends on `core` only, so nothing about the agent leaks into it.

Dependencies point downward only. `tools` must never import `llm`, and `llm`
must never import `tools`. If a change needs that edge, the logic belongs in
`agent`.

## LLM strategy

One protocol, two implementations:

```python
class LLMProvider(Protocol):
    def complete(self, messages: list[Message], tools: list[ToolSchema]) -> Reply: ...
```

| Provider | Model | Cost | Role |
| --- | --- | --- | --- |
| `ollama` | `qwen2.5:3b` (Q4, 1.9 GB) | free, local | default |
| `gemini` | `gemini-2.0-flash` | free tier, rate-limited | fallback when local is too weak or unavailable |

Tool calling is the hard part for small local models. `qwen2.5:3b` is chosen
over `llama3.2:3b` because its tool-calling is noticeably more reliable at that
size. If accuracy proves unusable in testing, the provider switch is a config
line, not a rewrite — which is the whole reason for the protocol.

Phase 1 testing found its English good, its understanding of romanised Marathi
partial, and its Marathi output poor. The system prompt works around this by
asking for English replies. This is the clearest argument so far for finishing
the Gemini fallback.

## How the loop handles a small model

A 3B model gets things wrong in specific, recurring ways, and the loop treats
each as something to recover from rather than an error:

| What it does | What the loop does |
| --- | --- |
| Invents a tool name | Replies, as a tool result, that there is no such tool and lists the real ones |
| Calls a tool that raises | Replies with the failure text, so it can say so or try something else |
| Keeps calling tools forever | Stops after `MAX_ROUNDS` and asks once more with no tools offered |
| Sends junk in a tool call | The provider drops the malformed call and keeps the rest |
| Replies with nothing at all | The loop asks again rather than showing a blank answer |

Temperature is pinned at 0.2 rather than Ollama's default 0.8. Whether to call
a tool is a judgement that should come out the same way every time; at 0.8 it
did not, and the same question got a tool one run and not the next.

Nothing here raises. A turn ends with an answer or with a provider failure —
never with a traceback caused by the model being bad at its job.

## Safety model

Every tool declares a safety class when it is registered:

- **read_only** — looks at something and changes nothing. Runs unasked.
- **mutating** — makes a visible change the user would not mind undoing.
  Asks first.
- **destructive** — deletes, overwrites or sends something outward. Refused
  outright unless `allow_destructive` is set.

`rolex.agent.consent.Policy` decides, and `rolex.agent.loop` enforces it. A
tool cannot bypass it, because a tool is never the thing that calls itself.

Two details that matter more than they look:

- **With no way to ask, anything needing consent is refused.** A piped or
  scheduled run has nobody at the keyboard, and silence is not consent.
- **How the user is asked depends on how they are talking.** Typing at
  `rolex chat` gets a `[y/N]` prompt; speaking at `rolex listen` gets the
  question spoken aloud and the answer listened for. Asking the wrong way is
  not a cosmetic problem: the first voice version printed `[y/N]` to someone
  who was speaking, and in push-to-talk the Enter meant for the next recording
  answered the prompt instead, refusing before the user could say anything.
  Anything not clearly a yes is a no.
- **A refusal is reported to the model as a tool result beginning "Not
  done:".** Without that it answers as though the thing had happened. This was
  observed: told only "the user said no", the model replied "it seems the
  browser is already open".

### What the model never gets to decide

Tool arguments reach the shell as an argv list, never as a shell string, so a
name like `Safari; rm -rf ~` stays a name. `open_url` accepts only `http` and
`https`: macOS `open` will act on `file://` and on application schemes, and
neither is something a model should be able to reach for.

## Voice

Three subprocesses, no Python audio or ML dependencies:

| Job | Tool | Why |
| --- | --- | --- |
| Record | `ffmpeg` (avfoundation) | already installed; a PortAudio binding would be the project's first compiled dependency |
| Speech to text | `whisper-cli` (whisper.cpp) | `faster-whisper` would pull in ctranslate2 — hundreds of MB onto a disk with 13 GB free |
| Speak | `say` | ships with macOS; no model, no download |

Recording stops on silence rather than after a fixed time, so a three-word
command does not make the user wait out the limit. `silencedetect` only
reports silence, it does not act on it, so ffmpeg's output is read as it
arrives and ffmpeg is sent `q` on stdin at the first silence that follows
speech — `q` rather than a signal, so it writes the wav header it has been
holding back and the file is playable.

Two things about that are easy to get wrong, and were:

- The log level must be `info`. At `error` the detector's output never appears
  and nothing can stop early, silently.
- A clip that begins with speech emits `silence_start` and **no**
  `silence_end`. A rule that waits for an end never fires.

Neither showed up in unit tests written from imagination; both appeared
immediately when run against real ffmpeg. Those tests now use ffmpeg's actual
output, copied from a run.

Each clip is deleted as soon as it is transcribed — including when Ctrl-C
lands mid-recording — because a recording of someone's room is not something
to leave in `/tmp`.

### Hearing the wake word

Whisper transcribed the same spoken "rolex" as "Rolex", "Rulics" and "relax",
so the wake word cannot be matched literally. String similarity does not
separate the cases either — "rulics" scores 0.36 against "rolex" while "rules"
scores 0.60.

What works is comparing **consonant skeletons**: drop the vowels, normalise
`x`→`ks` and `c`→`k`, collapse doubled letters. Every observed mishearing
reduces to `rlks`; "rules", "release", "rolling", "robot", "ready" and the
rest do not.

The asymmetry is deliberate. A false wake costs a few wasted seconds of
listening; a missed wake means rolex ignored the user, which is the failure
people actually resent.

## Naming tools for a small model

A 3B model matches on a tool's *name* and barely reads its description. One
`system_info` tool reporting battery, disk and memory lost every battery
question to `get_time`, in every phrasing and every ordering tried. Three
tools named `battery_level`, `disk_space` and `machine_info` got all of them
right.

So: one tool per question a user would ask, named as the answer to it. Prefer
three obvious tools to one clever one.

## Coding standards

- Python 3.12, managed with `uv`. Lockfile committed.
- `src/` layout. Package is `rolex`.
- Full type hints on every public function. `mypy --strict` must pass.
- `ruff` for lint and format. Line length 100.
- `pytest` for tests. Every module gets a test file from the phase it is born in.
- No network calls in unit tests — providers are faked at the protocol boundary.
- Secrets only in `.env` (gitignored). `.env.example` documents every key.
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
