# rolex — Project Roadmap

Phases ship in order. Each one is reviewed and approved before the next starts.
A phase is done when its scope is built, its tests pass, and its docs are updated.

Status legend: `done` · `in review` · `todo`

---

## Phase 0 — Scaffold and tooling · `done`

Everything needed to write code with confidence, and nothing that runs an agent.

**Scope**
- `uv` project, Python 3.12, `src/rolex/` package layout
- Empty-but-real module skeleton: `core`, `llm`, `tools`, `agent`, `interface`
- `ruff` (lint + format), `mypy --strict`, `pytest`
- `Makefile` with `install`, `lint`, `typecheck`, `test`, `check`
- GitHub Actions running `make check` on push and PR
- `.gitignore`, `.env.example`
- `ARCHITECTURE.md`, `PROJECT_ROADMAP.md`, a real `README.md`

**Done when** `make check` passes clean and CI is green.

**Out of scope** any LLM call, any tool, any voice.

---

## Phase 1 — Config and LLM provider · `done`

**Scope**
- `rolex.core.config` — layered config (defaults → `rolex.toml` → env), typed
- `rolex.core.logging` — one place that configures handlers
- `rolex.llm.base` — `Message`, `ToolSchema`, `Reply`, `LLMProvider` protocol
- `rolex.llm.transport` — JSON-over-HTTP plumbing, shared with Gemini later
- `rolex.llm.ollama` — local provider, streaming, model availability check
- `rolex.llm.factory` — config name → provider instance
- `make ollama` — starts the server with weights on the external drive
- `rolex chat` — a REPL that proves the model answers
- `rolex doctor` — says what is configured and whether it is reachable

**Done when** `rolex chat` holds a conversation with a local model, offline. ✅

**Known limitation found in testing.** `qwen2.5:3b` answers well in English and
understands simple romanised Marathi ("browser open kar"), but its Marathi
*output* is poor and it misreads some Marathi words ("guniley"). The system
prompt therefore asks it to understand any language and reply in English.
Better Marathi needs a larger model than 8 GB comfortably holds, or the Gemini
fallback in Phase 5.

---

## Phase 2 — Agent loop · `in review`

**Scope**
- `rolex.agent.loop` — think → call tool → observe → repeat, with a round cap
- `rolex.tools.base` — what a tool is, including its safety class
- `rolex.tools.clock` — `get_time`, one real tool to exercise the loop
- Error recovery: an invented tool name or a tool that raises is reported back
  to the model as a result, not raised, so it can correct itself
- `rolex chat` runs through the loop and shows each step

**Done when** the model reliably calls a tool and uses its result in the answer. ✅

Verified against the real `qwen2.5:3b`. Tool *choice* is sound: it called
`get_time` for "what time is it?" and did not call it for "what is 2+2?",
"who are you?" or "what is the capital of France?".

**Tool choice was erratic until temperature was pinned down.** In use, the
model called `get_time` for "kasa ahes" and for "are you connected to the
internet?". The cause was Ollama's default temperature of 0.8: the same
question got a tool one minute and not the next. `temperature` is now a
setting, defaulting to 0.2, and the tool description says plainly when *not*
to use it. Measured over three runs of nine prompts afterwards: 9/9 correct
each time, where before it was 8/9 and varied run to run.

**The model claimed to be online.** It now knows from its prompt that it is
not, and says so.

**Romanised Marathi is the one thing that does not work, and tuning will not
fix it.** "aaj kaay var ahe?" worked; "aata kiti vajle?", "vel kiti zala" and
"kasa ahes" did not. The model invents translations ("How old are you?") or,
when it has no idea, answers by copying the tool description back at the user.
The same questions in English are handled correctly every time.

Three attempts were made at the tool description, from terse to heavily
worded. All three scored 9/9 on tool choice and none fixed the Marathi
behaviour, because the failure is comprehension, not instruction. The shortest
was kept, since the longest was the one that got copied back.

This is a hard limit of a 3B model, and it is now the decisive argument for
the Gemini provider: Phase 4 is spoken commands, and rolex cannot reliably
understand the user's own language.

**Traded away.** Answers no longer stream in as they are typed. A reply can
only be streamed if we know in advance no tool is coming, and we do not. The
work is printed as it happens instead, which is better feedback anyway.

---

## Phase 3 — Tool registry and real tools · `todo`

**Scope**
- `rolex.tools.registry` — decorator-based registration, schema generated from
  the type hints, safety class per tool
- Confirmation policy enforced in the loop
- First real tools: `open_app`, `open_url`, `search_web`, `system_info`
- This is the phase where "rolex, browser open kar" actually works, as text

**Done when** typed commands drive the Mac through tools, with confirmations.

---

## Phase 4 — Voice in and out · `todo`

**Scope**
- Wake word "rolex" (local, always-on, cheap)
- Speech to text, local (`faster-whisper`, small model — fits the RAM budget)
- Text to speech, starting with the macOS `say` voice
- Push-to-talk fallback for when the wake word misfires

**Done when** the spoken command works end to end, no keyboard.

**Risk** 8 GB RAM holding a 3B LLM *and* a Whisper model at once. May need to
load them one at a time.

---

## Phase 5 — Gemini fallback and provider switching · `todo`

**Scope**
- `rolex.llm.gemini` — free-tier provider behind the same protocol
- Automatic fallback when the local model fails or is unavailable
- Rate-limit and quota handling
- Side-by-side accuracy comparison on a fixed command set

**Done when** rolex works with the local model off, and the switch is one config line.

---

## Phase 6 — Memory and context · `todo`

**Scope**
- Conversation persistence across restarts
- Short-term summarisation so the small model's context window is not blown
- User preferences rolex remembers

---

## Phase 7 — Hardening and release · `todo`

**Scope**
- Error handling and recovery everywhere
- Structured logs, a `--debug` trace of each loop iteration
- Install docs, troubleshooting, a demo recording
- Version 0.1.0 tagged
