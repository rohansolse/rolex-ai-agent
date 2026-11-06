# rolex — Project Roadmap

Phases ship in order. Each one is reviewed and approved before the next starts.
A phase is done when its scope is built, its tests pass, and its docs are updated.

Status legend: `done` · `in review` · `todo`

---

## Phase 0 — Scaffold and tooling · `in review`

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

## Phase 1 — Config and LLM provider · `todo`

**Scope**
- `rolex.core.config` — layered config (defaults → `rolex.toml` → env), typed
- `rolex.llm.base` — `Message`, `ToolSchema`, `Reply`, `LLMProvider` protocol
- `rolex.llm.ollama` — local provider, streaming, model pull check
- Ollama bootstrap script that puts weights on `/Volumes/DEVHAN`, not the boot disk
- `rolex chat` — a bare REPL that proves the model answers

**Done when** `rolex chat` holds a conversation with a local model, offline.

---

## Phase 2 — Agent loop · `todo`

**Scope**
- `rolex.agent.loop` — think → call tool → observe → repeat, with an iteration cap
- Tool-call parsing and error recovery for a small model that formats badly
- Conversation state held in memory for the session
- One trivial built-in tool (`get_time`) purely to exercise the loop

**Done when** the model reliably calls a tool and uses its result in the answer.

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
