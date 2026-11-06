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

## Phase 2 — Agent loop · `done`

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

## Phase 3 — Tool registry and real tools · `done`

**Scope**
- `rolex.tools.registry` — a `Registry` class plus the shared `REGISTRY`;
  tools are declared with `@register` and their JSON Schema is derived from
  the function signature, so the two cannot drift
- `rolex.agent.consent` — the policy deciding what may run, enforced by the
  loop, not by the tools
- Real tools: `open_app`, `open_url`, `search_web`, `battery_level`,
  `disk_space`, `machine_info`
- `rolex chat` asks before anything that changes the machine

**Done when** typed commands drive the Mac through tools, with confirmations. ✅

**One combined `system_info` tool had to be split into three.** Asked "how
much battery do I have?", the model called `get_time` instead — for all six
phrasings tried, and whichever order the tools were listed in. Removing
`get_time` fixed it, which showed the problem was the model matching on names
rather than reading descriptions. Split into `battery_level`, `disk_space` and
`machine_info`, it chose correctly every time. Measured over three runs of
twelve prompts: 12/12 each time.

**The lesson for Phase 4 and later: a tool's name has to be the answer to the
user's question.** A small model matches on the name and barely reads the
description.

---

## Phase 4 — Voice in and out · `in review`

**Scope**
- `rolex.voice.record` — the microphone, via ffmpeg, stopping on silence
- `rolex.voice.transcribe` — speech to text, via whisper.cpp
- `rolex.voice.speak` — answers read aloud, via the macOS `say` voice
- `rolex.voice.wake` — recognising "rolex" in what was heard
- `rolex listen` — wake word mode; `rolex talk` — push to talk
- `rolex doctor` reports on every piece voice needs

**Done when** the spoken command works end to end, no keyboard. ✅ for the
chain; the microphone itself is for the user to confirm (see below).

**whisper.cpp was chosen over `faster-whisper`, and ffmpeg over a PortAudio
binding.** Both are already installed, both are called as subprocesses, and
the project still has zero Python runtime dependencies. `faster-whisper` would
have pulled in ctranslate2 and friends — hundreds of megabytes onto a disk
with 13 GB free.

**The wake word needed measuring, not guessing.** Whisper transcribed the same
spoken "rolex" as "Rolex", "Rulics" and "relax". A fixed list of spellings was
written first and failed on the second recording. String similarity failed in
both directions at once: "rulics" scored 0.36 against "rolex" while "rules"
scored 0.60, so no threshold separated them. Comparing *consonant skeletons*
did: all twelve observed mishearings reduce to `rlks`, and none of eighteen
ordinary sentence-opening words do.

**Stopping when the speaker stops took two attempts, and unit tests did not
catch the first.** ffmpeg's `silencedetect` only *reports* silence; it never
ends the recording. The first version passed its tests and did nothing, twice
over: the log level was `error`, so the detector's output never appeared at
all, and the stop rule waited for a `silence_end` before acting — but a clip
that begins with talking produces a `silence_start` and no `silence_end` ever.
Both were found by running it against real ffmpeg with a prepared file, not by
the suite. The tests are now written against ffmpeg's actual output, and the
fix is measured: a 14-second clip with 1.5 seconds of speech in it stopped
after 2.6 seconds instead of running the full 15.

**Not verified: the microphone itself.** Recording would have captured the
user's room, so it was left for them to run. Everything downstream was tested
on synthesised speech: `say` → wav → whisper → wake word → tool → spoken
answer, which worked for all three commands tried.

**Timing, measured.** Whisper takes 0.3–6.5 s for a short clip and the model
1–8 s, so an answer lands a few seconds after the question. Both models fit in
8 GB together in practice, so the one-at-a-time loading the roadmap feared was
not needed.

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
