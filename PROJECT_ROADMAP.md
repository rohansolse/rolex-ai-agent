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
fallback in Phase 6.

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

## Phase 4 — Voice in and out · `done`

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

**Permission has to be asked out loud, which the first version did not do.**
In a voice session rolex printed `allow open_app(name='Google Chrome')? [y/N]`
and waited at the keyboard — a question the user, who was speaking, never
answered. In push-to-talk it was worse: the Enter meant for the next recording
was swallowed by the prompt and read as a refusal, so rolex refused itself
before the user could reply. It now speaks the question ("Open Google
Chrome?"), listens for an answer, asks once more if it was unclear, and treats
anything still unclear as no.

**Two faults found in use, both measured rather than guessed.**

*It ignored people.* Whisper writes the wake word as "Rolik's" about as often
as "Rolex", and the word-splitting regex cut at the apostrophe, leaving
"rolik" — skeleton `rlk`, not `rlks`, so no match. One character in a regex.
Apostrophes now stay inside a word.

*It was deaf a quarter of the time.* The loop recorded a 3-second clip, then
closed the microphone while whisper worked on it — measured at 0.3 s to open
the device plus 0.3–1.1 s to transcribe, so roughly one second in every four
with nothing listening. Anything said in that gap was lost outright. The next
clip now starts recording before the last one is transcribed, so the
microphone is never shut. The clip captured while rolex is answering is
thrown away rather than transcribed, since it mostly contains rolex's own
voice.

Pre-starting the next clip then introduced three faults of its own, all
found by review rather than by the suite, because the fakes never opened a
device: the bare wake word and the permission question each opened a second
microphone beside the running one, and a clip whose recording failed stayed
in place so that every later turn re-raised it. Anything recording on the
main thread now drops the clip in flight first.

`say` also needed `--` before the text: an answer beginning with "- ", which
the model does produce, was read as a flag and nothing was spoken at all.

**Fixed-length clips had to go.** Recording three seconds at a time loses
whatever falls across a boundary. Measured: "rolex what is the time today"
takes 2.0 s to say, and when the speaker started 1.5 s into a clip it came
out as "what is the time" with "today" in the next clip, where it was
discarded for having no wake word. No larger model can reassemble a sentence
cut in half before it arrives.

`rolex listen` now keeps the microphone open through `whisper-stream` in
voice-activity mode, so sentences end when the speaker stops rather than when
a clock does. Verified: "hello rolex what is the time today" and "rolex open
google chrome", spoken one after the other, both arrived whole. Clip
recording stays for `rolex talk`, which only records when asked, and can be
chosen with `ROLEX_STREAM=false`.

**The wake word no longer has to be the first word.** "hello rolex", "hey
rolex" and "okay rolex" are at least as common as "rolex" alone, and
requiring the very first word meant ignoring all of them. It may now appear
within the first three words, which still leaves "I bought a Rolex watch"
alone.

**Not verified: the microphone itself.** Recording would have captured the
user's room, so it was left for them to run. Everything downstream was tested
on synthesised speech: `say` → wav → whisper → wake word → tool → spoken
answer, which worked for all three commands tried.

**Timing, measured.** Whisper takes 0.3–6.5 s for a short clip and the model
1–8 s, so an answer lands a few seconds after the question. Both models fit in
8 GB together in practice, so the one-at-a-time loading the roadmap feared was
not needed.

---

## Phase 5 — Acting on web pages · `in review`

Opening a page and stopping there is not an assistant. This is the phase that
answers "what is the use of an agent that cannot click?".

**Scope**
- `rolex.tools.browser` — `read_page`, `page_buttons`, `click_text`,
  `scroll_page`, `type_text`, `press_key`
- Everything through `osascript`: Chrome takes JavaScript over Apple Events,
  so no driver, no debugging port, and still no Python dependency
- The system prompt tells the model it can see the page, because it was
  flatly refusing — "I can't directly interact with web pages" — while
  holding the tools to do it

**Done when** rolex can be told what to do on a page and do it. ✅

Verified on a test page in the real Chrome: `page_buttons` listed what was
there, `click_text("MBRJ self group")` clicked it and the page changed,
`type_text` put exactly the right text in the focused box, `scroll_page` and
`press_key` worked. End to end through the agent: "click on MBRJ self group"
asked permission, clicked, and the page's state changed.

**Reading is free, acting is not.** `read_page`, `page_buttons` and
`scroll_page` run unasked — looking changes nothing. `click_text`,
`type_text` and `press_key` are `mutating` and asked about every time: one
wrong click sends a message to the wrong person.

**Typing had to become one script.** Bringing Chrome forward and typing were
two separate `osascript` calls, and the window could still be coming forward
when the keys arrived — the first character landed somewhere else. They are
now one script with a pause between.

**Review found six things worth having.** Clicking took the first substring
match in DOM order, so "Ram" clicked "Rama Sharma" — a different person;
exact labels now win over prefixes, and prefixes over substrings (verified in
Chrome: "Ram" clicks Ram, "Rama" clicks Rama Sharma). Hidden elements were
offered and clickable, since a closed menu keeps its size; visibility and
opacity are now checked, by the same code used for listing so the two cannot
disagree. `type_text` passed newlines straight to the keyboard, which in a
chat box presses Return and sends half a message while reporting the whole
thing typed — it now refuses and says to use `press_key`. `press_key` became
`destructive`, since the key people ask for is Enter. `read_page` made three
round-trips and could stitch a report from two different tabs; it is now one.
And the prompt did not mention `press_key`, so the model typed messages and
never sent them.

**Known limitation: the active tab can change under us.** Every call targets
whatever tab is in front at that moment, so `read_page` and a following
`click_text` can land on different pages if the user switches tabs in
between.

**Known limitation: a 3B model finds this hard.** Reading a page and deciding
where to click is a great deal harder than telling the time. It worked on a
simple page; a real chat application has hundreds of elements and the model
will sometimes pick the wrong one. This is the strongest case yet for a
larger model.

## Phase 6 — Gemini fallback and provider switching · `todo`

**Scope**
- `rolex.llm.gemini` — free-tier provider behind the same protocol
- Automatic fallback when the local model fails or is unavailable
- Rate-limit and quota handling
- Side-by-side accuracy comparison on a fixed command set

**Done when** rolex works with the local model off, and the switch is one config line.

---

## Phase 7 — Memory and context · `todo`

**Scope**
- Conversation persistence across restarts
- Short-term summarisation so the small model's context window is not blown
- User preferences rolex remembers

---

## Phase 8 — Hardening and release · `todo`

**Scope**
- Error handling and recovery everywhere
- Structured logs, a `--debug` trace of each loop iteration
- Install docs, troubleshooting, a demo recording
- Version 0.1.0 tagged
