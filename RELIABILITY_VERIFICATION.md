# Natural command reliability handoff

Verification date: 8 November 2026. Live messaging acceptance remains pending.

Update, 9 November: the local model was subsequently changed from `qwen3:4b` to
`qwen3.5:4b-q4_K_M` after [measured comparison](MODEL_COMPARISON.md).
The results below preserve the original verification context.

### App launch routing correction — 9 November

The configured Qwen3.5 model reproduced `cancel` plans for both `hello open chrome`
and `i aked to open chrome`. The browser-only planner had no application-launch
action and its validator accepted unrequested cancellation. The shared processor
now routes explicit installed-application requests to the session's registered
`open_app` tool, retaining its consent policy and reporting its actual result.
Unrelated system requests skip browser planning. Cancellation requires an explicit
source instruction; text inside a dictated message cannot cancel a workflow.

Both exact inputs passed through real `make chat` with the configured provider;
the OS independently confirmed Chrome running. Application recognition tests cover
other installed names, Marathi/Hindi word order, corrections, denial, failures and
negated/questions/compound inputs. The voice consent regression uses the same
route. Full checks pass with **889 tests**. Evidence is recorded in
[verification/open-app-routing.json](verification/open-app-routing.json).

## Root causes and changes

The small model described actions without tools, reused stale counts and sometimes
contradicted correct tool results. English routing missed Marathi commands. App
activation did not establish conversation selection; name-only checks could confuse
duplicates. Dictation needed separation from executable verbs.

Chat and voice now share `interface.processor`. Supported multilingual command
families extract source slots; unfamiliar wording uses constrained provider JSON
with complete-plan validation before execution. Ordered workflows report observed
results, suspend on duplicate contacts and resume after clarification. Manual
window, tab, recipient or draft changes stop dependent input. Literal drafts retain
their exact text; sending requires explicit intent and recipient/text approval.
Unknown send outcomes block automatic resending in the same session.

The native backend uses the user's existing Chrome through Apple Events, with
window/tab IDs and conversation identity checks. Tab scope is the front Chrome
window. Draft verification includes unchanged outgoing messages. Submission needs
a new matching outgoing message after the previous message anchor.

Local `qwen3:4b` is installed and configured, with no paid provider required.
No Python dependencies were added. Supported commands need no inference; arbitrary
phrasing can still fail validation or time out. Contact search alone transliterates
Devanagari; message bodies are never translated. Multilingual Whisper base and
`ROLEX_TRANSCRIBE_LANGUAGE=auto` are configured locally.

## Upstream techniques used

- [Playwright MCP](https://github.com/microsoft/playwright-mcp): distinguish an
  existing session from an isolated browser, identify tabs explicitly and verify
  actions against structured DOM roles/snapshots. Rolex keeps its native bridge;
  Playwright is not installed.
- [Hammerspoon application source](https://github.com/Hammerspoon/hammerspoon/blob/master/extensions/application/libapplication.m):
  activation establishes app focus, requiring additional window/tab/conversation
  observation before typing. No Hammerspoon dependency.
- [Ollama tool example](https://github.com/ollama/ollama-python/blob/main/examples/tools.py):
  dispatch calls, append actual tool results and continue the conversation. Rolex
  retains this general loop; observed workflows produce their own summaries to
  prevent invented action claims.

## Results

Exact inputs, starting states, expectations, observed results, transcripts,
inference counts and timings are in [verification/workflows.json](verification/workflows.json).
Each controlled workflow runs through `interface.chat.run_chat`, used by `make chat`.

| Workflow | Expected | Observed | Environment |
| --- | --- | --- | --- |
| A | Count 3, add 4, count 7 | 7; count query does not mutate | Chat + browser fixture |
| Earlier native A | Add 2 to an existing 3-tab window | Independent native count 5; created blank tabs cleaned up | User's Chrome; early count replies timed out |
| B | Separate Marathi focus/select/dictate turns | Nisha selected; exact `मी १२ मिनिटांत पोहोचतो.`; zero sends | Chat + fixture |
| C | Mixed compound, action word in body | Exact `Do not send ₹2,750 today`; zero sends | Chat + fixture |
| C paraphrase | Fresh wording, invoice body | Exact unsent invoice text | Chat + fixture |
| D | Duplicate names, then manual same-name chat change | Choice 2 resumed once; subsequent typing blocked | Chat + fixture |
| E | Approve, deny, cancel, lost acknowledgement | One message for approved/reconciled sends; zero otherwise; no duplicate retry | Chat + injected failures |
| Voice | Transcript enters same workflow | Shared-path and language-argument regressions pass | Mock microphone/transcripts |

The supported action fixtures above use **zero model generations**. A separate
greeting exercises the real configured local model. Earlier planner runs exposed
timeouts and misinterpretation, motivating source-grounded handling. These fixture
results do not establish live WhatsApp typing or universal semantic understanding.
Native session regressions mock OS interaction for Linux CI.

Full `make check` passed Ruff, formatting, strict mypy (76 source files) and
**871 tests**. The successful run used `UV_CACHE_DIR=/private/tmp/rolex-uv-cache`
and localhost access for existing HTTP transport tests. A prior restricted run
failed those socket-bind tests because the sandbox denied localhost listening.

## Remaining live prerequisites and limits

The latest OS check returned `Accessibility: False`; Chrome currently has zero
windows. Earlier observation reached the user's signed-in WhatsApp tab, but live
B–E clicking, paste and send remain unverified. No test messages went to personal
contacts. Enable **Codex and your Terminal** in **System Settings → Privacy &
Security → Accessibility**, open Chrome and sign into WhatsApp Web. Allow Chrome
JavaScript from Apple Events and macOS Automation control of Chrome/System Events.

Use a controlled test conversation. Missing unique conversation IDs stop typing,
including potentially empty chats. Search transliteration is approximate. Multiple
WhatsApp tabs retain the existing first-match reuse policy; keep the intended
account's tab open for acceptance. Clipboard restoration is best effort. Native
UI actions retain timing races, so identity is checked again immediately before
input. Uncertain-send protection is session scoped, not a persistent send journal.

Real Whisper transcribed an offline synthesized WAV as `Rolex how many tabs are
open.` Live microphone capture, wake-word behavior and Marathi speech accuracy
remain unverified.

## Final acceptance sequence

Start Ollama if necessary, then `make chat`. Replace `TEST CONTACT` with a
controlled recipient's actual name. Approve mutation prompts after checking targets.

| Input | Expected visible result |
| --- | --- |
| `How many tabs are open?` | Actual front-window count N; no changes |
| `आणखी दोन टॅब उघड.` | Exactly two new blank tabs; verified N+2 |
| `WhatsApp वर जा.` | Existing WhatsApp tab visibly focused |
| `त्यात TEST CONTACT चॅट उघड.` | Intended chat visible; duplicates ask one numbered question |
| `त्याला लिही, मी दहा मिनिटांत पोहोचतो.` | Exact unsent draft |
| `पाठव.` then decline | Draft remains; zero outgoing messages |
| `पाठव.` then approve | Exactly one outgoing message; verified submission |

Repeat those lines by voice through `make listen` with wake word `rolex`, or
push-to-talk `make talk`. Consent is spoken in voice mode. Also try a compound
instruction with the test contact and quoted body `Please send the invoice`:
the body must remain unsent.

```sh
uv run python scripts/verify_workflows.py --output /private/tmp/rolex-workflows.json
# Also verify existing Chrome tabs, cleaning up only newly created blank tabs:
uv run python scripts/verify_workflows.py --live-tabs --output /private/tmp/rolex-live.json
```
