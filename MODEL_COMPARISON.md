# Local model comparison — 9 November 2026

**Selected for this installation: `qwen3.5:4b-q4_K_M`.** It scored highest in
this small Rolex-specific comparison, including the repeat. This is a measured
choice for the current integration, not a claim that it is universally best or
reliable without application guards. The private `.env` now selects this tag.

All weights are stored outside the repository in
`/Volumes/DEVHAN/ollama-models`. Ollama was started with that `OLLAMA_MODELS`
value. No paid API or new Python dependency was used.

## Method

Apple M3 Mac with 8 GiB RAM; Ollama 0.40.14. All three candidates use Q4_K_M.
Each received the same 11 synthetic intent requests and 5 general-tool requests,
including Marathi, Hindi, Romanized Marathi, compound requests, exact multiline
dictation, an action word inside a message, an explicit send and a missing count.
No browser or system tools were executed; no personal contacts were involved.

Intent cases deliberately bypass `grounded.extract`: otherwise the deterministic
route would conceal model differences. They use the production schema and prompt,
temperature 0, context 2048, output cap 256. General-tool cases use the production
chat prompt and all 23 schemas, temperature 0.2, context 4096 and output cap 256.
We score exact first-response actions/arguments without model retries. Names are
case insensitive; message bodies are exact. Clarification wording may vary.

Models run sequentially and are unloaded between candidates. Each request records
timings, token counts, load duration and Ollama's reported allocation. Granite and
Qwen3.5 were repeated, giving **80 real inference requests** in total. Some initial
downloads overlapped the first round. This is a small development sample, not a
statistical benchmark or an isolated laboratory speed test.

## Results

| Model | First round | Repeat | First median | Repeat median | Max reported model allocation |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B Q4_K_M | **11/16** | **11/16** | 3.55 s | 10.42 s | 2.94 GiB |
| Granite 4 3B | 9/16 | 9/16 | 2.08 s | 3.08 s | 2.33 GiB |
| Qwen3 4B | 8/16 | Not repeated | 3.28 s | — | 2.96 GiB |

These are medians over mixed task lengths, including model/context loads, not
promised response times. During the repeat, the OS reported 3.35 GB swap in use
and 13% memory free. Allocation above is Ollama's reported model allocation, not
total application RAM or peak system usage. Memory pressure materially affected
latency; Qwen3.5's individual repeated cases took about 4.7–19.8 seconds.

Qwen3.5 got 6/11 intent cases and 5/5 tool cases correct in each round. Granite
got 5/11 and 4/5. Qwen3 got 3/11 and 5/5. Qwen3 also emitted lengthy reasoning
text despite the production `think=false` setting, increasing tool-response
latency. Its downloaded template ends with an unconditional `<think>` prefix;
this comparison does not separate model quality from template/runtime behavior.

All candidates failed some standalone follow-ups or underspecified instructions.
Qwen3.5 still misread `char navin tabs open kara`, added unrequested steps to a
Marathi draft/send, and guessed a number for `a few more tabs`. Validation rejected
several invalid plans, but did not make these wrong responses correct. Raw and
post-validation scores were identical. Do not treat 871 passing software tests as
100% model accuracy, or assume a model change verifies real UI actions.

## Artifacts and reproduction

- [First-round results](verification/model-comparison.json): exact requests,
  expectations, outputs, failures, timings, model digests and runtime allocation.
- [Repeat results](verification/model-comparison-repeat.json).
- [Selected-model chat integration](verification/workflows-qwen35.json): controlled
  A–E fixture workflows and a real-model greeting through normal `run_chat`.
- [Comparison runner](scripts/compare_models.py).

```sh
uv run python scripts/compare_models.py
uv run python scripts/verify_workflows.py --output /private/tmp/rolex-workflows.json
```

The comparison itself executes no tools. The separate integration harness uses
controlled fixtures; known action families use no model generation, while the
greeting exercises inference. All 11 integration checks passed with the selected
model; its real-model greeting took 17.815 seconds under the observed memory
pressure. `make chat` also started with this model and exited cleanly, with memory
persistence disabled for that startup check. Live WhatsApp and microphone acceptance remain
separate prerequisites described in [RELIABILITY_VERIFICATION.md](RELIABILITY_VERIFICATION.md).
Repository `make check` passed Ruff, formatting, strict mypy and 871 tests.

## Decision

Use Qwen3.5 for the stronger observed multilingual/tool balance. Keep Granite as
a lighter, faster alternative, and retain the existing models for rollback.
The source-grounded command path, target checks, explicit consent and observed
results remain necessary with every model. Free means no local inference API fee;
normal hardware, electricity and download costs still apply.

Official model references and licenses:
[Qwen3.5](https://ollama.com/library/qwen3.5:4b-q4_K_M),
[Qwen3](https://ollama.com/library/qwen3:4b),
[Granite 4](https://ollama.com/library/granite4:3b).
