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
   voice / text →  │  interface layer          │  CLI, later push-to-talk
                   │  (src/rolex/interface/)   │
                   └─────────────┬─────────────┘
                                 │  user utterance (str)
                   ┌─────────────▼─────────────┐
                   │  agent loop               │  think → call tool → observe
                   │  (src/rolex/agent/)       │  → repeat → answer
                   └──────┬──────────────┬─────┘
                          │              │
          ┌───────────────▼───┐   ┌──────▼────────────────┐
          │ llm providers     │   │ tool registry         │
          │ (src/rolex/llm/)  │   │ (src/rolex/tools/)    │
          │  • ollama (local) │   │  • open_app           │
          │  • gemini (free)  │   │  • open_url           │
          │  • base.Provider  │   │  • ... (later phases) │
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
  exist or what they do.
- `rolex.tools` — each tool is a pure-ish function plus a JSON schema. Knows
  nothing about the LLM.
- `rolex.agent` — owns the loop. The only module that imports both `llm` and
  `tools`.
- `rolex.interface` — entrypoints (CLI now, voice later). Owns I/O and nothing else.

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
| `ollama` | `qwen2.5:3b-instruct` (Q4) | free, local | default |
| `gemini` | `gemini-2.0-flash` | free tier, rate-limited | fallback when local is too weak or unavailable |

Tool calling is the hard part for small local models. `qwen2.5:3b-instruct` is
chosen over `llama3.2:3b` because its tool-calling is noticeably more reliable at
that size. If accuracy proves unusable in testing, the provider switch is a
config line, not a rewrite — which is the whole reason for the protocol.

## Safety model

Tools are classified when they are registered:

- **read-only** — run without asking (read a file, list windows).
- **mutating** — ask before running (open an app, write a file).
- **destructive** — refuse unless the config explicitly allows it (delete, send).

The agent loop enforces this; a tool cannot bypass it.

## Coding standards

- Python 3.12, managed with `uv`. Lockfile committed.
- `src/` layout. Package is `rolex`.
- Full type hints on every public function. `mypy --strict` must pass.
- `ruff` for lint and format. Line length 100.
- `pytest` for tests. Every module gets a test file from the phase it is born in.
- No network calls in unit tests — providers are faked at the protocol boundary.
- Secrets only in `.env` (gitignored). `.env.example` documents every key.
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
