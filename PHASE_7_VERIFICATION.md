# Phase 7 verification

Verified on 8 November 2026.

## Automated checks

`make check` passed with 745 tests. Coverage includes provider selection, native tool replay, thought signatures, credential headers, quota and authentication errors, and a chat session with the local backend unavailable.

## Live comparison

One run per provider on seven synthetic prompts with the registered tool schemas. Exact tool names and arguments determine tool-selection scores. General answers pass when nonempty and free of tool calls. Tools are never executed by the benchmark. Scores cover this command set and do not establish overall model accuracy.

| Prompt | Local seconds | Gemini seconds | Fallback seconds |
| --- | --- | --- | --- |
| What time is it? | 14.366 | 1.373 | 0.888 |
| How much battery is left? | 0.569 | 1.143 | 0.848 |
| How much disk space is free? | 0.551 | 1.014 | 0.885 |
| Open Safari. | 0.767 | 1.017 | 1.23 |
| Open https://example.com | 0.786 | 1.129 | 0.89 |
| Hello! | 0.332 | 0.931 | 1.11 |
| What is a Python list? | 2.058 | 1.165 | 1.074 |

All three providers scored 7/7. Local model was qwen2.5:3b. Cloud model was gemini-3.1-flash-lite. The first local request included cold model loading. Forced fallback used http://127.0.0.1:1 to make the local backend unavailable while leaving the existing server running.

## Live chat tool round

The normal run_chat entry point used the actual Gemini API with local inference unavailable. Gemini requested a harmless get_token tool, received TEST-42, and returned an answer containing TEST-42. The session then exited successfully. Synthetic data was used for this check.

## Model compatibility

The earlier gemini-2.5-flash default returned HTTP 404 for this new account. The API suggested gemini-3.8-flash, but that model encountered HTTP 503. gemini-3.1-flash-lite passed the live checks and is now the default cloud model.
Native response parts, including thought signatures, are preserved when replaying Gemini replies.
See [Google function-calling guidance](https://ai.google.dev/gemini-api/docs/generate-content/function-calling).

## Repeat verification

```bash
uv run python -m rolex.interface.benchmark --provider all
make check
```

Set GEMINI_API_KEY in .env. ROLEX_GEMINI_MODEL selects the cloud model. ROLEX_PROVIDER=gemini selects it directly. With ROLEX_PROVIDER=local, ROLEX_GEMINI_FALLBACK=true enables automatic cloud fallback. Fallback remains opt-in because conversation and tool context go to Google.
