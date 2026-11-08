# Rolex 0.1.0 live test

Code and package checks are ready. The following checks exercise your actual Mac,
microphone, Chrome and WhatsApp account. Use English voice commands with the current
English Whisper model.

## Start

From the project folder:

```bash
make install
make doctor
```

Review any MISSING or TRUNCATED lines. If local inference is selected, start the
server in a separate terminal:

```bash
make ollama
```

Use `make chat` for typed commands, `make talk` for push-to-talk, or
`make listen` for continuous listening.

## What to try

| Check | Instruction | Expected behavior |
| --- | --- | --- |
| Conversation | “Hello, who are you?” | Identifies itself as Rolex. |
| Read-only action | “What time is it?” | Shows get_time and answers with the current time. |
| Permission | “Open Safari” | Asks permission. Saying no leaves it unopened. Try again and approve. |
| Voice | “Rolex, how much battery is left?” | Answers on screen and aloud when speaking is enabled. |
| Repeated voice | Ask the same question three times with pauses | Responds each time. |
| Chrome | “What is on the page?” | Reads the active Chrome tab. |
| Chat search | “Open chat with NAME” | Searches WhatsApp Web for the intended person or group. |
| Draft | “Reply to this saying Friday after 5” | Places a draft in the message box for review. |
| Existing draft | Start writing a message yourself, then request another draft | Asks before replacing your text. |
| Send refusal | In a test chat, say “send it” and refuse confirmation | Leaves the message unsent. |
| Send approval | In a test chat, request a harmless message and approve | Shows recipient and words, then sends after approval. |
| Changed chat | Switch chats while a send confirmation is pending, then approve | Stops if the recipient or message differs from what you approved. |
| Memory | Tell Rolex a test project code, exit, restart and ask for the code | Recalls it with ROLEX_MEMORY=true. |
| Preferences | Run the remember command below, then restart | Uses the saved preference. |
| Recovery | With local fallback disabled, stop Ollama between typed questions, then restart it | Reports the failure and accepts the next question. |

The final send check has a small timing window between the last comparison and the
macOS Enter event. Browser interaction still targets the active tab.

```bash
uv run rolex remember "Use short answers."
```

## Gemini and fallback

To use Gemini directly, set `ROLEX_PROVIDER=gemini` in .env with your key and
`ROLEX_GEMINI_MODEL=gemini-3.1-flash-lite`.

For automatic fallback, set `ROLEX_PROVIDER=local` and
`ROLEX_GEMINI_FALLBACK=true`. Stop the local server you started with Ctrl+C and
ask a question. Rolex should log that it is using Gemini and answer. This sends
conversation context to Google.

## Capture a problem

```bash
uv run rolex --debug chat 2> /private/tmp/rolex-debug.jsonl
uv run rolex --debug listen 2> /private/tmp/rolex-voice-debug.jsonl
```

Debug records include round number, tool name, outcome and model-response latency.
Normal errors and warnings may also appear on stderr. Record the command you used,
what you expected and what happened. A screenshot or short screen recording of
your live test can accompany the report.

The recorded [CLI smoke demo](demo/cli-smoke.cast) covers version, help and invalid
arguments. Play it with `asciinema play demo/cli-smoke.cast` if asciinema is installed.
It records actual CLI output. Voice and WhatsApp acceptance are covered by your
live test.
