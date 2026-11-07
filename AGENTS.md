# Repository Guidelines

## Project Structure & Module Organization

Rolex is a local, voice-driven macOS assistant using Python 3.12 and Ollama. Source lives in `src/rolex/`:

- `core/`: configuration, logging, and errors.
- `llm/`: provider protocols, Ollama, and transport.
- `tools/`: registered browser and system actions.
- `agent/`: execution loop and consent handling.
- `voice/`: recording, transcription, speech, and wake words.
- `interface/`: CLI, text chat, and listening entry points.

Tests live in `tests/`. Consult `ARCHITECTURE.md` for dependency boundaries and `PROJECT_ROADMAP.md` for planned work. Keep model weights outside the repository; there is no dedicated application-assets directory.

## Build, Test, and Development Commands

- `make install`: synchronize dependencies with uv; handles the external-volume virtualenv symlink.
- `make ollama`: start the local model server.
- `make doctor`: check runtime prerequisites.
- `make chat`, `make listen`, `make talk`: start text chat, continuous listening, or push-to-talk.
- `make lint`: run Ruff lint and formatting checks.
- `make format`: format Python and apply fixable lint corrections.
- `make typecheck`: run strict mypy checks.
- `make test`: run pytest.
- `make check`: run all checks, matching CI.
- `uv build`: build distribution artifacts using Hatchling.

## Coding Style & Naming Conventions

Use four-space indentation, a 100-character line limit, and Ruff formatting/import ordering. Use `snake_case` for modules and functions, `PascalCase` for classes, and explicit type annotations compatible with strict mypy. Preserve package boundaries: `core` is independent, and tools and LLM providers must not import each other. Ignore macOS `._*` sidecar files.

## Testing Guidelines

Use pytest with `tests/test_<module>.py` files and descriptive `test_<behavior>` functions. Mock subprocesses, model transport, microphone access, and browser interactions so tests run on Linux CI. Run focused tests with `uv run pytest tests/test_registry.py`. Cover changed behavior, failures, and consent handling. Architecture rules are enforced by `tests/test_architecture.py`; no numerical coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use descriptive `feat: ...` subjects. Follow that prefix style with an imperative summary. Keep changes focused. In PRs, describe the behavior change, link relevant issues, and report checks performed; include terminal output or recordings when interaction changes benefit from demonstration. Run `make check` before review.

## Security & Configuration

Copy `.env.example` to `.env` for local settings; never commit secrets or model weights. Configuration precedence is defaults, `rolex.toml`, then environment. Preserve tool safety classifications and consent gates when adding actions.
