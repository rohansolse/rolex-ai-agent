# When the repo sits on an external macOS volume, its filesystem makes macOS
# write AppleDouble ("._name") sidecars, and those break wheel installation.
# In that case only, the virtualenv is kept on the internal disk and ./.venv is
# a symlink to it, so plain `uv run` works here exactly as it does anywhere
# else. Everywhere else — including Linux CI — uv's normal ./.venv is used.
MODELS_DIR ?= /Volumes/DEVHAN/ollama-models
OLLAMA_HOST ?= http://127.0.0.1:11434
EXTERNAL := $(findstring /Volumes/,$(CURDIR))
ifneq (,$(EXTERNAL))
VENV := $(HOME)/.venvs/rolex
export UV_LINK_MODE ?= copy
endif

.PHONY: install lint format typecheck test check run chat doctor ollama clean

install:
ifneq (,$(EXTERNAL))
	@# A real directory here means a half-installed venv from a bare `uv sync`.
	@if [ ! -L .venv ]; then rm -rf .venv ._.venv; mkdir -p "$(VENV)"; ln -s "$(VENV)" .venv; \
		echo "linked ./.venv -> $(VENV) (external volume)"; fi
endif
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

run:
	uv run rolex

chat:
	uv run rolex chat

doctor:
	uv run rolex doctor

# Start the model server with its weights on the external drive. Leave this
# running in its own terminal; `make chat` talks to it. Already running is
# success, not an error — "address already in use" tells the user nothing.
ollama:
	@if curl -sf -m 2 $(OLLAMA_HOST)/api/tags >/dev/null 2>&1; then \
		echo "Ollama is already running at $(OLLAMA_HOST). Nothing to do — run 'make chat'."; \
	else \
		echo "starting Ollama (models in $(MODELS_DIR)) — leave this running"; \
		OLLAMA_MODELS=$(MODELS_DIR) ollama serve; \
	fi

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
