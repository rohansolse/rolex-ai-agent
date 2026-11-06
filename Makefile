# When the repo sits on an external macOS volume, its filesystem makes macOS
# write AppleDouble ("._name") sidecars, and those break wheel installation.
# In that case only, the virtualenv is kept on the internal disk and ./.venv is
# a symlink to it, so plain `uv run` works here exactly as it does anywhere
# else. Everywhere else — including Linux CI — uv's normal ./.venv is used.
MODELS_DIR ?= /Volumes/DEVHAN/ollama-models
WHISPER_DIR ?= /Volumes/DEVHAN/whisper-models
WHISPER_MODEL ?= ggml-base.en.bin
# Smallest the real weights could be. Anything less is a failed download,
# which is worse than no file at all because every check reports it present.
WHISPER_MIN_SIZE ?= 10M
OLLAMA_HOST ?= http://127.0.0.1:11434
EXTERNAL := $(findstring /Volumes/,$(CURDIR))
ifneq (,$(EXTERNAL))
VENV := $(HOME)/.venvs/rolex
export UV_LINK_MODE ?= copy
endif

.PHONY: install lint format typecheck test check run chat listen talk doctor ollama whisper-model clean

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

listen:
	uv run rolex listen

talk:
	uv run rolex talk

doctor:
	uv run rolex doctor

# Speech-to-text weights, onto the external drive. 141 MB, once.
# Downloaded to a temporary name and moved into place only once complete: an
# interrupted curl otherwise leaves a stub that the existence check reports as
# "already there" forever, while whisper fails on a truncated model.
whisper-model:
	@mkdir -p $(WHISPER_DIR)
	@if find "$(WHISPER_DIR)" -name "$(WHISPER_MODEL)" -size +$(WHISPER_MIN_SIZE) 2>/dev/null | grep -q .; then \
		echo "already there: $(WHISPER_DIR)/$(WHISPER_MODEL)"; \
	else \
		echo "downloading $(WHISPER_MODEL) (141 MB) to $(WHISPER_DIR)"; \
		curl -L --fail -o "$(WHISPER_DIR)/$(WHISPER_MODEL).part" \
			"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$(WHISPER_MODEL)" \
		&& mv "$(WHISPER_DIR)/$(WHISPER_MODEL).part" "$(WHISPER_DIR)/$(WHISPER_MODEL)" \
		|| { rm -f "$(WHISPER_DIR)/$(WHISPER_MODEL).part"; exit 1; }; \
	fi

# Start the model server with its weights on the external drive. Leave this
# running in its own terminal; `make chat` and `make listen` talk to it.
# Already running is success, not an error — "address already in use" tells
# the user nothing.
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
