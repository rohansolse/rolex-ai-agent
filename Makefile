# When the repo sits on an external macOS volume, its filesystem makes macOS
# write AppleDouble ("._name") sidecars, and those break wheel installation.
# In that case only, the virtualenv is kept on the internal disk and ./.venv is
# a symlink to it, so plain `uv run` works here exactly as it does anywhere
# else. Everywhere else — including Linux CI — uv's normal ./.venv is used.
EXTERNAL := $(findstring /Volumes/,$(CURDIR))
ifneq (,$(EXTERNAL))
VENV := $(HOME)/.venvs/rolex
export UV_LINK_MODE ?= copy
endif

.PHONY: install lint format typecheck test check run clean

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

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
