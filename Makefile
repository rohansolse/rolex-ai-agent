# When the repo sits on an external macOS volume, its filesystem makes macOS
# write AppleDouble ("._name") sidecars, and those break wheel installation.
# In that case only, keep the virtualenv on the internal disk. Everywhere else
# — including Linux CI — uv's normal ./.venv is used.
ifneq (,$(findstring /Volumes/,$(CURDIR)))
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/rolex
export UV_LINK_MODE ?= copy
endif

.PHONY: install lint format typecheck test check run clean

install:
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
