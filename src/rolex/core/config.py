"""Configuration for rolex.

Settings come from three places, each overriding the one before it:

1. the defaults in :class:`Config`
2. a ``rolex.toml`` file beside the project root, if there is one
3. environment variables, including anything in a ``.env`` file

Environment wins because it is the easiest thing to change for one run, and
because secrets belong there rather than in a committed file.
"""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

from rolex.core.errors import ConfigError

ProviderName = Literal["local", "gemini"]


def _project_root() -> Path:
    """Where ``.env`` and ``rolex.toml`` are looked for.

    The repo root when running from a source checkout. Installed into
    site-packages there is no such root, so the working directory is used —
    a config file beside wherever the user ran rolex is the sane fallback.
    """
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


#: Resolved once, at import, so every caller agrees on it.
PROJECT_ROOT = _project_root()

#: Prefix for rolex's own environment variables. ``OLLAMA_*`` keys are read
#: under their standard names instead, since Ollama itself defines them.
ENV_PREFIX = "ROLEX_"


@dataclass(frozen=True, slots=True)
class Config:
    """Everything rolex needs to know before it starts.

    Frozen, so a stray assignment cannot change behaviour halfway through a
    run. Build a changed copy with :func:`dataclasses.replace`.
    """

    provider: ProviderName = "local"
    """Which LLM backend to use."""

    model: str = "qwen2.5:3b"
    """Model name, as the chosen provider spells it."""

    ollama_host: str = "http://127.0.0.1:11434"
    """Where the local Ollama server listens."""

    ollama_models_dir: Path = Path("/Volumes/DEVHAN/ollama-models")
    """Where model weights live.

    Deliberately not the default ``~/.ollama``: the internal disk on this
    machine has very little room and the weights are gigabytes.
    """

    gemini_api_key: str | None = None
    """Free-tier Google API key. Only needed when ``provider`` is ``gemini``."""

    request_timeout: float = 120.0
    """Seconds to wait for a reply. A small model on CPU can be slow."""

    log_level: str = "INFO"

    def require_gemini_key(self) -> str:
        """Return the Gemini key, or explain what is missing."""
        if not self.gemini_api_key:
            raise ConfigError(
                "provider is 'gemini' but GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and put it in .env"
            )
        return self.gemini_api_key


def load_config(
    *,
    root: Path | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Build a :class:`Config` from the file and environment layers.

    Args:
        root: Directory to look in for ``rolex.toml`` and ``.env``. Defaults to
            the project root.
        env: Environment to read. Defaults to the real one. Tests pass their own
            rather than mutating ``os.environ``.
    """
    root = PROJECT_ROOT if root is None else root
    environ = dict(os.environ) if env is None else dict(env)

    values: dict[str, Any] = {}
    values.update(_from_toml(root / "rolex.toml"))
    values.update(_from_env({**_dotenv(root / ".env"), **environ}))

    return _build(values)


def _dotenv(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file. Absent file means no settings, not an error."""
    if not path.is_file():
        return {}

    found: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ConfigError(f"{path}:{number}: expected KEY=value, got {raw!r}")
        found[key.strip()] = _unquote(value)
    return found


def _unquote(value: str) -> str:
    """Strip quotes, and an unquoted trailing ``# comment``, from a .env value."""
    value = value.strip()
    for quote in ("'", '"'):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value.partition(" #")[0].rstrip()


def _from_toml(path: Path) -> dict[str, Any]:
    """Read the ``[rolex]`` table from a config file, if the file exists."""
    if not path.is_file():
        return {}

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    table = document.get("rolex", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [rolex] must be a table, got {type(table).__name__}")

    known = {field.name for field in fields(Config)}
    if unknown := sorted(set(table) - known):
        raise ConfigError(
            f"{path}: unknown setting(s) {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
    return dict(table)


#: Environment variable name for each setting it can come from.
_ENV_KEYS: dict[str, str] = {
    "provider": f"{ENV_PREFIX}PROVIDER",
    "model": f"{ENV_PREFIX}MODEL",
    "log_level": f"{ENV_PREFIX}LOG_LEVEL",
    "request_timeout": f"{ENV_PREFIX}REQUEST_TIMEOUT",
    "ollama_host": "OLLAMA_HOST",
    "ollama_models_dir": "OLLAMA_MODELS",
    "gemini_api_key": "GEMINI_API_KEY",
}


def _from_env(environ: dict[str, str]) -> dict[str, Any]:
    """Pick up the settings that are present in `environ`, ignoring blanks."""
    return {
        name: environ[key].strip()
        for name, key in _ENV_KEYS.items()
        if key in environ and environ[key].strip()
    }


def _build(values: dict[str, Any]) -> Config:
    """Coerce raw strings into the field types, with errors a human can act on."""
    coerced: dict[str, Any] = {}

    for field in fields(Config):
        if field.name not in values:
            continue
        raw = values[field.name]

        if field.name == "provider":
            allowed = get_args(ProviderName)
            if raw not in allowed:
                raise ConfigError(f"provider must be one of {', '.join(allowed)}, got {raw!r}")
            coerced[field.name] = raw
        elif field.name == "ollama_models_dir":
            coerced[field.name] = Path(raw).expanduser()
        elif field.name == "request_timeout":
            try:
                coerced[field.name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"request_timeout must be a number, got {raw!r}") from exc
            if coerced[field.name] <= 0:
                raise ConfigError(f"request_timeout must be positive, got {raw!r}")
        else:
            coerced[field.name] = str(raw)

    return Config(**coerced)
