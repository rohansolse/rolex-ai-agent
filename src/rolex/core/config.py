"""Configuration for rolex.

Settings come from three places, each overriding the one before it:

1. the defaults in :class:`Config`
2. a ``rolex.toml`` file beside the project root, if there is one
3. environment variables, including anything in a ``.env`` file

Environment wins because it is the easiest thing to change for one run, and
because secrets belong there rather than in a committed file.
"""

import math
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

from rolex.core.errors import ConfigError

ProviderName = Literal["local", "gemini"]

REPLY_LANGUAGES = ("same", "english", "marathi")
"""What a drafted reply may be written in. ``same`` follows the conversation."""


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

    gemini_model: str = "gemini-3.1-flash-lite"
    """Cloud model used for Gemini and local fallback."""

    gemini_fallback: bool = False
    """Allow sending conversation context to Gemini when the local provider fails."""

    request_timeout: float = 120.0
    """Seconds to wait for a reply. A small model on CPU can be slow."""

    temperature: float = 0.2
    """How much the model is allowed to vary its answer.

    Deliberately far below the usual 0.8. Deciding whether to call a tool is a
    judgement the model should make the same way every time; at 0.8 the same
    question gets a tool one minute and not the next, which reads to the user
    as rolex being erratic. Creativity is not what is wanted here.
    """

    # --- voice (Phase 4) ---

    whisper_model: Path = Path("/Volumes/DEVHAN/whisper-models/ggml-base.en.bin")
    """Speech-to-text weights.

    On the external drive for the same reason as the LLM: the internal disk
    has no room. ``base.en`` is chosen over ``small`` because this machine has
    8 GB and already holds a 2 GB language model.
    """

    transcribe_language: str = "en"
    """Whisper language code. Use auto with a multilingual speech model."""

    transcribe_threads: int = 0
    """Threads whisper may use. 0 means one fewer than the machine has, so a
    long transcription does not make the rest of the system unusable."""

    stream: bool = True
    """Whether `rolex listen` keeps the microphone open rather than recording
    in fixed clips. Clips lose whatever falls across a boundary."""

    audio_device: str = ""
    """Which microphone for clip recording, as ffmpeg's avfoundation index.
    Empty means the system default. List them with:
    ``ffmpeg -f avfoundation -list_devices true -i ""``"""

    stream_device: str = ""
    """Which microphone for continuous listening, as whisper-stream's SDL
    capture index.

    A separate setting because the two number their devices differently:
    ffmpeg's avfoundation index and SDL's capture index are not the same
    list, so one value cannot serve both without opening the wrong
    microphone. Empty means the system default, which is almost always what
    is wanted."""

    listen_seconds: float = 8.0
    """How long to record after the wake word before giving up on a sentence."""

    silence_seconds: float = 1.2
    """How much quiet ends a sentence."""

    speak: bool = True
    """Whether rolex answers out loud as well as on screen."""

    voice: str = ""
    """A macOS voice name, as ``say -v '?'`` lists them. Empty is the system
    default."""

    reply_model: str = ""
    """Model used for drafting replies. Empty means the same as `model`.

    Worth setting to something larger. Measured on a client asking for
    wedding charges with no price anywhere in the conversation: `qwen2.5:3b`
    answered "I charge $500" — a number it invented, about to be sent to a
    client — while `qwen2.5:7b` said it did not have the pricing and asked
    what they needed. Choosing a tool is where the big model earns nothing;
    writing to a client is where it earns its seconds.
    """

    reply_language: str = "same"
    """What language drafted replies are written in.

    ``same`` follows whatever the user has been writing in that conversation,
    which is usually what they want. ``english`` or ``marathi`` force one.
    """

    allow_destructive: bool = False
    """Whether rolex may run tools that delete, overwrite or send something.

    Off by default. A model that has misunderstood should not be one
    confirmation away from doing something irreversible.
    """

    memory: bool = False
    """Persist conversations and load saved preferences when enabled."""

    memory_path: Path = Path.home() / ".local" / "share" / "rolex" / "memory.sqlite3"
    """Local SQLite memory store, outside the source checkout."""

    log_level: str = "WARNING"
    """Default quiet: INFO lines land in the middle of the conversation and
    the terminal is the user's, not a log sink. ROLEX_LOG_LEVEL=DEBUG when
    diagnosing."""

    @property
    def drafting_model(self) -> str:
        """Which model writes replies."""
        return self.reply_model or self.model

    @property
    def threads(self) -> int:
        """How many threads whisper should actually use."""
        return self.transcribe_threads or max(1, (os.cpu_count() or 2) - 1)

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


#: Spellings accepted for a boolean setting, so "1", "yes" and "true" all work.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def _as_positive_float(name: str, raw: object) -> float:
    """Read a duration or timeout, which is never zero or negative."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be positive, got {raw!r}")
    return value


def _as_threads(name: str, raw: object) -> int:
    """Read a thread count. Zero means "decide for me"."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} cannot be negative, got {raw!r}")
    return value


def _as_choice(name: str, raw: object, allowed: tuple[str, ...]) -> str:
    """Read a setting that may only be one of a few words."""
    text = str(raw).strip().casefold()
    if text not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}, got {raw!r}")
    return text


def _as_bool(name: str, raw: object) -> bool:
    """Read a boolean from a config file or an environment string."""
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


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
    "gemini_model": f"{ENV_PREFIX}GEMINI_MODEL",
    "gemini_fallback": f"{ENV_PREFIX}GEMINI_FALLBACK",
    "memory": f"{ENV_PREFIX}MEMORY",
    "memory_path": f"{ENV_PREFIX}MEMORY_PATH",
    "provider": f"{ENV_PREFIX}PROVIDER",
    "model": f"{ENV_PREFIX}MODEL",
    "log_level": f"{ENV_PREFIX}LOG_LEVEL",
    "request_timeout": f"{ENV_PREFIX}REQUEST_TIMEOUT",
    "temperature": f"{ENV_PREFIX}TEMPERATURE",
    "allow_destructive": f"{ENV_PREFIX}ALLOW_DESTRUCTIVE",
    "reply_language": f"{ENV_PREFIX}REPLY_LANGUAGE",
    "reply_model": f"{ENV_PREFIX}REPLY_MODEL",
    "whisper_model": f"{ENV_PREFIX}WHISPER_MODEL",
    "audio_device": f"{ENV_PREFIX}AUDIO_DEVICE",
    "stream_device": f"{ENV_PREFIX}STREAM_DEVICE",
    "stream": f"{ENV_PREFIX}STREAM",
    "transcribe_threads": f"{ENV_PREFIX}TRANSCRIBE_THREADS",
    "transcribe_language": f"{ENV_PREFIX}TRANSCRIBE_LANGUAGE",
    "listen_seconds": f"{ENV_PREFIX}LISTEN_SECONDS",
    "silence_seconds": f"{ENV_PREFIX}SILENCE_SECONDS",
    "speak": f"{ENV_PREFIX}SPEAK",
    "voice": f"{ENV_PREFIX}VOICE",
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
        elif field.name in {"ollama_models_dir", "memory_path"}:
            coerced[field.name] = Path(raw).expanduser()
        elif field.name == "request_timeout":
            coerced[field.name] = _as_positive_float(field.name, raw)
        elif field.name == "transcribe_language":
            coerced[field.name] = _as_choice(field.name, raw, ("en", "mr", "hi", "auto"))
        elif field.name == "transcribe_threads":
            coerced[field.name] = _as_threads(field.name, raw)
        elif field.name == "reply_language":
            coerced[field.name] = _as_choice(field.name, raw, REPLY_LANGUAGES)
        elif field.name in {"allow_destructive", "speak", "stream", "gemini_fallback", "memory"}:
            coerced[field.name] = _as_bool(field.name, raw)
        elif field.name == "whisper_model":
            coerced[field.name] = Path(raw).expanduser()
        elif field.name in {"listen_seconds", "silence_seconds"}:
            coerced[field.name] = _as_positive_float(field.name, raw)
        elif field.name == "temperature":
            try:
                coerced[field.name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"temperature must be a number, got {raw!r}") from exc
            if not 0.0 <= coerced[field.name] <= 2.0:
                raise ConfigError(f"temperature must be between 0 and 2, got {raw!r}")
        else:
            coerced[field.name] = str(raw)

    return Config(**coerced)
