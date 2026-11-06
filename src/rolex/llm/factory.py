"""Pick a provider from configuration.

The one place that maps a provider name to a class. Phase 5 adds Gemini here
and nothing else has to change.
"""

from rolex.core.config import Config
from rolex.core.errors import ConfigError
from rolex.llm.base import LLMProvider
from rolex.llm.ollama import OllamaProvider


def build_provider(config: Config) -> LLMProvider:
    """Return the backend `config` asks for."""
    if config.provider == "local":
        return OllamaProvider(config)

    if config.provider == "gemini":
        raise ConfigError(
            "the gemini provider arrives in Phase 5. Set ROLEX_PROVIDER=local for now."
        )

    raise ConfigError(f"unknown provider {config.provider!r}")
