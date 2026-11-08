"""Construct the configured local or cloud backend."""

from rolex.core.config import Config
from rolex.core.errors import ConfigError
from rolex.llm.base import LLMProvider
from rolex.llm.fallback import FallbackProvider
from rolex.llm.gemini import GeminiProvider
from rolex.llm.ollama import OllamaProvider


def build_provider(config: Config) -> LLMProvider:
    """Build a provider, enabling cloud fallback only when requested."""
    if config.provider == "local":
        local = OllamaProvider(config)
        if config.gemini_fallback:
            config.require_gemini_key()
            return FallbackProvider(local, GeminiProvider(config))
        return local
    if config.provider == "gemini":
        config.require_gemini_key()
        return GeminiProvider(config)
    raise ConfigError(f"unknown provider {config.provider!r}")
