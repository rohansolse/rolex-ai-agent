"""Try a configured cloud backend after a local provider failure."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rolex.core.errors import ProviderError
from rolex.llm.base import LLMProvider, Message, Reply, ToolSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FallbackProvider:
    primary: LLMProvider
    secondary: LLMProvider

    @property
    def name(self) -> str:
        return f"{self.primary.name} (fallback {self.secondary.name})"

    def is_available(self) -> bool:
        return self.primary.is_available() or self.secondary.is_available()

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        try:
            return self.primary.complete(messages, tools)
        except ProviderError:
            logger.warning("Local provider failed; using configured Gemini fallback.")
            return self.secondary.complete(messages, tools)

    def structured(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        primary = getattr(self.primary, "structured", None)
        secondary = getattr(self.secondary, "structured", None)
        if primary is None:
            raise ProviderError("The primary provider does not support structured interpretation.")
        try:
            result = primary(messages, schema)
        except ProviderError:
            if secondary is None:
                raise
            logger.warning("Local interpretation failed; using configured Gemini fallback.")
            result = secondary(messages, schema)
        if not isinstance(result, dict):
            raise ProviderError("Structured interpretation must return an object.")
        return result

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        emitted = False
        try:
            for chunk in self.primary.stream(messages, tools):
                emitted = True
                yield chunk
        except ProviderError:
            if emitted:
                raise
            logger.warning("Local provider failed; using configured Gemini fallback.")
            yield from self.secondary.stream(messages, tools)
