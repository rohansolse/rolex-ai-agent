"""Gemini generateContent backend using the existing JSON transport."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from rolex.core.config import Config
from rolex.core.errors import ConfigError, HTTPProviderError, ProviderError
from rolex.llm.base import Message, Reply, ToolCall, ToolSchema
from rolex.llm.transport import Transport, UrllibTransport

HOST = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(frozen=True, slots=True)
class GeminiProvider:
    config: Config = field(repr=False)
    transport: Transport | None = None

    @property
    def name(self) -> str:
        return f"gemini:{self.config.gemini_model}"

    def _transport(self) -> Transport:
        key = self.config.require_gemini_key()
        return self.transport or UrllibTransport(headers={"x-goog-api-key": key})

    @property
    def url(self) -> str:
        return f"{HOST}/models/{quote(self.config.gemini_model, safe='')}"

    def is_available(self) -> bool:
        try:
            self._transport().get_json(self.url, timeout=2.0)
            return True
        except (ProviderError, ConfigError):
            return False

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        try:
            payload = self._transport().post_json(
                f"{self.url}:generateContent",
                self.build_body(messages, tools),
                timeout=self.config.request_timeout,
            )
        except HTTPProviderError as exc:
            if exc.status == 429:
                detail = "rate limit or quota exhausted; wait or check your Gemini quota"
            elif exc.status in (401, 403):
                detail = "authentication or access denied; check GEMINI_API_KEY and model access"
            elif exc.status == 404:
                detail = "model unavailable; check ROLEX_GEMINI_MODEL"
            else:
                detail = f"HTTP {exc.status}; try again later or check Gemini API configuration"
            raise ProviderError(f"{self.name}: {detail}") from exc
        except ProviderError as exc:
            # Keep credentials and provider response bodies out of terminal output.
            raise ProviderError(
                f"{self.name}: request failed. Check connectivity, API key, model access "
                "and Gemini quota or rate limits before trying again."
            ) from exc
        return self.parse_reply(payload)

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        # The protocol permits one chunk. generateContent also preserves tool-call
        # validation, and avoids treating SSE as Ollama's NDJSON.
        yield self.complete(messages, tools).content

    def build_body(self, messages: list[Message], tools: list[ToolSchema] | None) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                system.append({"text": message.content})
                continue
            parts: list[dict[str, Any]] = []
            if message.role == "tool":
                parts.append(
                    {
                        "functionResponse": {
                            "name": message.name,
                            "response": {"result": message.content},
                        }
                    }
                )
            else:
                if message.content:
                    parts.append({"text": message.content})
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
            if parts:
                role = "model" if message.role == "assistant" else "user"
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": role, "parts": parts})
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": self.config.temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": system}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tools
                    ]
                }
            ]
        return body

    def parse_reply(self, payload: dict[str, Any]) -> Reply:
        try:
            candidate = payload["candidates"][0]
            if candidate.get("finishReason") not in (None, "STOP"):
                raise ValueError("generation stopped")
            parts = candidate["content"]["parts"]
            if not isinstance(parts, list) or not parts:
                raise ValueError("empty response")
            text: list[str] = []
            calls: list[ToolCall] = []
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("invalid part")
                if "text" in part and not part.get("thought"):
                    if not isinstance(part["text"], str):
                        raise ValueError("invalid text")
                    text.append(part["text"])
                if "functionCall" in part:
                    call = part["functionCall"]
                    if (
                        not isinstance(call, dict)
                        or not isinstance(call.get("name"), str)
                        or not isinstance(call.get("args", {}), dict)
                    ):
                        raise ValueError("invalid function call")
                    calls.append(ToolCall(call["name"], call.get("args", {})))
            if not text and not calls:
                raise ValueError("no answer")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError(f"{self.name}: blocked, incomplete or malformed response") from exc
        return Reply("".join(text), tuple(calls), self.config.gemini_model)
