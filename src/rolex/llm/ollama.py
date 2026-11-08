"""Local LLM backend, talking to an Ollama server over HTTP.

This is rolex's default: free, offline, and nothing leaves the machine.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.llm.base import Message, Reply, ToolCall, ToolSchema
from rolex.llm.transport import Transport, UrllibTransport

#: Ollama answers /api/tags at once or not at all, so an availability check
#: does not deserve the full request timeout.
PROBE_TIMEOUT = 2.0


@dataclass(frozen=True, slots=True)
class OllamaProvider:
    """Chat through a local Ollama server."""

    config: Config
    transport: Transport = field(default_factory=UrllibTransport)

    @property
    def name(self) -> str:
        return f"ollama:{self.config.model}"

    @property
    def host(self) -> str:
        return self.config.ollama_host.rstrip("/")

    def is_available(self) -> bool:
        """True when the server is up *and* the configured model is pulled."""
        try:
            return self.config.model in self.installed_models()
        except ProviderError:
            return False

    def installed_models(self) -> list[str]:
        """Names of the models this server has on disk.

        A model tagged ``qwen2.5:latest`` is reported under both that name and
        the bare ``qwen2.5``, so either spelling in config resolves.
        """
        payload = self._get("/api/tags", timeout=PROBE_TIMEOUT)
        entries = payload.get("models", [])
        if not isinstance(entries, list):
            raise ProviderError(f"{self.host}/api/tags returned an unexpected shape")

        names: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("model") or entry.get("name")
            if isinstance(name, str):
                names.append(name)
                if name.endswith(":latest"):
                    names.append(name.removesuffix(":latest"))
        return names

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> Reply:
        payload = self._post("/api/chat", self.build_body(messages, tools, stream=False))
        return self.parse_reply(payload)

    def structured(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        """Generate schema-constrained data without executing tools."""
        import json

        body = self.build_body(messages, None, stream=False)
        body["format"] = schema
        if self.config.model.startswith("qwen3"):
            body["think"] = False
        body["options"]["temperature"] = 0
        body["options"]["num_predict"] = 256
        body["options"]["num_ctx"] = 2048
        reply = self.parse_reply(self._post("/api/chat", body))
        try:
            result: object = json.loads(reply.content)
        except json.JSONDecodeError as exc:
            raise ProviderError("local interpretation did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise ProviderError("local interpretation must return an object")
        return result

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> Iterator[str]:
        body = self.build_body(messages, tools, stream=True)
        for chunk in self._stream("/api/chat", body):
            if error := chunk.get("error"):
                raise ProviderError(f"{self.name}: {error}")
            message = chunk.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield content

    # --- shaping requests and replies -------------------------------------

    def build_body(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """The JSON Ollama's /api/chat expects."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.as_dict() for message in messages],
            "stream": stream,
            "options": {"temperature": self.config.temperature},
        }
        if self.config.model.startswith("qwen3"):
            body["think"] = False
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        return body

    def parse_reply(self, payload: dict[str, Any]) -> Reply:
        """Read one /api/chat response.

        Malformed individual tool calls are dropped rather than raised on: a 3B
        model emits junk now and then, and losing one call beats losing the turn.
        """
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ProviderError(f"{self.name}: reply had no message field")

        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ProviderError(f"{self.name}: tool_calls must be a list")
        for raw in raw_calls:
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            arguments = function.get("arguments")
            calls.append(
                ToolCall(
                    name=function["name"],
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        content = message.get("content")
        return Reply(
            content=content if isinstance(content, str) else "",
            tool_calls=tuple(calls),
            model=str(payload.get("model") or self.config.model),
        )

    # --- transport, with a fixable error message --------------------------

    def _get(self, path: str, *, timeout: float) -> dict[str, Any]:
        try:
            return self.transport.get_json(f"{self.host}{path}", timeout=timeout)
        except UnreachableError as exc:
            raise self.not_running(exc) from exc

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport.post_json(
                f"{self.host}{path}", body, timeout=self.config.request_timeout
            )
        except UnreachableError as exc:
            raise self.not_running(exc) from exc

    def _stream(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            yield from self.transport.post_stream(
                f"{self.host}{path}", body, timeout=self.config.request_timeout
            )
        except UnreachableError as exc:
            raise self.not_running(exc) from exc

    def not_running(self, cause: Exception) -> UnreachableError:
        """The 'server is down' error, carrying the command that fixes it.

        The cause already names the host and the reason, so this adds only
        the fix.
        """
        return UnreachableError(
            f"{cause}. Start Ollama with:  "
            f"OLLAMA_MODELS={self.config.ollama_models_dir} ollama serve"
        )
