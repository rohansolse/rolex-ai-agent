"""Gemini requests and fallback behavior without network or API keys."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from rolex.core.config import Config, load_config
from rolex.core.errors import ConfigError, ProviderError
from rolex.llm.base import Message, Reply, ToolCall, ToolSchema
from rolex.llm.factory import build_provider
from rolex.llm.fallback import FallbackProvider
from rolex.llm.gemini import GeminiProvider
from rolex.llm.ollama import OllamaProvider
from test_ollama_provider import FakeTransport


def test_factory_requires_explicit_cloud_configuration(tmp_path: Path) -> None:
    assert isinstance(build_provider(Config()), OllamaProvider)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_provider(Config(provider="gemini"))
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        build_provider(Config(gemini_fallback=True))
    assert isinstance(
        build_provider(Config(provider="gemini", gemini_api_key="test")), GeminiProvider
    )
    config = load_config(
        root=tmp_path,
        env={
            "ROLEX_GEMINI_FALLBACK": "true",
            "ROLEX_GEMINI_MODEL": "custom",
            "GEMINI_API_KEY": "test",
        },
    )
    provider = build_provider(config)
    assert isinstance(provider, FallbackProvider)
    assert provider.secondary.name == "gemini:custom"


def test_gemini_replays_tool_round_and_system_instruction() -> None:
    provider = GeminiProvider(Config(gemini_api_key="test"))
    body = provider.build_body(
        [
            Message("system", "Be helpful"),
            Message("user", "time?"),
            Message("assistant", "", (ToolCall("get_time", {}),)),
            Message.tool_result("get_time", "5pm"),
        ],
        [ToolSchema("get_time", "Current time", {"type": "object", "properties": {}})],
    )
    assert body["systemInstruction"] == {"parts": [{"text": "Be helpful"}]}
    assert body["contents"][1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_time", "args": {}}}],
    }
    assert body["contents"][2]["parts"][0]["functionResponse"] == {
        "name": "get_time",
        "response": {"result": "5pm"},
    }
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "get_time"


def test_gemini_complete_and_single_chunk_stream() -> None:
    transport = FakeTransport(
        reply={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Hello"}]},
                }
            ]
        }
    )
    provider = GeminiProvider(Config(gemini_api_key="test"), transport)
    assert provider.is_available()
    assert provider.complete([Message("user", "hi")]).content == "Hello"
    assert list(provider.stream([Message("user", "hi")])) == ["Hello"]
    assert transport.urls[-1].endswith("/models/gemini-3.1-flash-lite:generateContent")
    assert "test" not in transport.urls[-1]
    assert "test" not in repr(provider)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"candidates": []},
        {"candidates": "bad"},
        {"candidates": [{"finishReason": "SAFETY"}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "x", "args": "bad"}}]}}]},
    ],
)
def test_bad_or_blocked_gemini_response_is_a_provider_error(payload: dict[str, object]) -> None:
    with pytest.raises(ProviderError):
        GeminiProvider(Config()).parse_reply(payload)


def test_gemini_parses_parallel_tool_calls() -> None:
    reply = GeminiProvider(Config()).parse_reply(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "a", "args": {"n": 1}}},
                            {"functionCall": {"name": "b", "args": {}}},
                        ]
                    }
                }
            ]
        }
    )
    assert reply.tool_calls == (ToolCall("a", {"n": 1}), ToolCall("b", {}))


def test_gemini_failure_hides_provider_body_and_key() -> None:
    provider = GeminiProvider(
        Config(gemini_api_key="secret"),
        FakeTransport(fail_with=ProviderError("secret provider response")),
    )
    assert not provider.is_available()
    with pytest.raises(ProviderError, match="quota") as caught:
        provider.complete([Message("user", "hi")])
    assert "secret" not in str(caught.value)


class Backend:
    name = "fake"

    def __init__(self, chunks: tuple[str, ...] = (), fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.calls = 0

    def is_available(self) -> bool:
        return not self.fail

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        self.calls += 1
        if self.fail:
            raise ProviderError("offline")
        return Reply("ok")

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        self.calls += 1
        yield from self.chunks
        if self.fail:
            raise ProviderError("offline")


def test_fallback_runs_only_after_failure() -> None:
    primary, secondary = Backend(), Backend()
    provider = FallbackProvider(primary, secondary)
    assert provider.is_available()
    assert provider.complete([]).content == "ok"
    assert secondary.calls == 0
    provider = replace(provider, primary=Backend(fail=True))
    assert provider.is_available()
    assert provider.complete([]).content == "ok"
    assert secondary.calls == 1


def test_stream_failure_after_partial_output_does_not_mix_cloud_answer() -> None:
    secondary = Backend(("cloud",))
    stream = FallbackProvider(Backend(("partial",), fail=True), secondary).stream([])
    assert next(stream) == "partial"
    with pytest.raises(ProviderError):
        next(stream)
    assert secondary.calls == 0
    assert list(FallbackProvider(Backend(fail=True), secondary).stream([])) == ["cloud"]


def test_both_providers_failing_returns_error() -> None:
    provider = FallbackProvider(Backend(fail=True), Backend(fail=True))
    assert not provider.is_available()
    with pytest.raises(ProviderError):
        provider.complete([])


@pytest.mark.parametrize(
    "status, detail",
    [
        (429, "quota exhausted"),
        (403, "access denied"),
        (404, "model unavailable"),
        (503, "HTTP 503"),
    ],
)
def test_http_failures_give_actionable_errors_without_retrying(status: int, detail: str) -> None:
    from rolex.core.errors import HTTPProviderError

    transport = FakeTransport(fail_with=HTTPProviderError("private response", status))
    provider = GeminiProvider(Config(gemini_api_key="test"), transport)
    with pytest.raises(ProviderError, match=detail) as caught:
        provider.complete([Message("user", "hi")])
    assert len(transport.posted) == 1
    assert "private response" not in str(caught.value)


def test_fallback_prompt_does_not_claim_offline_execution() -> None:
    from rolex.interface.chat import system_prompt

    assert "no internet" not in system_prompt(Config(gemini_fallback=True))


def test_chat_completes_a_tool_round_when_ollama_is_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typing import Any

    from rolex.interface.chat import run_chat
    from rolex.tools.base import Tool

    class CloudTransport(FakeTransport):
        def post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
            self.posted.append(body)
            assert body.get("generationConfig", {}).get("responseMimeType") != "application/json"
            parts = (
                [{"functionCall": {"name": "get_token", "args": {}}}]
                if len(self.posted) == 1
                else [{"text": "Token is TEST-42."}]
            )
            return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}

    cloud = CloudTransport()
    local = OllamaProvider(Config(), FakeTransport(fail_with=ProviderError("offline")))
    config = Config(gemini_api_key="test", gemini_fallback=True)
    provider = FallbackProvider(local, GeminiProvider(config, cloud))
    lines = iter(["Please get my token.", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    token = Tool(name="get_token", description="Get my token", run=lambda _: "TEST-42")

    assert run_chat(config, provider=provider, tools=[token]) == 0
    assert "Token is TEST-42." in capsys.readouterr().out
    assert len(cloud.posted) == 2
    result = cloud.posted[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert result == {"name": "get_token", "response": {"result": "TEST-42"}}


def test_signed_parts_survive_reply_history_and_replay() -> None:
    provider = GeminiProvider(Config())
    parts = [
        {
            "functionCall": {"name": "get_time", "args": {}},
            "thoughtSignature": "opaque-signature",
        }
    ]
    reply = provider.parse_reply({"candidates": [{"content": {"parts": parts}}]})
    body = provider.build_body(
        [
            Message("user", "time?"),
            reply.as_message(),
            Message.tool_result("get_time", "5pm"),
        ],
        None,
    )
    assert body["contents"][1]["parts"] == parts
    assert "provider_data" not in reply.as_message().as_dict()
