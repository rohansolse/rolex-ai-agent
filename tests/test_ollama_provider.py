"""The Ollama provider, against a fake transport.

No network and no server. A fake :class:`Transport` is injected, so the real
request-building and reply-parsing code is what runs — nothing private is
patched out from under the test.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from rolex.core.config import Config
from rolex.core.errors import ProviderError, UnreachableError
from rolex.llm.base import LLMProvider, Message, ToolSchema
from rolex.llm.ollama import OllamaProvider


@dataclass
class FakeTransport:
    """Returns canned payloads and records what it was asked for."""

    tags: Any = field(default_factory=lambda: {"models": []})
    reply: Any = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    fail_with: Exception | None = None

    posted: list[dict[str, Any]] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    def get_json(self, url: str, *, timeout: float) -> dict[str, Any]:
        self.urls.append(url)
        if self.fail_with:
            raise self.fail_with
        return dict(self.tags)

    def post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.urls.append(url)
        self.posted.append(body)
        if self.fail_with:
            raise self.fail_with
        return dict(self.reply)

    def post_stream(
        self, url: str, body: dict[str, Any], *, timeout: float
    ) -> Iterator[dict[str, Any]]:
        self.urls.append(url)
        self.posted.append(body)
        if self.fail_with:
            raise self.fail_with
        yield from self.chunks


def make(transport: FakeTransport, **settings: Any) -> OllamaProvider:
    return OllamaProvider(Config(**settings), transport=transport)


def test_provider_satisfies_the_protocol() -> None:
    assert isinstance(make(FakeTransport()), LLMProvider)


def test_name_includes_the_model() -> None:
    assert make(FakeTransport(), model="qwen2.5:3b").name == "ollama:qwen2.5:3b"


def test_host_trailing_slash_is_dropped() -> None:
    assert make(FakeTransport(), ollama_host="http://x:1/").host == "http://x:1"


# --- availability ---------------------------------------------------------


def test_available_when_the_model_is_installed() -> None:
    transport = FakeTransport(tags={"models": [{"model": "qwen2.5:3b"}]})
    assert make(transport, model="qwen2.5:3b").is_available()
    assert transport.urls == ["http://127.0.0.1:11434/api/tags"]


def test_unavailable_when_a_different_model_is_installed() -> None:
    transport = FakeTransport(tags={"models": [{"model": "llama3.2:3b"}]})
    assert not make(transport, model="qwen2.5:3b").is_available()


def test_unavailable_rather_than_raising_when_the_server_is_down() -> None:
    transport = FakeTransport(fail_with=UnreachableError("refused"))
    assert make(transport).is_available() is False


def test_latest_tag_also_matches_the_bare_name() -> None:
    transport = FakeTransport(tags={"models": [{"model": "qwen2.5:latest"}]})
    assert make(transport, model="qwen2.5").is_available()


def test_entries_using_the_older_name_key_are_read() -> None:
    transport = FakeTransport(tags={"models": [{"name": "qwen2.5:3b"}]})
    assert make(transport, model="qwen2.5:3b").is_available()


def test_unexpected_tags_shape_is_an_error() -> None:
    with pytest.raises(ProviderError, match="unexpected shape"):
        make(FakeTransport(tags={"models": "nonsense"})).installed_models()


def test_junk_entries_in_the_model_list_are_skipped() -> None:
    transport = FakeTransport(tags={"models": ["junk", {"model": "qwen2.5:3b"}, {}]})
    assert make(transport).installed_models() == ["qwen2.5:3b"]


# --- request building -----------------------------------------------------


def test_request_carries_model_and_messages() -> None:
    body = make(FakeTransport(), model="qwen2.5:3b").build_body(
        [Message("user", "hi")], None, stream=False
    )
    assert body == {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_tools_are_sent_in_openai_function_shape() -> None:
    tool = ToolSchema("open_app", "Open an app", {"type": "object"})
    body = make(FakeTransport()).build_body([Message("user", "hi")], [tool], stream=False)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open an app",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_no_tools_key_when_there_are_none() -> None:
    body = make(FakeTransport()).build_body([Message("user", "hi")], [], stream=False)
    assert "tools" not in body


def test_complete_posts_to_the_chat_endpoint() -> None:
    transport = FakeTransport(reply={"message": {"content": "ok"}})
    make(transport).complete([Message("user", "hi")])
    assert transport.urls == ["http://127.0.0.1:11434/api/chat"]
    assert transport.posted[0]["stream"] is False


# --- reading replies ------------------------------------------------------


def test_plain_answer_is_returned() -> None:
    transport = FakeTransport(reply={"model": "qwen2.5:3b", "message": {"content": "hello"}})
    reply = make(transport).complete([Message("user", "hi")])
    assert reply.content == "hello"
    assert reply.model == "qwen2.5:3b"
    assert not reply.wants_tool


def test_tool_calls_are_parsed() -> None:
    transport = FakeTransport(
        reply={
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "open_app", "arguments": {"a": "Safari"}}}],
            }
        }
    )
    reply = make(transport).complete([Message("user", "open safari")])
    assert reply.wants_tool
    assert reply.tool_calls[0].name == "open_app"
    assert reply.tool_calls[0].arguments == {"a": "Safari"}


def test_malformed_tool_calls_are_skipped_not_fatal() -> None:
    """A 3B model emits junk sometimes. One bad call must not kill the turn."""
    transport = FakeTransport(
        reply={
            "message": {
                "content": "",
                "tool_calls": [
                    "not a dict",
                    {"function": {"arguments": {}}},
                    {"function": {"name": "good", "arguments": "not a dict either"}},
                ],
            }
        }
    )
    reply = make(transport).complete([Message("user", "hi")])
    assert [call.name for call in reply.tool_calls] == ["good"]
    assert reply.tool_calls[0].arguments == {}


def test_reply_without_a_message_field_is_an_error() -> None:
    with pytest.raises(ProviderError, match="no message field"):
        make(FakeTransport(reply={"done": True})).complete([Message("user", "hi")])


# --- streaming ------------------------------------------------------------


def test_stream_yields_only_content_chunks() -> None:
    transport = FakeTransport(
        chunks=[
            {"message": {"content": "he"}},
            {"message": {"content": "llo"}},
            {"message": {"content": ""}},
            {"done": True},
        ]
    )
    assert "".join(make(transport).stream([Message("user", "hi")])) == "hello"


def test_stream_surfaces_a_server_error() -> None:
    transport = FakeTransport(chunks=[{"error": "model not found"}])
    with pytest.raises(ProviderError, match="model not found"):
        list(make(transport).stream([Message("user", "hi")]))


def test_stream_asks_for_streaming() -> None:
    transport = FakeTransport(chunks=[])
    list(make(transport).stream([Message("user", "hi")]))
    assert transport.posted[0]["stream"] is True


# --- the error that has a fix ---------------------------------------------


@pytest.mark.parametrize("call", ["installed_models", "complete", "stream"])
def test_a_down_server_says_how_to_start_it(call: str) -> None:
    transport = FakeTransport(fail_with=UnreachableError("connection refused"))
    provider = make(transport, ollama_models_dir="/Volumes/DEVHAN/ollama-models")

    with pytest.raises(UnreachableError) as caught:
        if call == "installed_models":
            provider.installed_models()
        elif call == "complete":
            provider.complete([Message("user", "hi")])
        else:
            list(provider.stream([Message("user", "hi")]))

    message = str(caught.value)
    assert "ollama serve" in message
    assert "/Volumes/DEVHAN/ollama-models" in message
