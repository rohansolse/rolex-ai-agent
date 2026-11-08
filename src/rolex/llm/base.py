"""The contract every LLM backend implements.

Nothing here knows about Ollama, Gemini, or any specific vendor. The agent
loop talks to this and only this, which is what makes swapping the model a
config change rather than a rewrite.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to run a tool."""

    name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """The shape to send back when replaying this turn to the model."""
        return {"function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    A tool round needs three kinds of turn, not one: the model asking to run
    something (``assistant`` carrying ``tool_calls``), the result coming back
    (``tool`` carrying ``name``), and ordinary text. All three must be
    replayable, because the next request resends the whole conversation.
    """

    role: Role
    content: str
    tool_calls: tuple["ToolCall", ...] = ()
    name: str | None = None
    """Which tool produced this. Only meaningful on a ``tool`` message."""

    provider_data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The shape most chat APIs expect."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.as_dict() for call in self.tool_calls]
        if self.name is not None:
            payload["tool_name"] = self.name
        return payload

    @classmethod
    def tool_result(cls, name: str, content: str) -> "Message":
        """What a tool returned, addressed back to the model."""
        return cls(role="tool", content=content, name=name)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool the model may call, described so the model can decide to.

    The provider-facing twin of :class:`rolex.tools.base.Tool`. Two types
    rather than one because ``tools`` must not depend on ``llm``; the agent
    translates between them.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Reply:
    """What a provider gives back for one turn."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""

    provider_data: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        """Whether the model asked to run something rather than just answer."""
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        """This reply as a conversation turn, to send back with the next request."""
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
            provider_data=self.provider_data,
        )


@runtime_checkable
class LLMProvider(Protocol):
    """What the agent loop requires of a backend."""

    @property
    def name(self) -> str:
        """Short identifier, for logs and error messages."""
        ...

    def is_available(self) -> bool:
        """Whether this backend can be used right now.

        Must not raise. A down server, a missing model or a missing key are all
        just ``False``, so a caller can fall back without catching anything.
        """
        ...

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> Reply:
        """Answer the conversation in one shot.

        Raises:
            ProviderError: the backend was unreachable or returned nonsense.
        """
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> Iterator[str]:
        """Answer the conversation as text chunks, for a live-typing effect.

        Raises:
            ProviderError: the backend was unreachable or returned nonsense.
        """
        ...
