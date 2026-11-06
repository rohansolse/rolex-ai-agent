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
class Message:
    """One turn in a conversation."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        """The shape most chat APIs expect."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool the model may call, described so the model can decide to.

    Phase 1 never sends these — no tools exist yet — but the type is defined
    here so providers are written against the final signature from the start.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to run a tool."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Reply:
    """What a provider gives back for one turn."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""

    @property
    def wants_tool(self) -> bool:
        """Whether the model asked to run something rather than just answer."""
        return bool(self.tool_calls)


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
