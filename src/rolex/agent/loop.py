"""The think → act → observe loop.

A model on its own can only produce text. The loop is what lets it *do*
something: send the question along with a list of tools, and if the model asks
for one, run it, hand back the result, and ask again. Repeat until the model
answers in words rather than asking for another tool.

This is the only module that imports both :mod:`rolex.llm` and
:mod:`rolex.tools`.
"""

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rolex.agent.consent import Policy
from rolex.core.errors import ToolError
from rolex.llm.base import LLMProvider, Message, ToolCall, ToolSchema
from rolex.tools.base import Tool

log = logging.getLogger(__name__)

_EMPTY_REPLY_NUDGE = (
    "That reply was empty and no tool ran. Answer the question in plain words, "
    "or call one of the tools listed above."
)

_NOTHING_TO_SAY = "I could not work that out. Try asking in a different way."

MAX_ROUNDS = 6
"""How many tool rounds before giving up.

A small model that has misunderstood will ask for the same tool forever. Six
is enough for any sensible chain and short enough that a confused model wastes
seconds rather than minutes.
"""


# --- what the caller sees -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A tool is about to run."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool has run."""

    name: str
    result: str
    failed: bool = False


@dataclass(frozen=True, slots=True)
class ToolRefused:
    """A tool was not run, because the policy said no."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class Answer:
    """The model's reply in words. Always the last event."""

    content: str
    gave_up: bool = False
    """True when the round limit was hit rather than the model finishing."""


Event = ToolStarted | ToolFinished | ToolRefused | Answer


# --- the loop -------------------------------------------------------------


@dataclass(slots=True)
class Agent:
    """Runs one conversation, with tools."""

    provider: LLMProvider
    tools: Sequence[Tool] = ()
    policy: Policy = field(default_factory=Policy)
    max_rounds: int = MAX_ROUNDS
    history: list[Message] = field(default_factory=list)

    _by_name: dict[str, Tool] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._by_name = {tool.name: tool for tool in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ToolError("two tools share a name")

    @property
    def schemas(self) -> list[ToolSchema]:
        """The tools, in the shape a provider wants."""
        return [
            ToolSchema(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool in self.tools
        ]

    def ask(self, question: str) -> Iterator[Event]:
        """Answer `question`, running tools as the model asks for them.

        Yields each step as it happens, so a caller can show the work. The
        conversation is only committed to history once a turn completes, so an
        abandoned turn leaves nothing half-finished behind.
        """
        turn: list[Message] = [Message("user", question)]
        # Prodding the model into saying something is scaffolding: it is sent
        # with the next request and then dropped. Committed to history it
        # would read, ever after, as the user complaining about an empty
        # reply; left in place it would do the same for the rest of the turn,
        # telling the model "no tool ran" right after one did.
        nudges: list[Message] = []

        for round_number in range(self.max_rounds):
            reply = self.provider.complete(self.history + turn + nudges, self.schemas)

            answered = reply.wants_tool or bool(reply.content.strip())

            if not reply.wants_tool:
                if answered:
                    self.history.extend([*turn, reply.as_message()])
                    yield Answer(reply.content)
                    return

                # Nothing said and nothing to run. This is what is left when
                # the provider drops a malformed tool call. The empty turn is
                # not kept — replaying it only teaches the model that saying
                # nothing is acceptable — so the nudge replaces it.
                log.debug("round %d: empty reply, prompting for a real one", round_number + 1)
                nudges.append(Message("user", _EMPTY_REPLY_NUDGE))
                continue

            nudges.clear()
            turn.append(reply.as_message())

            log.debug(
                "round %d: model asked for %d tool(s)", round_number + 1, len(reply.tool_calls)
            )
            for call in reply.tool_calls:
                yield from self._invoke(call, turn)

        # Out of rounds. Ask once more with no tools offered, so the model has
        # to put what it found into words instead of reaching for another tool.
        # History is only extended once that call succeeds: committing first
        # would leave a tool request with no answer in every later request.
        final = self.provider.complete(self.history + turn + nudges, [])
        content = final.content.strip() or _NOTHING_TO_SAY
        self.history.extend([*turn, Message("assistant", content)])
        yield Answer(content, gave_up=True)

    def _invoke(self, call: ToolCall, turn: list[Message]) -> Iterator[Event]:
        """Run one tool call and append its result to the turn."""
        tool = self._by_name.get(call.name)

        if tool is None:
            # Not an exception: the model can recover from this if we tell it
            # what it got wrong, and small models get names wrong often.
            known = ", ".join(sorted(self._by_name)) or "none"
            message = f"There is no tool called '{call.name}'. Available tools: {known}."
            turn.append(Message.tool_result(call.name, message))
            yield ToolFinished(call.name, message, failed=True)
            return

        decision = self.policy.decide(tool, call.arguments)
        if not decision.allowed:
            # Told to the model as a tool result, so it explains itself to the
            # user instead of silently doing nothing or trying again.
            log.debug("refused %s: %s", tool.name, decision.reason)
            turn.append(Message.tool_result(tool.name, decision.reason))
            yield ToolRefused(tool.name, decision.reason)
            return

        yield ToolStarted(tool.name, call.arguments)

        try:
            result = tool.run(call.arguments)
            # Inside the guard: a tool returning something unserialisable is
            # the tool misbehaving, and must be handled the same way.
            text = result if isinstance(result, str) else json.dumps(result)
        # Broad on purpose: a tool misbehaving must not take the agent down.
        except Exception as exc:
            # DEBUG, not WARNING: the failure is already reported to the user
            # as a ToolFinished event and to the model as a tool result, so a
            # log line would be the same thing said a third time, in the
            # middle of the conversation.
            log.debug("tool %s failed: %s", tool.name, exc, exc_info=True)
            text = f"The tool '{tool.name}' failed: {exc}"
            turn.append(Message.tool_result(tool.name, text))
            yield ToolFinished(tool.name, text, failed=True)
            return

        turn.append(Message.tool_result(tool.name, text))
        yield ToolFinished(tool.name, text)
