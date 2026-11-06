"""The agent loop, with a scripted model.

The provider is a fake that replies from a list, so the tests are about the
loop's decisions — when it runs a tool, what it tells the model afterwards,
and how it behaves when things go wrong — not about any real model.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from rolex.agent.consent import Policy
from rolex.agent.loop import (
    Agent,
    Answer,
    Event,
    ToolFinished,
    ToolRefused,
    ToolStarted,
)
from rolex.core.errors import ProviderError, ToolError
from rolex.llm.base import Message, Reply, ToolCall, ToolSchema
from rolex.tools.base import Tool


class ScriptedProvider:
    """Replies from a script and records what it was sent."""

    def __init__(self, replies: Sequence[Reply | Exception]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Message]] = []
        self.offered: list[list[ToolSchema]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def is_available(self) -> bool:
        return True

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        self.seen.append(list(messages))
        self.offered.append(list(tools or []))
        if not self.replies:
            raise AssertionError("the loop asked for more replies than the script has")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        yield self.complete(messages, tools).content


def says(text: str) -> Reply:
    return Reply(content=text)


def calls(name: str, **arguments: Any) -> Reply:
    return Reply(content="", tool_calls=(ToolCall(name=name, arguments=dict(arguments)),))


def tool(name: str = "echo", result: str = "done") -> Tool:
    return Tool(name=name, description=f"the {name} tool", run=lambda _: result)


def collect(events: Iterator[Event]) -> list[Event]:
    return list(events)


def answer_of(events: Sequence[Event]) -> Answer:
    last = events[-1]
    assert isinstance(last, Answer)
    return last


# --- the straightforward paths -------------------------------------------


def test_a_plain_question_needs_no_tool() -> None:
    provider = ScriptedProvider([says("hello")])
    agent = Agent(provider, tools=[tool()])

    events = collect(agent.ask("hi"))

    assert events == [Answer("hello")]
    assert provider.offered[0][0].name == "echo"


def test_a_tool_is_run_and_its_result_reaches_the_model() -> None:
    provider = ScriptedProvider([calls("echo"), says("the answer is done")])
    agent = Agent(provider, tools=[tool(result="42")])

    events = collect(agent.ask("run it"))

    assert events[0] == ToolStarted("echo", {})
    assert events[1] == ToolFinished("echo", "42")
    assert answer_of(events).content == "the answer is done"

    second_request = provider.seen[1]
    assert second_request[-1].role == "tool"
    assert second_request[-1].content == "42"
    assert second_request[-1].name == "echo"


def test_the_models_tool_request_is_replayed_to_it() -> None:
    """Without the assistant turn, the tool result answers a question the
    model cannot see it asked."""
    provider = ScriptedProvider([calls("echo", x=1), says("ok")])
    agent = Agent(provider, tools=[tool()])

    collect(agent.ask("go"))

    replayed = provider.seen[1][-2]
    assert replayed.role == "assistant"
    assert replayed.tool_calls[0].name == "echo"
    assert replayed.as_dict()["tool_calls"] == [
        {"function": {"name": "echo", "arguments": {"x": 1}}}
    ]


def test_arguments_are_passed_to_the_tool() -> None:
    seen: list[Mapping[str, Any]] = []

    def remember(args: Mapping[str, Any]) -> str:
        seen.append(args)
        return "ok"

    spy = Tool(name="echo", description="d", run=remember)
    provider = ScriptedProvider([calls("echo", city="Pune"), says("done")])

    collect(Agent(provider, tools=[spy]).ask("go"))

    assert seen == [{"city": "Pune"}]


def test_several_tool_calls_in_one_turn_all_run() -> None:
    provider = ScriptedProvider(
        [
            Reply(
                content="",
                tool_calls=(ToolCall("a", {}), ToolCall("b", {})),
            ),
            says("both done"),
        ]
    )
    agent = Agent(provider, tools=[tool("a", "A"), tool("b", "B")])

    events = collect(agent.ask("go"))

    assert [e.name for e in events if isinstance(e, ToolFinished)] == ["a", "b"]


def test_history_carries_across_turns() -> None:
    provider = ScriptedProvider([says("first"), says("second")])
    agent = Agent(provider, tools=[], history=[Message("system", "be rolex")])

    collect(agent.ask("one"))
    collect(agent.ask("two"))

    assert [m.role for m in provider.seen[1]] == ["system", "user", "assistant", "user"]
    assert provider.seen[1][2].content == "first"


# --- when the model or a tool misbehaves ---------------------------------


def test_an_unknown_tool_is_reported_back_rather_than_raised() -> None:
    """Small models invent tool names. The loop must coach, not crash."""
    provider = ScriptedProvider([calls("opne_app"), says("sorry, fixed")])
    agent = Agent(provider, tools=[tool("open_app")])

    events = collect(agent.ask("go"))

    finished = events[0]
    assert isinstance(finished, ToolFinished)
    assert finished.failed
    assert "no tool called 'opne_app'" in finished.result
    assert "open_app" in finished.result
    assert answer_of(events).content == "sorry, fixed"


def test_no_tool_started_event_for_a_tool_that_does_not_exist() -> None:
    provider = ScriptedProvider([calls("nope"), says("ok")])
    events = collect(Agent(provider, tools=[tool()]).ask("go"))
    assert not any(isinstance(e, ToolStarted) for e in events)


def test_a_tool_that_raises_does_not_kill_the_agent() -> None:
    def explode(_: Mapping[str, Any]) -> str:
        raise RuntimeError("disk on fire")

    provider = ScriptedProvider([calls("boom"), says("I could not do that")])
    agent = Agent(provider, tools=[Tool(name="boom", description="d", run=explode)])

    events = collect(agent.ask("go"))

    finished = events[1]
    assert isinstance(finished, ToolFinished)
    assert finished.failed
    assert "disk on fire" in finished.result
    assert answer_of(events).content == "I could not do that"
    assert provider.seen[1][-1].content.endswith("disk on fire")


def test_a_provider_failure_propagates() -> None:
    """The REPL catches this to keep the session alive; the loop must not
    swallow it and pretend an answer happened."""
    provider = ScriptedProvider([ProviderError("server died")])
    with pytest.raises(ProviderError, match="server died"):
        collect(Agent(provider).ask("go"))


def test_a_failed_turn_leaves_the_history_untouched() -> None:
    provider = ScriptedProvider([says("fine"), ProviderError("died"), says("ok")])
    agent = Agent(provider, history=[Message("system", "s")])

    collect(agent.ask("one"))
    before = list(agent.history)
    with pytest.raises(ProviderError):
        collect(agent.ask("two"))

    assert agent.history == before


def test_duplicate_tool_names_are_rejected_at_construction() -> None:
    with pytest.raises(ToolError, match="share a name"):
        Agent(ScriptedProvider([]), tools=[tool("same"), tool("same")])


# --- the round limit ------------------------------------------------------


def test_a_model_stuck_in_a_loop_is_stopped() -> None:
    provider = ScriptedProvider([calls("echo")] * 3 + [says("fine, here it is")])
    agent = Agent(provider, tools=[tool()], max_rounds=3)

    events = collect(agent.ask("go"))

    assert answer_of(events).gave_up
    assert answer_of(events).content == "fine, here it is"


def test_the_last_attempt_offers_no_tools() -> None:
    """Offering tools again is what kept it looping."""
    provider = ScriptedProvider([calls("echo")] * 2 + [says("done")])
    agent = Agent(provider, tools=[tool()], max_rounds=2)

    collect(agent.ask("go"))

    assert provider.offered[-1] == []


def test_a_normal_answer_is_not_marked_as_giving_up() -> None:
    provider = ScriptedProvider([says("hi")])
    assert not answer_of(collect(Agent(provider).ask("go"))).gave_up


# --- the two bugs the Phase 2 review found --------------------------------


def test_a_failure_on_the_last_attempt_leaves_no_dangling_turn() -> None:
    """Otherwise every later request carries a tool call that was never answered."""
    provider = ScriptedProvider([calls("echo"), ProviderError("died on the last word")])
    agent = Agent(provider, tools=[tool()], max_rounds=1, history=[Message("system", "s")])

    with pytest.raises(ProviderError):
        collect(agent.ask("go"))

    assert [m.role for m in agent.history] == ["system"]


def test_the_final_attempt_sees_the_tool_results() -> None:
    """Giving up must not mean throwing away what the tools found."""
    provider = ScriptedProvider([calls("echo"), says("it was 42")])
    agent = Agent(provider, tools=[tool(result="42")], max_rounds=1)

    collect(agent.ask("go"))

    final_request = provider.seen[-1]
    assert any(m.role == "tool" and m.content == "42" for m in final_request)


def test_an_empty_reply_is_not_shown_as_an_answer() -> None:
    """A provider that drops a malformed tool call leaves content="" and no
    calls. Showing that as the answer gives the user a blank line."""
    provider = ScriptedProvider([says(""), says("sorry, here it is")])
    agent = Agent(provider, tools=[tool()])

    events = collect(agent.ask("go"))

    assert answer_of(events).content == "sorry, here it is"
    assert not any(isinstance(e, Answer) and e.content == "" for e in events[:-1])


def test_an_empty_reply_is_fed_back_to_the_model() -> None:
    provider = ScriptedProvider([says("   "), says("ok")])
    collect(Agent(provider).ask("go"))

    nudge = provider.seen[1][-1]
    assert nudge.role == "user"
    assert "empty" in nudge.content


def test_an_empty_reply_does_not_reach_the_history() -> None:
    provider = ScriptedProvider([says(""), says("real answer")])
    agent = Agent(provider, history=[Message("system", "s")])

    collect(agent.ask("go"))

    assert [m.content for m in agent.history if m.role == "assistant"] == ["real answer"]


def test_a_model_that_only_ever_replies_empty_still_terminates() -> None:
    provider = ScriptedProvider([says("")] * 10)
    agent = Agent(provider, max_rounds=3)

    events = collect(agent.ask("go"))

    assert answer_of(events).gave_up


def test_a_tool_returning_something_unserialisable_is_handled() -> None:
    """Serialising the result is still the tool misbehaving, not the agent."""

    class Opaque:
        pass

    provider = ScriptedProvider([calls("weird"), says("that went badly")])
    weird = Tool(name="weird", description="d", run=lambda _: Opaque())  # type: ignore[arg-type,return-value]
    agent = Agent(provider, tools=[weird])

    events = collect(agent.ask("go"))

    finished = events[1]
    assert isinstance(finished, ToolFinished)
    assert finished.failed
    assert answer_of(events).content == "that went badly"


def test_the_nudge_is_not_kept_in_history() -> None:
    """Kept, it reads ever after as the user complaining about a blank reply."""
    provider = ScriptedProvider([says(""), says("real answer")])
    agent = Agent(provider, history=[Message("system", "s")])

    collect(agent.ask("go"))

    assert [m.content for m in agent.history] == ["s", "go", "real answer"]


def test_the_nudge_is_still_sent_in_the_same_turn() -> None:
    provider = ScriptedProvider([says(""), says("ok")])
    collect(Agent(provider).ask("go"))
    assert any("empty" in m.content for m in provider.seen[1])


def test_giving_up_with_nothing_to_say_still_says_something() -> None:
    provider = ScriptedProvider([calls("echo"), says("")])
    agent = Agent(provider, tools=[tool()], max_rounds=1)

    events = collect(agent.ask("go"))

    assert answer_of(events).gave_up
    assert answer_of(events).content.strip()
    assert agent.history[-1].content.strip()


# --- the consent policy in the loop ---------------------------------------


def test_a_refused_tool_does_not_run() -> None:
    ran = False

    def record(_: Mapping[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "did it"

    provider = ScriptedProvider([calls("risky"), says("I did not do that")])
    risky = Tool(name="risky", description="d", run=record, safety="mutating")
    agent = Agent(provider, tools=[risky], policy=Policy(ask=lambda *_: False))

    events = collect(agent.ask("do it"))

    assert not ran
    assert isinstance(events[0], ToolRefused)
    assert answer_of(events).content == "I did not do that"


def test_a_refusal_is_explained_to_the_model() -> None:
    """Otherwise it reports success for something that never happened."""
    provider = ScriptedProvider([calls("risky"), says("ok")])
    risky = Tool(name="risky", description="d", run=lambda _: "", safety="mutating")
    agent = Agent(provider, tools=[risky], policy=Policy(ask=lambda *_: False))

    collect(agent.ask("do it"))

    told = provider.seen[1][-1]
    assert told.role == "tool"
    assert told.content.startswith("Not done:")


def test_an_approved_tool_runs() -> None:
    provider = ScriptedProvider([calls("risky"), says("done")])
    risky = Tool(name="risky", description="d", run=lambda _: "did it", safety="mutating")
    agent = Agent(provider, tools=[risky], policy=Policy(ask=lambda *_: True))

    events = collect(agent.ask("do it"))

    assert any(isinstance(e, ToolFinished) and e.result == "did it" for e in events)


def test_a_read_only_tool_is_never_put_to_the_user() -> None:
    asked = False

    def ask(*_: Any) -> bool:
        nonlocal asked
        asked = True
        return True

    provider = ScriptedProvider([calls("echo"), says("done")])
    agent = Agent(provider, tools=[tool()], policy=Policy(ask=ask))

    collect(agent.ask("go"))

    assert not asked


def test_the_default_policy_refuses_mutating_tools_with_no_one_to_ask() -> None:
    """Agent() with no policy is what a scripted or piped run gets."""
    provider = ScriptedProvider([calls("risky"), says("could not")])
    risky = Tool(name="risky", description="d", run=lambda _: "", safety="mutating")

    events = collect(Agent(provider, tools=[risky]).ask("go"))

    assert isinstance(events[0], ToolRefused)


def test_the_nudge_stops_being_sent_once_the_model_speaks() -> None:
    """Left in place it tells the model "no tool ran" right after one did."""
    provider = ScriptedProvider([says(""), calls("echo"), says("done")])
    agent = Agent(provider, tools=[tool()])

    collect(agent.ask("go"))

    assert any("empty" in m.content for m in provider.seen[1])
    assert not any("empty" in m.content for m in provider.seen[2])


def test_the_nudge_is_not_sent_on_the_final_attempt() -> None:
    provider = ScriptedProvider([says(""), calls("echo"), says("done")])
    agent = Agent(provider, tools=[tool()], max_rounds=2)

    collect(agent.ask("go"))

    assert not any("empty" in m.content for m in provider.seen[-1])
