"""One command path for typed instructions and voice transcripts."""

from collections.abc import Iterable

from rolex.agent.loop import Agent, Event
from rolex.agent.reply import Approve
from rolex.agent.workflow import Workflow
from rolex.core.config import Config
from rolex.interface.handle import _remember, carry_out, recognised
from rolex.llm.base import LLMProvider


def respond(
    agent: Agent,
    said: str,
    provider: LLMProvider,
    config: Config,
    approve: Approve | None,
) -> Iterable[Event]:
    match = recognised(said, agent.history)
    # Keep established tab-close handling; it observes and confirms the target.
    if match is not None and match.name == "close_tab":
        return carry_out(
            match,
            said,
            provider,
            config,
            approve,
            agent.history,
            policy=agent.policy,
        )
    workflow = Workflow(provider, agent.command_state, agent.policy, approve)
    events = workflow.process(said)
    if events is not None:
        return _remember(events, said, agent.history)
    match = recognised(said, agent.history)
    if match is not None:
        return carry_out(
            match,
            said,
            provider,
            config,
            approve,
            agent.history,
            policy=agent.policy,
        )
    return agent.ask(said)
