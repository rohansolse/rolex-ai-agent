"""One command path for typed instructions and voice transcripts."""

from collections.abc import Iterable, Iterator, Sequence

from rolex.agent.applications import matches_installed, requested_application
from rolex.agent.loop import Agent, Answer, Event, ToolFinished, ToolRefused, ToolStarted
from rolex.agent.reply import Approve
from rolex.agent.workflow import Workflow
from rolex.core.config import Config
from rolex.core.errors import RolexError
from rolex.interface.handle import _remember, carry_out, recognised
from rolex.llm.base import LLMProvider
from rolex.tools.base import Tool
from rolex.tools.mac import installed_apps, resolve_app


def respond(
    agent: Agent,
    said: str,
    provider: LLMProvider,
    config: Config,
    approve: Approve | None,
) -> Iterable[Event]:
    application = requested_application(said)
    launcher = next((tool for tool in agent.tools if tool.name == "open_app"), None)
    if application is not None and launcher is not None:
        installed = installed_apps()
        if matches_installed(application, installed):
            return _remember(
                _open_application(agent, launcher, application, installed), said, agent.history
            )
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


def _open_application(
    agent: Agent, tool: Tool, name: str, installed: Sequence[str]
) -> Iterator[Event]:
    try:
        arguments = {"name": resolve_app(name, installed)}
    except RolexError as exc:
        yield ToolFinished(tool.name, str(exc), failed=True)
        yield Answer(str(exc))
        return
    decision = agent.policy.decide(tool, arguments)
    if not decision.allowed:
        yield ToolRefused(tool.name, decision.reason)
        yield Answer("Application was not opened because permission was not granted.")
        return
    yield ToolStarted(tool.name, arguments)
    try:
        result = tool.run(arguments)
    except RolexError as exc:
        yield ToolFinished(tool.name, str(exc), failed=True)
        yield Answer(str(exc))
        return
    yield ToolFinished(tool.name, result)
    yield Answer(result)
