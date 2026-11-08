"""Carrying out the commands rolex recognises on its own.

The router in :mod:`rolex.agent.commands` decides *what* was asked; this
does it. Kept in the interface layer because the result is something to show
the user, and because both `rolex chat` and `rolex listen` need the same
behaviour.
"""

import re
from collections.abc import Iterator
from dataclasses import replace

from rolex.agent.commands import Match, understand
from rolex.agent.consent import Policy
from rolex.agent.loop import Answer, Event, ToolFinished, ToolRefused, ToolStarted
from rolex.agent.reply import Approve, draft_reply
from rolex.core.config import Config
from rolex.core.errors import RolexError
from rolex.llm.base import LLMProvider, Message
from rolex.llm.factory import build_provider
from rolex.tools import get
from rolex.tools.chat_page import (
    message_box_text,
    open_conversation,
    open_whatsapp_chat,
    put_in_message_box,
    send_message,
)


def recognised(said: str) -> Match | None:
    """The command this is, or None to let the agent loop decide."""
    return understand(said)


def carry_out(
    match: Match,
    said: str,
    provider: LLMProvider,
    config: Config,
    approve: Approve | None = None,
    history: list[Message] | None = None,
    *,
    policy: Policy | None = None,
) -> Iterator[Event]:
    """Do what was asked, reporting the same events the agent loop would.

    Same events so the caller shows the work identically however it was
    handled, and the user never has to care which path was taken.
    """
    if match.name == "open_whatsapp":
        yield from _remember(_open_whatsapp(match, policy or Policy()), said, history)
        return

    if match.name == "reply":
        yield from _remember(_reply(match, provider, config, approve), said, history)
        return

    if match.name == "open_chat":
        yield from _remember(_open_chat(match), said, history)
        return

    if match.name in {"send", "send_to", "type_previous"}:
        yield from _remember(
            _send(match, approve, history, send_after=match.name != "type_previous"), said, history
        )
        return

    yield Answer(f"rolex recognised '{match.name}' but does not know how to do it yet.")


def _open_whatsapp(match: Match, policy: Policy) -> Iterator[Event]:
    """Execute the registered browser action through the caller's consent policy."""
    tool = get("open_whatsapp")
    arguments = {"new_tab": match.rest == "new_tab"}
    decision = policy.decide(tool, arguments)
    if not decision.allowed:
        yield ToolRefused(tool.name, decision.reason)
        yield Answer("WhatsApp was not opened because permission was not granted.")
        return
    yield ToolStarted(tool.name, arguments)
    try:
        result = tool.run(arguments)
    except RolexError as exc:
        yield ToolFinished(tool.name, str(exc), failed=True)
        yield Answer(str(exc))
        return
    text = str(result)
    yield ToolFinished(tool.name, text)
    yield Answer(text)


def _open_chat(match: Match) -> Iterator[Event]:
    """Open a WhatsApp conversation by name.

    A rule, not a judgement: the model was clicking numbered items at random
    rather than searching, and opening the wrong conversation is how a
    message reaches the wrong person.
    """
    yield ToolStarted("open_whatsapp_chat", {"name": match.rest})

    try:
        done = open_whatsapp_chat(match.rest)
    except RolexError as exc:
        yield ToolFinished("open_whatsapp_chat", str(exc), failed=True)
        yield Answer(str(exc))
        return

    yield ToolFinished("open_whatsapp_chat", done)
    yield Answer(done)


def _send(
    match: Match,
    approve: Approve | None,
    history: list[Message] | None = None,
    *,
    send_after: bool = True,
) -> Iterator[Event]:
    """Put a message in the box and, once the user agrees, send it.

    Sending is the one thing rolex does that cannot be undone: a moment
    later the words are on someone else's phone. So the user is always
    shown who it is going to and exactly what it says, and always asked —
    whatever else is configured.
    """
    step = "open_whatsapp_chat"
    try:
        if match.who:
            yield ToolStarted("open_whatsapp_chat", {"name": match.who})
            yield ToolFinished("open_whatsapp_chat", open_whatsapp_chat(match.who))

        step = "put_in_message_box"

        text = match.rest
        if not text and (not send_after or not message_box_text().strip()):
            text = _conversation_draft(history)
            if not text and not send_after:
                yield Answer("I could not identify a drafted message. Ask me to write one first.")
                return

        if text:
            # Asked about, like everywhere else: the box may hold something
            # the user started writing, and filling it types over the lot.
            existing = message_box_text().strip()
            if existing and (approve is None or not approve(existing, text)):
                yield ToolRefused("put_in_message_box", "the box already has something")
                yield Answer(
                    f"Not done. The message box already has: {existing}\n\n"
                    f"Clear it, or ask again once it is empty."
                )
                return
            yield ToolStarted("put_in_message_box", {"text": text})
            yield ToolFinished("put_in_message_box", put_in_message_box(text))

        if not send_after:
            yield Answer("Draft is in the message box. It has not been sent.")
            return

        waiting = message_box_text().strip()
        if not waiting:
            yield Answer("There is nothing in the message box to send.")
            return

        to = open_conversation()
        if not to:
            yield Answer("Could not identify the recipient. Open the chat and try again.")
            return
    except RolexError as exc:
        # Named for the step that failed, so the ToolStarted above it has a
        # matching end rather than being left open.
        yield ToolFinished(step, str(exc), failed=True)
        yield Answer(str(exc))
        return

    if approve is None or not approve(f"send:{to}", waiting):
        yield ToolRefused("send_message", "the user did not confirm")
        yield Answer(f"Not sent. It is still in the message box for {to}:\n\n{waiting}")
        return

    try:
        done = send_message(expected_recipient=to, expected_text=waiting)
    except RolexError as exc:
        yield ToolFinished("send_message", str(exc), failed=True)
        yield Answer(str(exc))
        return

    yield ToolFinished("send_message", done)
    yield Answer(done)


_DRAFT_REQUEST = re.compile(
    r"\b(?:write|compose|draft|give|make)\b.*"
    r"\b(?:msg|message|greeting|wish|wishes)\b",
    re.IGNORECASE,
)


def _conversation_draft(history: list[Message] | None) -> str:
    """Find a response to an explicit writing request, preserving its exact words."""
    recent = (history or [])[-12:]
    for index in range(len(recent) - 2, -1, -1):
        request, response = recent[index : index + 2]
        if request.role != "user":
            continue
        if _DRAFT_REQUEST.search(request.content):
            if response.role != "assistant" or response.tool_calls:
                return ""
            if any(
                message.role == "user"
                and not re.search(r"\b(?:send|type|put|paste)\b", message.content, re.IGNORECASE)
                for message in recent[index + 2 :]
            ):
                return ""
            if any(
                message.role == "assistant"
                and message.content.startswith(("Sent to ", "Draft is in the message box"))
                for message in recent[index + 2 :]
            ):
                return ""
            # Common presentation wrapper from the model: introduction, divider,
            # message, divider, closing advice. Only remove explicit dividers.
            blocks = re.split(r"(?m)^\s*---+\s*$", response.content)
            return (blocks[1] if len(blocks) == 3 else response.content).strip()
    return ""


def _remember(events: Iterator[Event], said: str, history: list[Message] | None) -> Iterator[Event]:
    """Pass events through, recording the exchange in the conversation.

    A routed command skips the agent loop, so without this neither the
    request nor the draft is in the history and a follow-up — "make it
    shorter" — has nothing to work from.
    """
    for event in events:
        if isinstance(event, Answer) and history is not None:
            history.append(Message("user", said))
            history.append(Message("assistant", event.content))
        yield event


def _writer(provider: LLMProvider, config: Config) -> tuple[LLMProvider, Config]:
    """The model that writes replies, which may not be the one that chats.

    A small model is the right choice for conversation and a poor one for
    writing to a client: measured, `qwen2.5:3b` invented a price where
    `qwen2.5:7b` asked for it. Setting `reply_model` keeps the fast one for
    everything else.
    """
    if not config.reply_model or config.reply_model == config.model:
        return provider, config

    writing = replace(config, model=config.reply_model)
    return build_provider(writing), writing


def _reply(
    match: Match,
    provider: LLMProvider,
    config: Config,
    approve: Approve | None,
) -> Iterator[Event]:
    """Read the open conversation, write a reply, put it in the message box."""
    yield ToolStarted("read_chat", {})

    try:
        # Inside the guard: building a second provider can fail on bad
        # configuration, and a traceback would end the session.
        writer, writing_config = _writer(provider, config)
        draft = draft_reply(
            writer,
            writing_config,
            instruction=match.rest,
            language=config.reply_language,
            approve=approve,
        )
    except RolexError as exc:
        yield ToolFinished("draft_reply", str(exc), failed=True)
        yield Answer(str(exc))
        return

    first = draft.conversation.splitlines()[0] if draft.conversation else "the chat"
    yield ToolFinished("read_chat", first)

    if not draft.placed:
        yield Answer(
            f"Here is a draft. It is not in the message box, because there was "
            f"already something there.\n\n{draft.text}"
        )
        return

    yield ToolFinished("put_in_message_box", draft.text)
    yield Answer(
        f"Draft is in the message box, not sent. Read it, change it if you "
        f"like, then press Enter.\n\n{draft.text}"
    )
