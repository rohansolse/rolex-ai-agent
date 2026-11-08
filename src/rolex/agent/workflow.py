"""Observe, resolve, confirm, execute and verify dependent browser steps."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from rolex.agent.consent import Policy
from rolex.agent.intent import CommandState, Step, interpret
from rolex.agent.loop import Answer, Event, ToolFinished, ToolRefused, ToolStarted
from rolex.agent.reply import Approve
from rolex.core.errors import ProviderError, ToolError
from rolex.llm.base import LLMProvider
from rolex.tools import session
from rolex.tools.base import Tool
from rolex.tools.session import AmbiguousContactError, Contact, SendUncertainError, Target


class Backend(Protocol):
    def observe(self) -> Target: ...
    def focus(self) -> Target: ...
    def tabs(self) -> tuple[int, int]: ...
    def open_tabs(self, count: int, window: int) -> tuple[int, int]: ...
    def select(self, name: str, expected: Target, choice: Contact | None = None) -> Target: ...
    def write(self, expected: Target, text: str) -> Target: ...
    def send(self, expected: Target) -> Target: ...


class NativeBackend:
    def __init__(self) -> None:
        self.created: list[int] = []

    def observe(self) -> Target:
        return session.observe()

    def focus(self) -> Target:
        return session.focus_whatsapp()

    def tabs(self) -> tuple[int, int]:
        return session.tab_count()

    def open_tabs(self, count: int, window: int) -> tuple[int, int]:
        return session.open_tabs(count, window, self.created)

    def select(self, name: str, expected: Target, choice: Contact | None = None) -> Target:
        return session.select_chat(name, expected, choice)

    def write(self, expected: Target, text: str) -> Target:
        return session.write_draft(expected, text)

    def send(self, expected: Target) -> Target:
        return session.send(expected)


@dataclass
class Workflow:
    provider: LLMProvider
    state: CommandState
    policy: Policy
    approve: Approve | None
    backend: Backend = field(default_factory=lambda: NativeBackend())

    def process(self, said: str) -> Iterator[Event] | None:
        if said.strip().casefold() in {
            "cancel",
            "stop",
            "never mind",
            "रद्द",
            "रद्द कर",
            "नको",
            "नको पाठवू",
            "मत भेजो",
        }:
            self.state.pending.clear()
            self.state.candidates.clear()
            return iter([Answer("Cancelled. No pending instruction will be resumed.")])
        recheck_words = {
            "please",
            "how",
            "many",
            "again",
            "total",
            "totel",
            "actual",
            "live",
            "check",
            "chek",
            "kara",
            "kiti",
            "ahe",
            "ahet",
            "it",
            "now",
            "पुन्हा",
            "तपासा",
            "पहा",
            "एकूण",
            "किती",
            "आहेत",
        }
        words = set(said.casefold().strip(" .!?").split())
        if self.state.topic == "tabs" and words and words <= recheck_words:
            return self.execute([Step("count_tabs")])
        if self.state.pending:
            choice = self._choice(said)
            if choice is not None:
                resumed = self.state.pending[:]
                self.state.pending.clear()
                self.state.candidates.clear()
                return self.execute(resumed, choice)
            correction = next(
                (
                    said[len(prefix) :].strip(" ,.")
                    for prefix in ("actually ", "instead ", "no, ", "नाही ", "नहीं ")
                    if said.casefold().startswith(prefix)
                ),
                "",
            )
            if not correction:
                return iter([Answer(self._question())])
            resumed = [Step("select_chat", recipient=correction), *self.state.pending[1:]]
            self.state.pending.clear()
            self.state.candidates.clear()
            return self.execute(resumed)
        try:
            plan = interpret(self.provider, said)
        except ProviderError as exc:
            return iter([Answer(str(exc))])
        if plan is None or not plan or all(step.action == "chat" for step in plan):
            return None
        return self.execute(plan)

    def _choice(self, said: str) -> Contact | None:
        words = {
            "first": 1,
            "second": 2,
            "पहिला": 1,
            "पहिलं": 1,
            "दुसरा": 2,
            "दुसरं": 2,
            "पहले": 1,
            "दूसरा": 2,
        }
        value = said.strip().casefold()
        number = int(value) if value.isdecimal() else words.get(value, 0)
        if 1 <= number <= len(self.state.candidates):
            return self.state.candidates[number - 1]
        hits = [
            candidate
            for candidate in self.state.candidates
            if value == candidate.label.casefold() or value == candidate.detail.casefold()
        ]
        return hits[0] if len(hits) == 1 else None

    def _question(self) -> str:
        return "Which conversation? Reply with one number:\n" + "\n".join(
            f"{i}. {candidate.label} — {candidate.detail}"
            for i, candidate in enumerate(self.state.candidates, 1)
        )

    def _allow(self, name: str, detail: str) -> bool:
        # The policy sees a description of the intended action without executing it.
        tool = Tool(name=name, description=detail, safety="mutating", run=lambda _: "")
        return self.policy.decide(tool, {"details": detail}).allowed

    def _target(self, *, draft: bool = False) -> Target:
        expected = self.state.target
        if expected is None or not expected.recipient:
            raise ToolError("Select the intended conversation before dictating or sending.")
        if not expected.conversation_id:
            raise ToolError(
                "A unique conversation identity is unavailable. Typing and sending stopped."
            )
        actual = self.backend.observe()
        if actual.identity != expected.identity:
            raise ToolError(
                "The active chat or window changed. Select the intended recipient again."
            )
        if draft and actual.draft != expected.draft:
            raise ToolError("The draft changed manually. Select the chat again and review it.")
        return actual

    def execute(self, plan: list[Step], choice: Contact | None = None) -> Iterator[Event]:
        completed: list[str] = []
        for index, step in enumerate(plan):
            try:
                if step.action == "clarify":
                    yield Answer(step.text or "Which action or target do you mean?")
                    return
                if step.action == "cancel":
                    self.state.pending.clear()
                    yield Answer("Cancelled.")
                    return
                if step.action == "chat":
                    raise ToolError("Please split the conversational question from the action.")
                if step.action == "count_tabs":
                    yield ToolStarted("list_tabs", {})
                    window, count = self.backend.tabs()
                    self.state.window = window
                    self.state.topic = "tabs"
                    text = f"Front Chrome window {window}: {count} tabs."
                elif step.action == "open_tabs":
                    window, count = self.backend.tabs()
                    if self.state.window is not None and window != self.state.window:
                        raise ToolError("Window changed. Query its count to select this scope.")
                    self.state.window = window
                    if not self._allow(
                        "open_tabs", f"Open {step.count} blank tabs in window {window}"
                    ):
                        yield ToolRefused("open_tabs", "Permission denied")
                        yield Answer("Stopped before opening tabs.")
                        return
                    yield ToolStarted("open_tabs", {"count": step.count})
                    _, total = self.backend.open_tabs(step.count, window)
                    if total != count + step.count:
                        raise ToolError(
                            "The observed tab count does not match the requested increase."
                        )
                    self.state.topic = "tabs"
                    text = f"Verified front window {window}: {count} → {total} tabs."
                elif step.action == "focus_whatsapp":
                    if not self._allow("open_whatsapp", "Focus or open WhatsApp Web in Chrome"):
                        yield ToolRefused("open_whatsapp", "Permission denied")
                        yield Answer("Stopped before focusing WhatsApp.")
                        return
                    yield ToolStarted("open_whatsapp", {})
                    self.state.target = self.backend.focus()
                    self.state.topic = "whatsapp"
                    text = "WhatsApp Web is observed in the selected Chrome window and tab."
                elif step.action == "select_chat":
                    if not step.recipient:
                        raise ToolError("Which recipient should I select?")
                    if self.state.target is None:
                        raise ToolError("Focus WhatsApp first, then select a conversation.")
                    if not self._allow(
                        "open_whatsapp_chat", f"Select conversation {step.recipient}"
                    ):
                        yield ToolRefused("open_whatsapp_chat", "Permission denied")
                        yield Answer("Stopped before selecting a conversation or typing.")
                        return
                    yield ToolStarted("open_whatsapp_chat", {})
                    try:
                        self.state.target = self.backend.select(
                            step.recipient,
                            self.state.target,
                            choice,
                        )
                    except AmbiguousContactError as exc:
                        self.state.candidates = exc.contacts
                        self.state.pending = plan[index:]
                        yield ToolFinished("open_whatsapp_chat", "Recipient clarification needed")
                        yield Answer(self._question())
                        return
                    choice = None
                    text = f"Verified selected conversation: {self.state.target.recipient}."
                elif step.action == "draft":
                    target = self._target(draft=True)
                    if not self._allow(
                        "put_in_message_box", f"Write an unsent draft in {target.recipient}"
                    ):
                        yield ToolRefused("put_in_message_box", "Permission denied")
                        yield Answer("Stopped before entering text.")
                        return
                    if target.draft and (
                        self.approve is None or not self.approve(target.draft, step.text)
                    ):
                        yield Answer("Existing draft kept. No text was replaced.")
                        return
                    yield ToolStarted("put_in_message_box", {})
                    self.state.target = self.backend.write(target, step.text)
                    if (
                        self.state.target.identity != target.identity
                        or self.state.target.draft != step.text
                        or self.state.target.outgoing != target.outgoing
                    ):
                        raise ToolError(
                            "Recipient, exact draft or unsent state could not be verified."
                        )
                    text = f"Verified unsent draft in {target.recipient}:\n{step.text}"
                elif step.action == "send":
                    if self.state.uncertain_send is not None:
                        previous = self.state.uncertain_send
                        observed = self.backend.observe()
                        if session.submission_observed(previous, observed):
                            self.state.target = observed
                            self.state.uncertain_send = None
                            yield Answer(
                                "The earlier message is now observed as submitted. No resend."
                            )
                            return
                        raise ToolError(
                            "An earlier send is unconfirmed. Check the chat before resending."
                        )
                    target = self._target(draft=True)
                    if not target.draft:
                        raise ToolError("There is no draft to send.")
                    if self.approve is None or not self.approve(
                        f"send:{target.recipient}", target.draft
                    ):
                        yield ToolRefused("send_message", "Sending was not confirmed")
                        yield Answer("Draft kept. No message was sent.")
                        return
                    # Re-observe after the user has answered the consent prompt.
                    target = self._target(draft=True)
                    yield ToolStarted("send_message", {})
                    self.state.uncertain_send = target
                    submitted = self.backend.send(target)
                    if not session.submission_observed(target, submitted):
                        raise SendUncertainError(
                            "Submission outcome is not independently verified."
                        )
                    self.state.target = submitted
                    self.state.uncertain_send = None
                    text = f"Verified message submitted to {target.recipient}:\n{target.draft}"
                else:
                    raise ToolError("Unsupported plan step.")
            except (ToolError, ProviderError) as exc:
                name = step.action
                yield ToolFinished(name, str(exc), failed=True)
                prefix = "Completed: " + ", ".join(completed) + ". " if completed else ""
                yield Answer(prefix + f"Stopped at {name}: {exc}")
                return
            name = {
                "count_tabs": "list_tabs",
                "focus_whatsapp": "open_whatsapp",
                "select_chat": "open_whatsapp_chat",
                "draft": "put_in_message_box",
                "send": "send_message",
            }.get(step.action, step.action)
            yield ToolFinished(name, text)
            completed.append(step.action)
        if completed:
            yield Answer(text)
