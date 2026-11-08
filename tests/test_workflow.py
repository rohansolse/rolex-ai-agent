"""Shared chat and voice command processing, without app access."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

import pytest

from rolex.agent.intent import CommandState, Step, interpret
from rolex.agent.loop import Answer
from rolex.agent.workflow import Workflow
from rolex.core.config import Config
from rolex.core.errors import ProviderError
from rolex.interface.chat import run_chat
from rolex.interface.listen import run_listen
from rolex.llm.base import Message, Reply, ToolSchema
from rolex.voice.speak import Speaker
from test_listen import FakeEars, FakeSpeaker
from workflow_fixture import FixtureBrowser


def plan(*steps: Step) -> dict[str, Any]:
    return {
        "steps": [
            {
                "action": step.action,
                "value": str(step.count)
                if step.action == "open_tabs"
                else step.recipient
                if step.action == "select_chat"
                else step.text,
            }
            for step in steps
        ],
        "question": "",
    }


class Planner:
    name = "fixture-planner"

    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self.plans = plans[:]
        self.last: dict[str, Any] = {}

    def is_available(self) -> bool:
        return True

    def structured(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        if self.plans:
            self.last = self.plans.pop(0)
        return self.last

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        pytest.fail("Workflow must not delegate action reporting to a language model")

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        yield self.complete(messages, tools).content


def chat(
    monkeypatch: pytest.MonkeyPatch,
    browser: FixtureBrowser,
    lines: list[str],
    plans: list[dict[str, Any]],
    approvals: list[str] | None = None,
) -> None:
    monkeypatch.setattr("rolex.agent.workflow.NativeBackend", lambda: browser)
    inputs = iter([*lines, "exit"])
    answers = iter(approvals or ["y"] * 30)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: next(answers) if "[y/N]" in prompt else next(inputs)
    )
    assert run_chat(Config(), provider=Planner(plans), tools=[]) == 0


def test_a_variable_tab_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        ["How many tabs are open?", "Open four additional tabs"],
        [plan(Step("count_tabs")), plan(Step("open_tabs", 4))],
    )
    assert browser.count == 7


def test_b_followups_exact_unsent_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    text = "मी १२ मिनिटांत पोहोचतो."
    chat(
        monkeypatch,
        browser,
        ["WhatsApp वर जा.", "Nisha Shah ka chat kholo", f"त्याला लिही, {text}"],
        [
            plan(Step("focus_whatsapp")),
            plan(Step("select_chat", recipient="Nisha Shah")),
            plan(Step("draft", text=text)),
        ],
    )
    assert browser.writes == [("nisha", text)]
    assert browser.current.draft == text
    assert browser.sent == []


def test_c_action_word_inside_compound_payload_is_data(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    text = "Do not send ₹2,750 today"
    chat(
        monkeypatch,
        browser,
        [
            f"WhatsApp kholo, Nisha Shah ka chat open karo aur type karo '{text}'",
        ],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Nisha Shah"),
                Step("draft", text=text),
            )
        ],
    )
    assert browser.current.draft == text
    assert browser.sent == []


def test_d_duplicate_choice_resumes_dependent_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        [
            "Open WhatsApp and Sam chat and type 'Meet at gate 7'",
            "2",
        ],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Sam"),
                Step("draft", text="Meet at gate 7"),
            )
        ],
    )
    assert browser.current.conversation_id == "sam-home"
    assert browser.writes == [("sam-home", "Meet at gate 7")]


def test_d_manual_same_name_switch_stops_typing(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    state = CommandState()
    from rolex.agent.consent import always_allow

    workflow = Workflow(Planner([]), state, always_allow(), lambda *_: True, browser)
    list(workflow.execute([Step("focus_whatsapp"), Step("select_chat", recipient="Sam")]))
    list(workflow.process("1") or [])
    browser.current = replace(browser.current, conversation_id="sam-home")
    events = list(workflow.execute([Step("draft", text="hello")]))
    assert browser.writes == []
    assert isinstance(events[-1], Answer) and "changed" in events[-1].content


@pytest.mark.parametrize("approval, count", [("y", 1), ("n", 0)])
def test_e_send_consent(monkeypatch: pytest.MonkeyPatch, approval: str, count: int) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        [
            "Open WhatsApp, open Nisha Shah and type 'Ready at 6'",
            "पाठव",
        ],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Nisha Shah"),
                Step("draft", text="Ready at 6"),
            ),
            plan(Step("send")),
        ],
        ["y", "y", "y", approval],
    )
    assert len(browser.sent) == count


@pytest.mark.parametrize("submitted", [True, False])
def test_e_lost_ack_never_duplicates(monkeypatch: pytest.MonkeyPatch, submitted: bool) -> None:
    browser = FixtureBrowser()
    browser.submit_then_fail = submitted
    browser.lose_ack = not submitted
    chat(
        monkeypatch,
        browser,
        [
            "Open WhatsApp, open Nisha Shah and type 'Meet at 7'",
            "send it",
            "send it",
        ],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Nisha Shah"),
                Step("draft", text="Meet at 7"),
            ),
            plan(Step("send")),
            plan(Step("send")),
        ],
    )
    assert len(browser.sent) == int(submitted)


def test_failed_selection_stops_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    browser.selection_fails = True
    chat(
        monkeypatch,
        browser,
        ["Open WhatsApp, open Nisha Shah, type 'Hello'"],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Nisha Shah"),
                Step("draft", text="Hello"),
            ),
        ],
    )
    assert browser.writes == []


def test_cancel_abandons_pending_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        ["Open WhatsApp and Sam chat and type 'Hello'", "cancel"],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Sam"),
                Step("draft", text="Hello"),
            ),
        ],
    )
    assert browser.writes == []


def test_voice_transcript_uses_shared_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    monkeypatch.setattr("rolex.agent.workflow.NativeBackend", lambda: browser)
    assert (
        run_listen(
            Config(),
            provider=Planner([plan(Step("count_tabs"))]),
            tools=[],
            ears=FakeEars(["rolex किती टॅब आहेत?"]),
            speaker=cast(Speaker, FakeSpeaker()),
        )
        == 0
    )


def test_message_body_cannot_authorize_send() -> None:
    provider = Planner([plan(Step("draft", text="Please send the invoice"), Step("send"))])
    assert interpret(provider, "type 'Please send the invoice'") == [
        Step("draft", text="Please send the invoice")
    ]


def test_message_must_be_verbatim() -> None:
    provider = Planner([plan(Step("draft", text="I arrive in 10 minutes"))])
    assert interpret(provider, "लिही, मी दहा मिनिटांत पोहोचतो.") == [
        Step("draft", text="मी दहा मिनिटांत पोहोचतो.")
    ]


def test_wrong_tab_action_is_rejected() -> None:
    provider = Planner([plan(Step("open_tabs", 2))])
    with pytest.raises(ProviderError, match="tabs absent"):
        interpret(provider, "Bring up the messaging service")


def test_explicit_roman_count_corrects_count_query_misclassification() -> None:
    provider = Planner([plan(Step("count_tabs"))])
    assert interpret(provider, "Aur teen tabs khol do") == [Step("open_tabs", 3)]


def test_contact_correction_resumes_original_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        [
            "Open WhatsApp, select Sam and type 'The total is ₹8,430.'",
            "actually Nisha Shah",
        ],
        [
            plan(
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Sam"),
                Step("draft", text="The total is ₹8,430."),
            )
        ],
    )
    assert browser.writes == [("nisha", "The total is ₹8,430.")]
    assert browser.sent == []


def test_recipient_change_during_send_consent_blocks_submission() -> None:
    from rolex.agent.consent import always_allow

    browser = FixtureBrowser()
    state = CommandState()

    def approve(about: str, text: str) -> bool:
        browser.current = replace(browser.current, conversation_id="other-id")
        return True

    workflow = Workflow(Planner([]), state, always_allow(), approve, browser)
    list(
        workflow.execute(
            [
                Step("focus_whatsapp"),
                Step("select_chat", recipient="Nisha Shah"),
                Step("draft", text="Hello"),
            ]
        )
    )
    events = list(workflow.execute([Step("send")]))
    assert browser.sent == []
    assert isinstance(events[-1], Answer) and "changed" in events[-1].content


def test_combined_steps_cannot_execute_out_of_dependency_order() -> None:
    from rolex.agent.intent import validate

    with pytest.raises(ProviderError, match="order"):
        validate(
            plan(Step("draft", text="Hello"), Step("select_chat", recipient="Sam")),
            "Select Sam and type 'Hello'",
        )


def test_tab_recheck_uses_session_topic_without_replaying_old_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = FixtureBrowser()
    chat(
        monkeypatch,
        browser,
        ["How many tabs are open?", "totel check kara kiti ahe live chek kara"],
        [plan(Step("count_tabs"))],
    )
    assert browser.count == 3
