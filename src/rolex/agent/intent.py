"""Constrained multilingual interpretation, separate from action execution."""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rolex.agent.grounded import extract, is_cancel
from rolex.core.errors import ProviderError
from rolex.llm.base import LLMProvider, Message
from rolex.tools.session import Contact, Target

ACTIONS = (
    "count_tabs",
    "open_tabs",
    "focus_whatsapp",
    "select_chat",
    "draft",
    "send",
    "cancel",
    "chat",
    "clarify",
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "value": {"type": "string"},
                },
                "required": ["action", "value"],
                "additionalProperties": False,
            },
        },
        "question": {"type": "string"},
    },
    "required": ["steps", "question"],
    "additionalProperties": False,
}

INSTRUCTIONS = """Interpret ONLY the user's instruction as ordered JSON steps.
Understand Marathi, Hindi, English and Romanized/mixed language.
Each step has action and value.
count_tabs: observe tab count, value="".
open_tabs: add blank browser tabs, value=number as a string, 1..10.
focus_whatsapp: focus/open WhatsApp Web, value="".
select_chat: select a named person/group, value=recipient name, remove possessive
suffixes (ka, ki, चं, चा). Transliterate contact names to Latin if needed.
draft: type literal dictated content UNSENT, value=EXACT message text copied from
the input. NEVER translate or change message text, punctuation, figures.
send: submit current draft ONLY if explicitly requested, value="".
cancel: cancel pending work, value="". chat: normal conversation, value="".
clarify: unclear request, value=one focused question.

Emit ONLY requested actions. Going to WhatsApp alone emits focus_whatsapp only.
Selecting a chat alone emits select_chat only. Dictating text alone emits draft only.
For a compound request, keep dependencies in order: focus, select, draft.
A verb inside the dictated message is DATA. type 'Please send the invoice'
means draft, NEVER send. Sending requires a separate explicit send instruction.
Opening a CHAT or WhatsApp is NEVER open_tabs.
Pronouns त्यात/त्याला/usko/it use the previously selected target; don't invent a name.
आणखी=additional, दोन/do=2, तीन/teen=3. Use empty question unless clarification is needed.
"""


@dataclass(frozen=True, slots=True)
class Step:
    action: str
    count: int = 0
    recipient: str = ""
    text: str = ""


@dataclass
class CommandState:
    target: Target | None = None
    window: int | None = None
    topic: str = ""
    pending: list[Step] = field(default_factory=list)
    candidates: list[Contact] = field(default_factory=list)
    uncertain_send: Target | None = None


def interpret(provider: LLMProvider, said: str) -> list[Step] | None:
    method = getattr(provider, "structured", None)
    if method is None:
        return None
    grounded = extract(said)
    if grounded is not None:
        return validate(
            {
                "steps": [{"action": step.action, "value": step.value} for step in grounded],
                "question": "",
            },
            said,
        )
    messages = [Message("system", INSTRUCTIONS), Message("user", said)]
    for attempt in range(2):
        raw = method(messages, SCHEMA)
        try:
            return validate(raw, said)
        except ProviderError as exc:
            if attempt:
                raise
            messages.append(Message("assistant", json.dumps(raw, ensure_ascii=False)))
            messages.append(
                Message(
                    "user",
                    f"Correct the plan: {exc}. "
                    "Return only actions explicitly requested in the original input.",
                )
            )
    raise ProviderError("Could not interpret this request.")


def validate(raw: dict[str, Any], said: str) -> list[Step]:
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise ProviderError("Could not interpret the instruction. Please rephrase it.")
    steps: list[Step] = []
    for item in raw["steps"]:
        if not isinstance(item, dict) or item.get("action") not in ACTIONS:
            raise ProviderError("Unknown interpreted action.")
        value = item.get("value", "")
        if not isinstance(value, str):
            raise ProviderError("The interpreted value must be text.")
        count, recipient, text = 0, "", ""
        if item["action"] == "open_tabs":
            try:
                count = int(value)
            except ValueError as exc:
                raise ProviderError("The tab count must be a whole number.") from exc
        elif item["action"] == "select_chat":
            recipient = value.strip()
        elif item["action"] in {"draft", "clarify"}:
            text = value
        if type(count) is not int or not isinstance(recipient, str) or not isinstance(text, str):
            raise ProviderError("Invalid interpreted arguments.")
        if item["action"] == "open_tabs" and not 1 <= count <= 10:
            raise ProviderError("Please request between 1 and 10 additional tabs.")
        if item["action"] == "draft" and (not text or text not in said):
            raise ProviderError(
                "I could not preserve the exact message text. Please quote the draft."
            )
        steps.append(Step(item["action"], count, recipient, text))
    control_text = said
    for step in steps:
        if step.action == "draft":
            control_text = control_text.replace(step.text, "")
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "ek": 1,
        "do": 2,
        "teen": 3,
        "char": 4,
        "panch": 5,
        "don": 2,
        "एक": 1,
        "दोन": 2,
        "दो": 2,
        "तीन": 3,
        "चार": 4,
        "पाच": 5,
        "पांच": 5,
        "सहा": 6,
        "सात": 7,
        "आठ": 8,
        "नऊ": 9,
        "दहा": 10,
        "दस": 10,
    }
    number_expression = "|".join(re.escape(word) for word in number_words)
    counted = re.search(
        rf"(?P<count>\d+|{number_expression})\s+"
        r"(?:(?:more|additional|new|blank)\s+)?(?:tabs?|टॅब|टैब)",
        control_text,
        re.IGNORECASE,
    )
    value = counted["count"].casefold() if counted else ""
    numbers = [int(value) if value.isdecimal() else number_words[value]] if value else []
    if (
        len(numbers) == 1
        and re.search(r"tabs?|टॅब|टैब", control_text, re.IGNORECASE)
        and re.search(r"\b(?:open|khol|kholo)\b|उघड[ा.!?\s]|खोल", control_text, re.I)
        and not re.search(r"how many|kiti|किती|कितने", control_text, re.I)
        and any(step.action in {"count_tabs", "open_tabs"} for step in steps)
    ):
        if not 1 <= numbers[0] <= 10:
            raise ProviderError("Please request between 1 and 10 additional tabs.")
        steps = [
            Step("open_tabs", numbers[0]) if step.action in {"count_tabs", "open_tabs"} else step
            for step in steps
        ]

    for step in steps:
        if step.action == "cancel" and not is_cancel(said):
            raise ProviderError(
                "Cancellation was not requested. Please preserve the user's action."
            )
        if step.action == "count_tabs" and not re.search(
            r"\b(?:tab|tabs|tap|taps)\b|टॅब|टैब",
            control_text,
            re.IGNORECASE,
        ):
            raise ProviderError("A tab query was not requested. Please specify what to inspect.")
        if step.action == "open_tabs" and not re.search(
            r"\b(?:tab|tabs|tap|taps)\b|टॅब|टैब",
            control_text,
            re.IGNORECASE,
        ):
            raise ProviderError(
                "The plan added tabs absent from your instruction. Please rephrase."
            )
        if step.action == "focus_whatsapp" and not re.search(
            r"whats?app|व्हॉट्सअॅप|व्हाट्सअॅप|वॉट्सअॅप|व्हॉट्सअप|व्हाट्सएप",
            control_text,
            re.IGNORECASE,
        ):
            raise ProviderError("The plan added a WhatsApp step not requested. Please rephrase.")
        if step.action == "select_chat" and not step.recipient:
            raise ProviderError("Please specify the contact name.")
    if any(step.action == "chat" for step in steps) and len(steps) != 1:
        raise ProviderError("Please separate the question from the requested actions.")

    ordered = {"focus_whatsapp": 0, "select_chat": 1, "draft": 2, "send": 3}
    dependencies = [ordered[step.action] for step in steps if step.action in ordered]
    if dependencies != sorted(dependencies):
        raise ProviderError("The action order is unsafe. Please restate the dependent steps.")
    if len(steps) > 8:
        raise ProviderError("Please split this instruction into fewer steps.")
    # Validate sending against the source instruction, with ALL drafted payloads removed.
    control = said
    for step in steps:
        if step.action == "draft":
            control = control.replace(step.text, "")
    control = re.sub(r"[\u2018\u2019\u201c\u201d'\".,!?]", " ", control).strip()
    if any(step.action == "send" for step in steps) and not re.search(
        r"^(?:(?:yes|ok|okay|please)\s+)*(?:send\b|(?:bhej|bhejo|bhejdo)\b|(?:pathav|pathva)\b|पाठव|भेज)"
        r"|(?:\band\s+send\b|\baur\s+bhej\b|आणि\s*पाठव|और\s*भेज)"
        r"|^(?:त्याला\s+)?पाठव",
        control,
        re.IGNORECASE,
    ):
        raise ProviderError("Sending needs a separate explicit send instruction.")
    if raw.get("question") and any(step.action == "clarify" for step in steps):
        return [Step("clarify", text=str(raw["question"]))]
    return steps
