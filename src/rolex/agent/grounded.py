"""Source-grounded command families; literal message content is never executable.

These rules recognize control vocabulary and source slots, not specific names or
message bodies. Unrecognized phrasing remains available to the constrained model.
"""

import re
from dataclasses import dataclass

from rolex.core.errors import ProviderError
from rolex.tools.names import search_name

_OPEN = re.compile(
    r"\b(?:open|focus|switch|go|select|locate|navigate|get|khol|kholo|chalu|ughad|ja|jaa)\b"
    r"|उघड|खोल|चालू|जा(?:\s|[.!?]|$)",
    re.I,
)
_TABS = re.compile(r"\b(?:tabs?|taps?)\b|टॅब|टैब", re.I)
_WHATSAPP = re.compile(r"whats?app|व्हॉट्सअॅप|व्हाट्सअॅप|वॉट्सअॅप|व्हॉट्सअप|व्हाट्सएप", re.I)
_CHAT = re.compile(r"\b(?:chat|conversation|group)\b|चॅट|चैट", re.I)
_WRITE = re.compile(
    r"\b(?:type|write|enter|insert|paste|likho|likh|lihi|liha)\b|लिही|लिहा|लिखो|लिख", re.I
)
_POLITE = re.compile(
    r"^(?:(?:hello|hi|ok|okay|yes|please|pls|can you|could you|would you)\s+)+",
    re.I,
)
_NUMBERS = {
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
    "don": 2,
    "teen": 3,
    "char": 4,
    "panch": 5,
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


@dataclass(frozen=True)
class GroundedAction:
    action: str
    value: str = ""


def _body(said: str) -> tuple[str, str | None, str]:
    found = _WRITE.search(said)
    if not found:
        return said, None, ""
    tail = said[found.end() :].lstrip(" ,:")
    tail = re.sub(r"^(?:karo|kar|kara|करो|कर|करा|दो)\b\s*[, :]*", "", tail, flags=re.I)
    if not tail:
        return said, None, ""
    # A request to generate content belongs to normal chat.
    if re.match(r"(?:me\s+)?(?:a|an|some)\s+", tail, re.I):
        return said, None, ""
    pairs = {'"': '"', "'": "'", "\u2018": "\u2019", "\u201c": "\u201d"}
    if tail[0] in pairs:
        end = tail.rfind(pairs[tail[0]])
        if end == 0:
            raise ProviderError("Please close the quote around the dictated message.")
        return said[: found.end()], tail[1:end], tail[end + 1 :]
    return said[: found.end()], tail, ""


def _recipient(clause: str) -> str:
    cleaned = re.sub(
        r"\b(?:open|focus|select|locate|show|khol|kholo|karo|kar|kara|chalu|ughad|"
        r"chat|conversation|group|ka|ki|ke|with|of|called|named|the|that|this)\b"
        r"|त्यात|त्याचं|चॅट|चैट|उघडा|उघड|खोलो|खोल|करा|करो|चालू",
        " ",
        clause,
        flags=re.I,
    )
    cleaned = cleaned.strip(" ,.!?:")
    cleaned = re.sub(r"(?i)['\u2019]s(?=\s|$)", "", cleaned)
    cleaned = re.sub(r"(?:चं|चे|चा)(?=\s|$)", "", cleaned)
    return " ".join(cleaned.split())


def extract(said: str) -> list[GroundedAction] | None:
    if said.strip(" .!?").casefold() in {"hello", "hi", "hey", "नमस्कार", "नमस्ते"}:
        return [GroundedAction("chat")]
    source = _POLITE.sub("", said.strip())
    if _WRITE.search(source) and re.match(
        r"how\s+(?:to|do|can)|what\b|why\b|कसे|कसं|काय|कैसे|क्यों",
        source,
        re.I,
    ):
        return None
    control, body, trailing = _body(source)
    plain = source.strip(" .!?").casefold()
    if re.fullmatch(
        r"(?:send(?:\s+(?:it|this|that|the message|the draft))?|"
        r"पाठव|पाठवा|भेजो|भेज दो|bhej do|bhejo|pathav|pathva)",
        plain,
    ):
        return [GroundedAction("send")]
    actions: list[GroundedAction] = []
    if _TABS.search(control):
        words = "|".join(re.escape(word) for word in _NUMBERS)
        number = re.search(
            rf"(?P<n>\d+|{words})\s+(?:(?:more|additional|blank|new)\s+)?"
            r"(?:tabs?|taps?|टॅब|टैब)",
            control,
            re.I,
        )
        if _OPEN.search(control) and number:
            token = number["n"].casefold()
            count = int(token) if token.isdecimal() else _NUMBERS[token]
            if not 1 <= count <= 10:
                raise ProviderError("Please request between 1 and 10 additional tabs.")
            actions.append(GroundedAction("open_tabs", str(count)))
        elif re.search(r"how many|count|check|किती|कितने|kiti|kitne", control, re.I):
            actions.append(GroundedAction("count_tabs"))
    if _WHATSAPP.search(control) and _OPEN.search(control):
        actions.append(GroundedAction("focus_whatsapp"))
    # Split only the control portion. A conjunction inside the message stays literal.
    clauses = re.split(r"[,;]|\b(?:and|aur|ani|aani)\b|आणि|और", control, flags=re.I)
    for clause in clauses:
        if _WHATSAPP.search(clause) and not _CHAT.search(clause):
            continue
        if (_CHAT.search(clause) and (_OPEN.search(clause) or actions)) or (
            _OPEN.search(clause) and any(action.action == "focus_whatsapp" for action in actions)
        ):
            name = _recipient(clause)
            if name and not _WHATSAPP.fullmatch(name):
                actions.append(GroundedAction("select_chat", search_name(name)))
    if body is not None:
        # Dictating requires a clear current-chat reference or a selected chat step.
        explicit = bool(
            _WRITE.search(control)
            or actions
            or re.search(r"त्याला|त्यात|उसको|इसे|\b(?:type|write)\b", control, re.I)
        )
        if explicit:
            actions.append(GroundedAction("draft", body))
    if trailing.strip(" ,.!?"):
        if re.fullmatch(r"\s*(?:and\s+send\s+it|आणि\s+पाठव|और\s+भेजो)\s*[.!?]*", trailing, re.I):
            actions.append(GroundedAction("send"))
        else:
            raise ProviderError("Please give any instruction after the quoted draft separately.")
    return actions or None
