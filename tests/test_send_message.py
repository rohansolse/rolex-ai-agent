"""Sending must use the recipient and words the user confirmed."""

import pytest

from rolex.core.errors import ToolError
from rolex.tools import chat_page


@pytest.mark.parametrize("change", ["recipient", "text", "missing_recipient", "none"])
def test_send_checks_the_approved_details_after_focusing(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    state = {"recipient": "Priya", "text": "Friday at 5"}
    keys: list[str] = []

    def click(x: float, y: float) -> None:
        if change == "recipient":
            state["recipient"] = "Rohan"
        elif change == "text":
            state["text"] = "Saturday at 6"
        elif change == "missing_recipient":
            state["recipient"] = ""

    def type_key(key: str) -> None:
        keys.append(key)
        state["text"] = ""

    monkeypatch.setattr(chat_page, "_require_chrome", lambda: None)
    monkeypatch.setattr(chat_page, "_focus", lambda: None)
    monkeypatch.setattr(chat_page, "_javascript", lambda script: '{"x": 1, "y": 2}')
    monkeypatch.setattr(chat_page, "message_box_text", lambda: state["text"])
    monkeypatch.setattr(chat_page, "open_conversation", lambda: state["recipient"])
    monkeypatch.setattr("rolex.tools.chat_page.pointer.click", click)
    monkeypatch.setattr(chat_page, "_type", type_key)
    monkeypatch.setattr("rolex.tools.chat_page.time.sleep", lambda seconds: None)

    if change == "none":
        assert (
            chat_page.send_message(expected_recipient="Priya", expected_text="Friday at 5")
            == "Sent to Priya: Friday at 5"
        )
        assert keys == ["key code 36"]
    else:
        with pytest.raises(ToolError, match="changed"):
            chat_page.send_message(expected_recipient="Priya", expected_text="Friday at 5")
        assert keys == []
