"""Native session guards and acknowledgement recovery, with all OS I/O mocked."""

import json
from dataclasses import replace

import pytest

from rolex.core.errors import ToolError
from rolex.tools import session


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state: dict[str, object] = {
        "target": session.Target(
            12,
            34,
            "https://web.whatsapp.com/",
            "Fixture",
            "",
            signed_in=True,
            conversation_id="fixture-id",
        ),
        "scripts": [],
        "clicks": [],
    }
    monkeypatch.setattr(session, "observe", lambda: state["target"])
    monkeypatch.setattr("rolex.tools.session.browser._focus", lambda: None)
    monkeypatch.setattr("rolex.tools.session.browser._javascript", lambda _: '{"x": 10, "y": 20}')
    monkeypatch.setattr("rolex.tools.session.pointer.click", lambda x, y: None)
    monkeypatch.setattr("rolex.tools.session.time.sleep", lambda _: None)
    return state


def test_native_exact_unicode_draft_uses_guarded_paste_and_restores_clipboard(
    native: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = native["target"]
    assert isinstance(target, session.Target)
    text = "मी १२ मिनिटांत पोहोचतो.\n₹2,750"
    scripts: list[str] = []

    def paste(script: str) -> str:
        scripts.append(script)
        native["target"] = replace(target, draft=text)
        return ""

    monkeypatch.setattr("rolex.tools.session.browser._osascript", paste)
    result = session.write_draft(target, text)
    assert result.draft == text
    assert result.outgoing == ()
    assert "fixture-id" in scripts[0]
    assert "saved_clipboard" in scripts[0]
    assert "set the clipboard to saved_clipboard" in scripts[0]
    assert "key code 36" not in scripts[0]


def test_manual_same_label_switch_blocks_native_typing(
    native: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = native["target"]
    assert isinstance(expected, session.Target)
    native["target"] = replace(expected, conversation_id="other-contact")
    scripts: list[str] = []
    monkeypatch.setattr("rolex.tools.session.browser._osascript", lambda text: scripts.append(text))
    with pytest.raises(ToolError, match="changed"):
        session.write_draft(expected, "hello")
    assert scripts == []


def test_timeout_after_enter_is_resolved_from_new_outgoing_message(
    native: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = native["target"]
    assert isinstance(initial, session.Target)
    expected = replace(initial, draft="Test payload")
    native["target"] = expected
    attempts: list[str] = []

    def submit(script: str) -> str:
        attempts.append(script)
        native["target"] = replace(expected, draft="", outgoing=(("new-id", "Test payload"),))
        raise ToolError("Apple Event timeout")

    monkeypatch.setattr("rolex.tools.session.browser._osascript", submit)
    result = session.send(expected)
    assert result.outgoing == (("new-id", "Test payload"),)
    assert len(attempts) == 1


def test_unknown_submission_outcome_is_not_reported_as_sent(
    native: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = native["target"]
    assert isinstance(initial, session.Target)
    expected = replace(initial, draft="Test payload")
    native["target"] = expected
    monkeypatch.setattr("rolex.tools.session.browser._osascript", lambda _: "")
    with pytest.raises(session.SendUncertainError):
        session.send(expected)


def test_duplicate_rows_are_returned_for_clarification_before_clicking(
    native: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = native["target"]
    assert isinstance(expected, session.Target)
    rows = [
        {"key": "a", "label": "Sam", "detail": "Work", "x": 1, "y": 2},
        {"key": "b", "label": "Sam", "detail": "Home", "x": 3, "y": 4},
    ]
    responses = iter(["ok", json.dumps(rows), json.dumps(rows), ""])
    monkeypatch.setattr("rolex.tools.session.browser._javascript", lambda _: next(responses))

    def no_click(x: float, y: float) -> None:
        pytest.fail("Ambiguous contacts must not be clicked")

    monkeypatch.setattr("rolex.tools.session.pointer.click", no_click)
    with pytest.raises(session.AmbiguousContactError) as caught:
        session.select_chat("Sam", expected)
    assert [contact.key for contact in caught.value.contacts] == ["a", "b"]


def test_window_ids_are_compared_as_text_when_creating_tabs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter([(955779485, 3), (955779485, 5)])
    scripts: list[str] = []
    monkeypatch.setattr(session, "tab_count", lambda: next(observations))

    def create(script: str) -> str:
        scripts.append(script)
        return "71\n72"

    monkeypatch.setattr("rolex.tools.session.browser._osascript", create)
    created: list[int] = []
    assert session.open_tabs(2, 955779485, created) == (955779485, 5)
    assert created == [71, 72]
    assert "(id of front window as text)" in scripts[0]


def test_an_older_matching_message_is_not_submission_evidence() -> None:
    expected = session.Target(
        1,
        2,
        "https://web.whatsapp.com/",
        "Fixture",
        "Hello",
        outgoing=(("last", "Earlier message"),),
        signed_in=True,
        conversation_id="fixture",
    )
    observed = replace(
        expected,
        draft="",
        outgoing=(
            ("historical", "Hello"),
            ("last", "Earlier message"),
        ),
    )
    assert not session.submission_observed(expected, observed)
    assert session.submission_observed(
        expected,
        replace(
            expected,
            draft="",
            outgoing=(*expected.outgoing, ("new", "Hello")),
        ),
    )
