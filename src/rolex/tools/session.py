"""Observed Chrome/WhatsApp targets for multi-step commands.

Apple Events attach to the user's running Google Chrome. Window and tab IDs,
rather than an isolated browser profile, identify the controlled surface.
"""

import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from rolex.core.errors import ToolError
from rolex.tools import browser, chat_page, pointer

_STATE_JS = r"""
JSON.stringify((() => {
 const main = document.querySelector('#main');
 const header = main && main.querySelector('header');
 const boxes = main && main.querySelectorAll('[contenteditable=true][role=textbox]');
 const box = boxes && boxes[boxes.length-1];
 const outgoing = main ? [...main.querySelectorAll('.message-out')].slice(-40).map(row => ({
   id: row.getAttribute('data-id') || (row.querySelector('[data-id]') || {}).dataset?.id || '',
   text: (row.querySelector('.selectable-text') || row).innerText || ''
 })) : [];
 const message = main && main.querySelector('[data-id]');
 const messageId = message ? message.getAttribute('data-id') : '';
 const conversation_id = messageId && messageId.includes('_') ? messageId.split('_')[1] : '';
 return {conversation_id, recipient: header ? (header.innerText || '').split('\n')[0].trim() : '',
         draft: box ? box.innerText : '', outgoing,
         signed_in: !!document.querySelector('#pane-side')};
})())
"""


@dataclass(frozen=True)
class Target:
    window: int
    tab: int
    url: str
    recipient: str
    draft: str
    outgoing: tuple[tuple[str, str], ...] = ()
    signed_in: bool = False
    conversation_id: str = ""

    @property
    def identity(self) -> tuple[int, int, str, str]:
        return self.window, self.tab, self.recipient, self.conversation_id


@dataclass(frozen=True)
class Contact:
    key: str
    label: str
    detail: str
    x: float
    y: float


class AmbiguousContactError(ToolError):
    def __init__(self, contacts: list[Contact]) -> None:
        super().__init__("More than one conversation matches. Choose one.")
        self.contacts = contacts


class SendUncertainError(ToolError):
    """Submission was attempted; repeating without new observation may duplicate it."""


def observe() -> Target:
    browser._require_chrome()
    raw = browser._osascript(
        'tell application "Google Chrome" to tell front window\n'
        "set w to id\nset t to active tab\n"
        f"set observed_data to execute t javascript {browser._quote(_STATE_JS)}\n"
        "return (w as text) & linefeed & (id of t as text) & linefeed & "
        "(URL of t) & linefeed & observed_data\nend tell"
    )
    try:
        w, t, url, data = raw.split("\n", 3)
        state = json.loads(data)
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("recipient"), str)
            or not isinstance(state.get("draft"), str)
            or not isinstance(state.get("outgoing"), list)
        ):
            raise ValueError("invalid observation")
        return Target(
            int(w),
            int(t),
            url,
            state["recipient"],
            state["draft"],
            tuple((row["id"], row["text"]) for row in state["outgoing"]),
            state["signed_in"],
            state.get("conversation_id", ""),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ToolError("Could not observe the active browser target.") from exc


def require_same(expected: Target, *, draft: bool = False) -> Target:
    current = observe()
    if (
        current.identity != expected.identity
        or urlparse(current.url).hostname != "web.whatsapp.com"
    ):
        raise ToolError(
            "The active window, tab or recipient changed. Select the intended chat again."
        )
    if draft and current.draft != expected.draft:
        raise ToolError("The draft changed. Review the new draft before requesting send again.")
    return current


def focus_whatsapp() -> Target:
    chat_page.open_whatsapp()
    for _ in range(24):
        current = observe()
        if urlparse(current.url).hostname != "web.whatsapp.com":
            raise ToolError("WhatsApp focus did not reach the intended Chrome tab.")
        if current.signed_in:
            return current
        time.sleep(0.25)
    raise ToolError("WhatsApp Web needs login or has not finished loading.")


def tab_count() -> tuple[int, int]:
    browser._require_chrome()
    raw = browser._osascript(
        'tell application "Google Chrome" to tell front window\n'
        'return (id as text) & ":" & (count of tabs as text)\nend tell'
    )
    try:
        window, count = raw.split(":")
        return int(window), int(count)
    except ValueError as exc:
        raise ToolError("Could not observe the front Chrome window's tab count.") from exc


def open_tabs(count: int, window: int, created: list[int] | None = None) -> tuple[int, int]:
    if not 1 <= count <= 10:
        raise ToolError("Request between 1 and 10 tabs.")
    before_window, before = tab_count()
    if before_window != window:
        raise ToolError("The front window changed before opening tabs.")
    raw = browser._osascript(
        'tell application "Google Chrome"\n'
        f'if (id of front window as text) is not "{window}" then error "window changed"\n'
        "set created_ids to {}\n"
        f"repeat {count} times\n"
        "set fresh_tab to make new tab at end of tabs of front window "
        'with properties {URL:"about:blank"}\n'
        "set end of created_ids to id of fresh_tab\n"
        "end repeat\n"
        "end tell\n"
        "set AppleScript's text item delimiters to linefeed\n"
        "return created_ids as text"
    )
    try:
        observed_ids = [int(value) for value in raw.splitlines()]
    except ValueError as exc:
        raise ToolError("Could not identify the newly created tabs.") from exc
    if created is not None:
        created.extend(observed_ids)
    after_window, after = tab_count()
    if after_window != window or after != before + count:
        raise ToolError("Tab opening was only partially verified. Check the current tab count.")
    return after_window, after


_CONTACTS_JS = r"""
(() => {
 const wanted = __NAME__.toLocaleLowerCase();
 const rows = [...document.querySelectorAll('#pane-side [role=listitem],#pane-side [role=row]')]
   .filter(row => row.getClientRects().length);
 const all = rows.map((row,i) => {
   const n = row.querySelector('[title]');
   const label = ((n && n.title) || row.innerText || '').split('\n')[0].trim();
   const detail = (row.innerText || '').replace(/\s+/g,' ').trim().slice(0,180);
   const b = row.getBoundingClientRect();
   const zoom = window.innerWidth ? window.outerWidth/window.innerWidth : 1;
   return {key: row.getAttribute('data-id') || row.id || label+'|'+detail+'|'+i,
     label, detail,
     x: Math.round(window.screenX+(b.left+b.width/2)*zoom),
     y: Math.round(window.screenY+window.outerHeight-window.innerHeight*zoom+
                   (b.top+b.height/2)*zoom)};
 });
 const exact = all.filter(row => row.label.toLocaleLowerCase() === wanted);
 return JSON.stringify(exact.length ? exact : all.filter(row =>
   wanted.split(/\s+/).every(word => row.label.toLocaleLowerCase().includes(word))));
})()
"""


def select_chat(name: str, expected: Target, choice: Contact | None = None) -> Target:
    require_same(expected)
    browser._focus()
    require_same(expected)
    if (
        browser._javascript(
            chat_page._SEARCH.replace("__WANTED__", browser._quote(chat_page._search_term(name)))
        )
        != "ok"
    ):
        raise ToolError("Could not search WhatsApp. Check that WhatsApp Web is signed in.")
    try:
        contacts: list[Contact] = []
        previous: list[Contact] = []
        for _ in range(24):
            require_same(expected)
            raw = browser._javascript(_CONTACTS_JS.replace("__NAME__", browser._quote(name)))
            try:
                values = json.loads(raw)
                if not isinstance(values, list):
                    raise ValueError("invalid candidates")
                contacts = [Contact(**row) for row in values]
            except (ValueError, TypeError) as exc:
                raise ToolError("Could not observe contact search results.") from exc
            if contacts and contacts == previous:
                break
            previous = contacts
            time.sleep(0.25)
        if not contacts:
            raise ToolError("No matching conversation was found. Selection stopped before typing.")
        if choice is not None:
            contacts = [
                contact
                for contact in contacts
                if contact.key == choice.key
                and contact.label == choice.label
                and contact.detail == choice.detail
            ]
            if len(contacts) != 1:
                raise ToolError("The contact choices changed. Search again.")
        if len(contacts) != 1:
            raise AmbiguousContactError(contacts)
        contact = contacts[0]
        require_same(expected)
        pointer.click(contact.x, contact.y)
        for _ in range(24):
            current = observe()
            if (current.window, current.tab) != (expected.window, expected.tab):
                raise ToolError("Browser target changed during contact selection.")
            if current.recipient.casefold() == contact.label.casefold():
                return current
            time.sleep(0.25)
        raise ToolError("The selected conversation did not become visible. Typing stopped.")
    finally:
        # Never clear a search box in a different window/tab after a manual change.
        current = observe()
        if (current.window, current.tab) == (expected.window, expected.tab):
            browser._javascript(chat_page._CLEAR_SEARCH)


def _message_position() -> dict[str, Any]:
    try:
        value = json.loads(browser._javascript(chat_page._BOX_POSITION))
        if not isinstance(value, dict) or "x" not in value or "y" not in value:
            raise ValueError("missing position")
        return value
    except (ValueError, TypeError) as exc:
        raise ToolError("No message box is visible in the selected conversation.") from exc


def write_draft(expected: Target, text: str) -> Target:
    if not text or "\x00" in text:
        raise ToolError("The draft must contain text.")
    require_same(expected, draft=True)
    browser._focus()
    require_same(expected, draft=True)
    position = _message_position()
    if "x" not in position:
        raise ToolError("No message box is visible in the selected conversation.")
    pointer.click(float(position["x"]), float(position["y"]))
    require_same(expected, draft=True)
    guard = _guard_script(expected)
    # Preserve the complete macOS clipboard object, including non-text contents.
    script = (
        guard
        + f"""
set saved_clipboard to the clipboard
try
 set the clipboard to {browser._quote(text)}
 tell application "System Events"
  keystroke "a" using command down
  keystroke "v" using command down
 end tell
 delay 0.3
 set the clipboard to saved_clipboard
on error message_text number message_number
 set the clipboard to saved_clipboard
 error message_text number message_number
end try
"""
    )
    try:
        browser._osascript(script)
    except ToolError as exc:
        raise ToolError("Draft entry failed. Check Accessibility and the active target.") from exc
    current = require_same(expected)
    if current.draft != text:
        raise ToolError("The observed draft differs from the dictated text. Sending stopped.")
    return current


def _guard_script(expected: Target) -> str:
    check = (
        "JSON.stringify((({recipient,draft,conversation_id})=>"
        "({recipient,draft,conversation_id}))(JSON.parse(" + _STATE_JS + ")))"
    )
    expected_json = json.dumps(
        {
            "recipient": expected.recipient,
            "draft": expected.draft,
            "conversation_id": expected.conversation_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        'tell application "Google Chrome"\n'
        f'if (id of front window as text) is not "{expected.window}" then error "window changed"\n'
        f'if (id of active tab of front window as text) is not "{expected.tab}" '
        'then error "tab changed"\n'
        f"set observed to execute active tab of front window javascript {browser._quote(check)}\n"
        f'if observed is not {browser._quote(expected_json)} then error "chat changed"\n'
        "end tell\n"
    )


def submission_observed(expected: Target, current: Target) -> bool:
    """Require a new matching outgoing row after the previously observed tail."""
    if current.identity != expected.identity or current.draft:
        return False
    recent = current.outgoing
    if expected.outgoing:
        try:
            anchor = recent.index(expected.outgoing[-1])
        except ValueError:
            return False
        recent = recent[anchor + 1 :]
    return any(row not in expected.outgoing and row[1] == expected.draft for row in recent)


def send(expected: Target) -> Target:
    require_same(expected, draft=True)
    if not expected.recipient or not expected.draft:
        raise ToolError("Select a conversation with a nonempty draft before sending.")
    browser._focus()
    require_same(expected, draft=True)
    position = _message_position()
    if "x" not in position:
        raise ToolError("No message box is visible.")
    pointer.click(float(position["x"]), float(position["y"]))
    require_same(expected, draft=True)
    with contextlib.suppress(ToolError):
        browser._osascript(
            _guard_script(expected) + 'tell application "System Events" to key code 36'
        )
    for _ in range(12):
        current = require_same(expected)
        if submission_observed(expected, current):
            return current
        time.sleep(0.25)
    raise SendUncertainError("Submission attempted; outcome unconfirmed. Do not resend yet.")
