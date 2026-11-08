"""Reading a conversation, and putting a draft into its message box.

Written for WhatsApp Web, which is what this is used for, but nothing here
names it: the selectors are the ones any chat application built with ARIA
roles uses, and the fallbacks are generic.

Separate from :mod:`rolex.tools.browser` because the job is different. That
module is about a page in general; this is about the one thing rolex is
asked to do most — read what a client said and help answer it.
"""

import json
import time
from typing import Any

from rolex.core.errors import ToolError
from rolex.tools import pointer
from rolex.tools.browser import _focus, _javascript, _quote, _require_chrome, _type
from rolex.tools.registry import register

MESSAGE_LIMIT = 30
"""How many recent messages to read.

Enough to see what is being discussed, few enough to leave the model room to
think. A long conversation read in full fills the context window with
history nobody asked about.
"""


@register(
    description=(
        "Read the chat conversation open in Chrome: who said what, most recent last. "
        "Use before writing a reply."
    )
)
def read_chat() -> str:
    """The open conversation, as lines of "them:" and "you:"."""
    _require_chrome()
    raw = _javascript(_READ_CHAT.replace("__LIMIT__", str(MESSAGE_LIMIT)))

    try:
        found = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("could not read the conversation on this page") from exc

    if error := found.get("error"):
        raise ToolError(error)

    lines = found.get("messages") or []
    if not lines:
        raise ToolError("there are no messages in this conversation yet")

    who = found.get("with") or "this chat"
    return f"Conversation with {who}:\n" + "\n".join(lines)


_READ_CHAT = """
(() => {
  const main = document.querySelector('#main') || document.querySelector('[role=main]');
  if (!main) return JSON.stringify({error: 'no conversation is open. Open one first.'});

  const header = main.querySelector('header');
  const who = header ? (header.innerText || '').split('\\n')[0].trim() : '';

  // WhatsApp marks incoming and outgoing messages with these classes. Where
  // they are absent, every row is treated as incoming, which is the safer
  // guess: a draft that answers your own words is obviously wrong, while one
  // that answers theirs is merely unattributed.
  const rows = [...main.querySelectorAll('.message-in, .message-out')];
  const source = rows.length ? rows : [...main.querySelectorAll('[role=row]')];

  const said = [];
  for (const row of source.slice(-__LIMIT__)) {
    const mine = row.classList.contains('message-out');
    const text = (row.innerText || '').replace(/\\s+/g, ' ').trim();
    if (!text) continue;
    const speaker = mine ? 'you: ' : row.classList.contains('message-in') ? 'them: ' : 'unknown: ';
    said.push(speaker + text.slice(0, 400));
  }

  return JSON.stringify({with: who, messages: said});
})()
"""


@register(
    description=(
        "Put text into the message box of the chat open in Chrome, ready for the "
        "user to read and send. Does not send it."
    ),
    safety="mutating",
)
def put_in_message_box(text: str) -> str:
    """Type `text` into the conversation's message box.

    Deliberately stops short of sending. A draft the user reads before it
    goes is a help; one that leaves on its own is a risk they did not agree
    to. Sending stays a deliberate press of Enter.

    Args:
        text: The draft to put there.
    """
    draft = " ".join(text.split())
    if not draft:
        raise ToolError("there is nothing to put in the message box")

    _require_chrome()
    _focus()

    # Typed for real, not inserted with JavaScript. WhatsApp's message box
    # is a React contenteditable: execCommand reported success and left the
    # box exactly as it was. Clicking it and typing is what a person does,
    # and it is the only thing the box accepts.
    where = _javascript(_BOX_POSITION)
    try:
        spot = json.loads(where or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("could not find the message box. Is a conversation open?") from exc
    if "x" not in spot:
        raise ToolError("could not find the message box. Is a conversation open?")

    pointer.click(float(spot["x"]), float(spot["y"]))
    _type('keystroke "a" using command down')
    _type(f"keystroke {_quote(draft)}")

    time.sleep(TYPING_SETTLE)
    now = " ".join(message_box_text().split())
    if now != draft:
        raise ToolError(
            f"the message box did not end up with the draft. It now holds: {now[:60]!r}"
        )
    return f"Put this in the message box, unsent, for you to check:\n{draft}"


TYPING_SETTLE = 0.6
"""Seconds to let the keystrokes arrive before checking what landed.

They are delivered by the window server, not by the call that sent them, so
reading the box straight away sees the text half typed.
"""

_BOX_POSITION = """
(() => {
  const main = document.querySelector('#main') || document.querySelector('[role=main]');
  if (!main) return '{}';
  const boxes = [...main.querySelectorAll('[contenteditable="true"][role="textbox"]')];
  const box = boxes[boxes.length - 1];
  if (!box) return '{}';
  const b = box.getBoundingClientRect();
  const zoom = window.innerWidth ? window.outerWidth / window.innerWidth : 1;
  return JSON.stringify({
    x: Math.round(window.screenX + (b.left + b.width / 2) * zoom),
    y: Math.round(window.screenY + (window.outerHeight - window.innerHeight * zoom)
         + (b.top + b.height / 2) * zoom)
  });
})()
"""


def open_conversation() -> str:
    """Who the open conversation is with, or empty if none is.

    Read before and after drafting. Between the two there is a model call
    that can take the better part of a minute, and every call here goes to
    whatever tab is in front at that moment: switch tabs mid-draft and one
    client's reply lands in another's message box, with no prompt because
    that box is empty.
    """
    _require_chrome()
    return _javascript(_WHO)


_WHO = """
(() => {
  const main = document.querySelector('#main') || document.querySelector('[role=main]');
  if (!main) return '';
  const header = main.querySelector('header');
  return header ? (header.innerText || '').split('\\n')[0].trim() : '';
})()
"""


@register(
    description=(
        "Send whatever is already in the WhatsApp message box. "
        "Only when the user has asked for it to be sent."
    ),
    safety="destructive",
)
def send_message(*, expected_recipient: str, expected_text: str) -> str:
    """Press Enter in the message box.

    ``destructive`` because it cannot be taken back: the message is on
    someone else's phone a moment later. Nothing else rolex does leaves the
    machine.
    """
    _require_chrome()

    waiting = message_box_text().strip()
    if not waiting:
        raise ToolError("there is nothing in the message box to send")

    to = open_conversation() or "this chat"

    _focus()
    where = _javascript(_BOX_POSITION)
    try:
        spot = json.loads(where or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("could not find the message box") from exc
    if "x" not in spot:
        raise ToolError("could not find the message box")

    pointer.click(float(spot["x"]), float(spot["y"]))
    # Focus and pointer operations can move the active page. Check after them,
    # immediately before pressing Enter, against the user's approved snapshot.
    if (
        not expected_recipient
        or open_conversation() != expected_recipient
        or message_box_text().strip() != expected_text
    ):
        raise ToolError("the chat or message changed; ask to send again and confirm the details")
    _type("key code 36")

    time.sleep(TYPING_SETTLE)
    left = message_box_text().strip()
    if left == waiting:
        raise ToolError(f"the message is still in the box; nothing was sent to {to}")

    return f"Sent to {to}: {waiting}"


def message_box_text() -> str:
    """Whatever is in the message box now.

    Used to notice a half-written message before replacing it: filling the
    box selects everything and types over it, and losing someone's
    half-finished sentence without asking is not acceptable.
    """
    _require_chrome()
    return _javascript(_READ_BOX)


_READ_BOX = """
(() => {
  const main = document.querySelector('#main') || document.querySelector('[role=main]');
  const scope = main || document;
  const boxes = [...scope.querySelectorAll('[contenteditable="true"][role="textbox"]')];
  const box = boxes[boxes.length - 1];
  return box ? (box.innerText || '').trim() : '';
})()
"""


@register(
    description=(
        "Open WhatsApp Web in Chrome. Reuse its tab by default. "
        "Set new_tab=true when asked to open a new WhatsApp tab."
    ),
    safety="mutating",
)
def open_whatsapp(new_tab: bool = False) -> str:
    """Open WhatsApp, optionally in a fresh tab.

    Args:
        new_tab: Open a new tab even if WhatsApp is already open.
    """
    from rolex.tools.browser import _osascript

    if new_tab:
        _osascript("""
tell application "Google Chrome"
    activate
    if (count of windows) is 0 then make new window
    make new tab at end of tabs of front window with properties {URL:"https://web.whatsapp.com/"}
end tell
""")
        return "Opened WhatsApp Web in a new Chrome tab."

    _osascript("""
tell application "Google Chrome"
    activate
    repeat with w in windows
        repeat with i from 1 to count of tabs of w
            if URL of tab i of w starts with "https://web.whatsapp.com/" then
                set active tab index of w to i
                set index of w to 1
                return
            end if
        end repeat
    end repeat
    if (count of windows) is 0 then make new window
    make new tab at end of tabs of front window with properties {URL:"https://web.whatsapp.com/"}
end tell
""")
    return "Opened WhatsApp Web in Chrome."


@register(
    description=(
        "Open a WhatsApp conversation by name, by searching for it. "
        "Use this instead of clicking when asked to open a chat or group."
    ),
    safety="mutating",
)
def open_whatsapp_chat(name: str) -> str:
    """Open a WhatsApp conversation by searching for it.

    Searching rather than hunting the list: the list is virtualised, so a
    chat far down it is not in the page at all until something scrolls to
    it, and searching is what a person does anyway.

    Args:
        name: The chat or group's name, as it appears in WhatsApp.
    """
    wanted = " ".join(name.split())
    if not wanted:
        raise ToolError("give the chat's name as it appears in WhatsApp")

    _require_chrome()
    if "web.whatsapp.com" not in _javascript("location.href"):
        raise ToolError("WhatsApp Web is not the page in front. Open it first.")

    # The click is a real one and goes wherever the pointer is told, so
    # Chrome has to be in front before it lands. Without this it was sent
    # to whatever window was, and the chat never opened.
    _focus()

    # WhatsApp's own search matches the phrase, so "MBRJ Self" finds
    # nothing when the chat is saved as "MBRJ | Self". Searching the
    # longest word narrows the list; the matching below picks from it.
    typed = _javascript(_SEARCH.replace("__WANTED__", _quote(_search_term(wanted))))
    if typed != "ok":
        raise ToolError(typed or "could not find WhatsApp's search box")

    # Cleared however this ends. A search left in place hides every other
    # chat from the next call, which is exactly what it is here to prevent.
    try:
        spot = _wait_for_result(wanted)
        pointer.click(float(spot["x"]), float(spot["y"]))
        opened = _wait_until_open(wanted)
    finally:
        _javascript(_CLEAR_SEARCH)
    if not opened:
        raise ToolError(
            f"searched for {wanted!r} but the conversation did not open. "
            f"It may be spelled differently in WhatsApp."
        )
    return f"Opened the WhatsApp chat with {wanted}."


def _search_term(wanted: str) -> str:
    """What to type into WhatsApp's search to narrow the list.

    The longest word rather than the whole phrase: punctuation and emoji in
    a chat's real name make a phrase search miss it, while one distinctive
    word brings it up.
    """
    words = [word for word in wanted.split() if len(word) > 2]
    return max(words, key=len) if words else wanted


def _wait_for_result(wanted: str) -> dict[str, Any]:
    """Where the matching search result is, once it has stopped moving.

    Results animate into place, and a click sent while they are still
    sliding lands on whatever was at that spot a moment earlier. So the
    position has to come back the same twice before it is used.
    """
    script = _FIND_RESULT.replace("__WANTED__", _quote(wanted))
    steady: dict[str, Any] | None = None

    for _ in range(SEARCH_TRIES):
        found = _javascript(script)
        if found and found != "waiting":
            try:
                spot: dict[str, Any] = json.loads(found)
            except json.JSONDecodeError as exc:
                # Escaping here would kill the session: the REPL catches
                # our own errors, not whatever the page happened to return.
                raise ToolError(f"could not read the search results: {found[:60]!r}") from exc
            if "error" in spot:
                raise ToolError(str(spot["error"]))
            if steady and steady["x"] == spot["x"] and steady["y"] == spot["y"]:
                return spot
            steady = spot
        time.sleep(POLL)

    if steady is not None:
        return steady
    raise ToolError(f"no WhatsApp chat called {wanted!r} came up in the search")


def _wait_until_open(wanted: str) -> bool:
    """Whether the conversation actually opened.

    Matched word by word, like the search: a chat saved as "MBRJ | Self" is
    asked for as "MBRJ Self", and a literal comparison calls a conversation
    that did open a failure.
    """
    words = [word.casefold() for word in wanted.split()]
    for _ in range(SEARCH_TRIES):
        header = _javascript(_WHO).casefold()
        if header and all(word in header for word in words):
            return True
        time.sleep(POLL)
    return False


SEARCH_TRIES = 24
"""How many times to look for the search result, and then for the chat to
open. With :data:`POLL` this is about six seconds each, which is generous
for a page that is only filtering a list it already has."""

POLL = 0.25
"""Seconds between looks."""

#: WhatsApp's search box. It used to be a contenteditable div and is now an
#: `input`; both are accepted, because it has changed once and will again.
_SEARCH_BOX = (
    'document.querySelector(\'#side input[role=textbox], #side [contenteditable="true"]'
    '[role=textbox], [aria-label*="Search" i]\')'
)

_SEARCH = (
    """
(() => {
  const box = """
    + _SEARCH_BOX
    + """;
  if (!box) return 'could not find the search box. Is WhatsApp Web signed in?';
  const wanted = __WANTED__;
  box.focus();

  if (box.tagName === 'INPUT' || box.tagName === 'TEXTAREA') {
    // React tracks the value itself and ignores a plain assignment, so the
    // prototype's setter is used and an input event announced.
    const proto = box.tagName === 'INPUT'
      ? window.HTMLInputElement.prototype : window.HTMLTextAreaElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(box, wanted);
    box.dispatchEvent(new Event('input', {bubbles: true}));
    return box.value === wanted ? 'ok' : 'the search box would not take the name';
  }

  document.execCommand('selectAll', false, null);
  document.execCommand('insertText', false, wanted);
  return box.innerText.trim() === wanted ? 'ok' : 'the search box would not take the name';
})()
"""
)

#: A search result matching the name, and where to click it. Exact first,
#: then one that merely contains it, so "testing ai" finds "testing ai"
#: before "testing ai archive".
_FIND_RESULT = """
(() => {
  const wanted = __WANTED__.toLowerCase();
  const rows = [...document.querySelectorAll('#pane-side [role=listitem], #pane-side [role=row]')]
    .filter(r => r.getClientRects().length);
  if (!rows.length) return 'waiting';

  const named = rows.map(row => {
    const titled = row.querySelector('[title]');
    const label = ((titled && titled.title) || row.innerText || '')
      .replace(/\\s+/g, ' ').trim();
    return [row, label];
  }).filter(([, label]) => label);

  // Exact, then containing, then every word present in any order. The
  // last one matters: a chat saved as "MBRJ | Self (emoji)" is asked for
  // as "MBRJ Self", and WhatsApp's own search finds nothing for that.
  const words = wanted.split(/\\s+/).filter(Boolean);
  const exact = named.filter(([, label]) => label.toLowerCase() === wanted);
  const contains = named.filter(([, label]) => label.toLowerCase().includes(wanted));
  const allWords = named.filter(([, label]) => {
    const lower = label.toLowerCase();
    return words.every(word => lower.includes(word));
  });
  const hits = exact.length ? exact : contains.length ? contains : allWords;
  if (!hits.length) return 'waiting';

  const [row, label] = hits[0];
  // Only scrolled if it is actually off screen. A search result is at the
  // top already, and scrolling a virtualised list re-renders it, so the
  // coordinates would describe where the row used to be.
  let box = row.getBoundingClientRect();
  if (box.top < 0 || box.bottom > window.innerHeight) {
    row.scrollIntoView({block: 'center', behavior: 'instant'});
    box = row.getBoundingClientRect();
  }
  const zoom = window.innerWidth ? window.outerWidth / window.innerWidth : 1;
  return JSON.stringify({
    x: Math.round(window.screenX + (box.left + box.width / 2) * zoom),
    y: Math.round(window.screenY + (window.outerHeight - window.innerHeight * zoom)
         + (box.top + box.height / 2) * zoom),
    label: label.slice(0, 60)
  });
})()
"""

#: Leaving a search in place hides every other chat from the next listing.
_CLEAR_SEARCH = (
    """
(() => {
  const box = """
    + _SEARCH_BOX
    + """;
  if (!box) return '';
  if (box.tagName === 'INPUT' || box.tagName === 'TEXTAREA') {
    const proto = window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(box, '');
    box.dispatchEvent(new Event('input', {bubbles: true}));
    return '';
  }
  box.focus();
  document.execCommand('selectAll', false, null);
  document.execCommand('delete', false, null);
  return '';
})()
"""
)
