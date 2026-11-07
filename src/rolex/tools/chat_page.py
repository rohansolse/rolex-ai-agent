"""Reading a conversation, and putting a draft into its message box.

Written for WhatsApp Web, which is what this is used for, but nothing here
names it: the selectors are the ones any chat application built with ARIA
roles uses, and the fallbacks are generic.

Separate from :mod:`rolex.tools.browser` because the job is different. That
module is about a page in general; this is about the one thing rolex is
asked to do most — read what a client said and help answer it.
"""

import json

from rolex.core.errors import ToolError
from rolex.tools.browser import _javascript, _quote, _require_chrome
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
    result = _javascript(_FILL_BOX.replace("__TEXT__", _quote(draft)))

    if result != "ok":
        raise ToolError(result or "could not find the message box. Is a conversation open?")
    return f"Put this in the message box, unsent, for you to check:\n{draft}"


_FILL_BOX = """
(() => {
  const main = document.querySelector('#main') || document.querySelector('[role=main]');
  const scope = main || document;
  const boxes = [...scope.querySelectorAll('[contenteditable="true"][role="textbox"]')];

  // The last one: a chat page often has a search box too, and the message
  // box is below it.
  const box = boxes[boxes.length - 1];
  if (!box) return 'no message box on this page';

  box.focus();

  // execCommand rather than setting textContent: a box built in React
  // ignores text it was handed and only notices an edit it believes a
  // person made.
  document.execCommand('selectAll', false, null);
  document.execCommand('insertText', false, __TEXT__);

  // Compared against what was asked for, not merely non-empty: if
  // insertText is refused the old text is still there and this would
  // otherwise report success over someone else's words.
  const now = (box.innerText || '').replace(/\\s+/g, ' ').trim();
  const wanted = __TEXT__.replace(/\\s+/g, ' ').trim();
  return now === wanted ? 'ok' : 'the message box would not take the text';
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
    description="Open WhatsApp Web in Chrome. Reuse its tab when available.", safety="mutating"
)
def open_whatsapp() -> str:
    from rolex.tools.browser import _osascript

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
    description="Search WhatsApp Web and open a chat by exact contact name.", safety="mutating"
)
def open_whatsapp_chat(name: str) -> str:
    import time

    if not name.strip():
        raise ToolError("Give the contact's name as it appears in WhatsApp.")
    guard = "if (location.origin !== 'https://web.whatsapp.com') return 'Open WhatsApp Web first';"
    search = """
      const box = document.querySelector('#side [contenteditable=true][role=textbox]');
      if (!box) return 'Open WhatsApp Web and sign in first';
      box.focus();
      document.execCommand('selectAll', false, null);
      document.execCommand('insertText', false, wanted);
      return box.innerText.trim() === wanted ? 'ok' : 'Search field did not accept the name';
    """
    prefix = "(() => {" + guard + "const wanted = " + _quote(name.strip()) + ";"
    result = _javascript(prefix + search + "})()")
    if result != "ok":
        raise ToolError(result)
    select = """
      const hits = [...document.querySelectorAll('#pane-side [title]')].filter(
        el => el.title === wanted && el.getClientRects().length);
      if (hits.length > 1) return 'Multiple chats match. Select the correct one manually.';
      if (!hits.length) return 'waiting';
      hits[0].click();
      return 'ok';
    """
    verify = """
      const header = document.querySelector('#main header');
      return header && [...header.querySelectorAll('[title]')].some(el => el.title === wanted)
        ? 'ok' : 'waiting';
    """
    for script in (select, verify):
        for _ in range(20):
            result = _javascript(prefix + script + "})()")
            if result == "ok":
                break
            if result != "waiting":
                raise ToolError(result)
            time.sleep(0.2)
        else:
            raise ToolError(f"Could not verify a WhatsApp chat named {name}.")
    return f"Opened the WhatsApp chat with {name}."
