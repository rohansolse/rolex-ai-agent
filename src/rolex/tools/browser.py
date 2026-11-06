"""Seeing and acting on the page in Chrome.

Opening a page and walking away is not much of an assistant. These let rolex
read what is on screen and act on it, which is the difference between "I have
opened WhatsApp for you" and actually opening a conversation.

Everything goes through ``osascript``. Chrome accepts JavaScript over Apple
Events, so no driver, no debugging port and no Python dependency is needed —
but the user has to switch it on once, under View -> Developer -> Allow
JavaScript from Apple Events.

Safety: looking changes nothing, so :func:`read_page`, :func:`page_buttons`
and :func:`scroll_page` run unasked. :func:`click_text` and :func:`type_text`
are ``mutating`` and asked about every time — one wrong click sends a message
to the wrong person. :func:`press_key` is ``destructive``, because the key
people ask for is Enter, and Enter is what sends the message.
"""

import json
import shutil
import subprocess

from rolex.core.errors import ToolError
from rolex.tools.registry import register

BROWSER = "Google Chrome"

TIMEOUT = 20.0
"""Seconds to wait for AppleScript. A page running heavy JavaScript is slow,
but nothing legitimate here takes longer."""

PAGE_LIMIT = 4_000
"""Characters of page text to hand the model.

A 3B model's context is small, and a chat page can run to tens of thousands
of characters. Better a readable first screenful than a truncated flood that
pushes the system prompt out.
"""


def _osascript(script: str) -> str:
    """Run an AppleScript and return what it printed.

    Raises:
        ToolError: osascript is missing, Chrome is not running, or the script
            failed.
    """
    executable = shutil.which("osascript")
    if executable is None:
        raise ToolError("this only works on macOS: osascript is not available")

    try:
        result = subprocess.run(
            [executable, "-e", script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"{BROWSER} did not respond within {TIMEOUT:.0f} seconds") from exc

    if result.returncode != 0:
        raise ToolError(_explain(result.stderr.strip()))

    return result.stdout.strip()


def _explain(stderr: str) -> str:
    """Turn AppleScript's complaint into something a user can act on."""
    lowered = stderr.lower()

    if "not allowed" in lowered or "apple events" in lowered or "-1743" in lowered:
        return (
            f"macOS has not been told this is allowed. Give the terminal permission under "
            f"System Settings -> Privacy & Security -> Automation, and switch on "
            f"{BROWSER}'s View -> Developer -> Allow JavaScript from Apple Events."
        )
    if "javascript" in lowered and "disabled" in lowered:
        return (
            f"{BROWSER} will not run JavaScript from scripts. Switch on "
            f"View -> Developer -> Allow JavaScript from Apple Events."
        )
    if "can't get window" in lowered or "invalid index" in lowered:
        return f"{BROWSER} has no window open."
    if "application isn" in lowered or "can't get application" in lowered:
        return f"{BROWSER} is not running. Ask to open it first."
    return stderr or "the browser did not do what was asked"


#: What AppleScript prints when JavaScript returned undefined or null.
MISSING = "missing value"


def _javascript(source: str) -> str:
    """Run JavaScript in Chrome's active tab and return its result as text.

    JavaScript that returns nothing comes back as AppleScript's "missing
    value", which would otherwise reach the model as if the page had said
    those words.
    """
    quoted = json.dumps(source)
    result = _osascript(
        f'tell application "{BROWSER}" to execute active tab of front window javascript {quoted}'
    )
    return "" if result == MISSING else result


def _require_chrome() -> None:
    """Fail with something useful if Chrome is not running."""
    running = _osascript(
        f'tell application "System Events" to (name of processes) contains "{BROWSER}"'
    )
    if running != "true":
        raise ToolError(f"{BROWSER} is not running. Ask to open it first.")


# --- looking -------------------------------------------------------------


@register(description="Read the text of the web page currently open in Chrome.")
def read_page() -> str:
    """What is on the page, so the model can decide what to do with it."""
    _require_chrome()
    # One round-trip, not three: the active tab can change between calls, and
    # a report that straddles a switch describes no page that ever existed.
    raw = _javascript(
        "(() => JSON.stringify({"
        "  title: document.title,"
        "  url: location.href,"
        f"  text: document.body.innerText.slice(0, {PAGE_LIMIT})"
        "}))()"
    )
    try:
        page = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"could not read the page: {raw[:100]!r}") from exc

    body = " ".join(str(page.get("text", "")).split()) or "(the page has no readable text)"
    return f"Page: {page.get('title', '')}\nURL: {page.get('url', '')}\n\n{body}"


@register(description="List the things that can be clicked on the page in Chrome.")
def page_buttons() -> str:
    """The clickable things, by their visible text.

    Named by what the user would say — "the MBRJ group", "Send" — because
    that is what they will ask to be clicked.
    """
    _require_chrome()
    found = _javascript(_LIST_CLICKABLE)
    if not found:
        return "Nothing clickable was found on this page."
    return f"Clickable on this page:\n{found}"


#: Shared by listing and clicking, so the two can never disagree about what
#: is on the page. A non-zero box is not enough on its own: a closed menu
#: keeps its size while being `visibility: hidden` or fully transparent, and
#: offering those for clicking means clicking something nobody can see.
_CLICKABLE_JS = """
  const SELECTOR =
    'a, button, [role=button], [role=listitem], [role=link], [role=row], input[type=submit]';

  const visible = (node) => {
    const box = node.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return false;
    if (box.bottom < 0 || box.right < 0) return false;
    const style = getComputedStyle(node);
    return style.visibility !== 'hidden' && style.display !== 'none'
      && Number(style.opacity) > 0;
  };

  const labelOf = (node) =>
    (node.innerText || node.value || node.getAttribute('aria-label') || '')
      .replace(/\\s+/g, ' ').trim();
"""

_LIST_CLICKABLE = (
    """
(() => {
"""
    + _CLICKABLE_JS
    + """
  const seen = new Set();
  const out = [];
  for (const node of document.querySelectorAll(SELECTOR)) {
    if (!visible(node)) continue;
    const label = labelOf(node).slice(0, 80);
    if (!label || seen.has(label)) continue;
    seen.add(label);
    out.push('- ' + label);
    if (out.length >= 40) break;
  }
  return out.join('\\n');
})()
"""
)


# --- acting --------------------------------------------------------------


@register(
    description="Click something on the page in Chrome, by the text shown on it.",
    safety="mutating",
)
def click_text(text: str) -> str:
    """Click the first visible thing whose text contains `text`.

    Args:
        text: What is written on it, as the user would read it aloud.
    """
    wanted = " ".join(text.split())
    if not wanted:
        raise ToolError("nothing was given to click on")

    _require_chrome()
    result = _javascript(_CLICK.replace("__WANTED__", json.dumps(wanted)))

    if result.startswith("missing:"):
        raise ToolError(
            f"nothing on the page says {wanted!r}. Use page_buttons to see what is there."
        )
    return f"Clicked {result or wanted}."


#: Exact label first, then one that starts with it, and only then one that
#: merely contains it. Asked for "Ram", taking the first substring match in
#: DOM order clicks "Rama Sharma" — a different person.
_CLICK = (
    """
(() => {
"""
    + _CLICKABLE_JS
    + """
  const wanted = __WANTED__.toLowerCase();
  const candidates = [];
  for (const node of document.querySelectorAll(SELECTOR)) {
    if (!visible(node)) continue;
    const label = labelOf(node);
    if (label) candidates.push([node, label, label.toLowerCase()]);
  }

  const tests = [
    ([, , lower]) => lower === wanted,
    ([, , lower]) => lower.startsWith(wanted),
    ([, , lower]) => lower.includes(wanted),
  ];

  for (const test of tests) {
    const found = candidates.filter(test);
    if (found.length === 0) continue;
    const [node, label] = found[0];
    node.scrollIntoView({block: 'center'});
    node.click();
    return label.slice(0, 80);
  }
  return 'missing:' + wanted;
})()
"""
)


@register(description="Scroll the page in Chrome up or down.")
def scroll_page(direction: str = "down", screens: float = 1.0) -> str:
    """Scroll by whole screenfuls.

    Args:
        direction: "up", "down", "top" or "bottom".
        screens: How many screenfuls, for up and down.
    """
    way = direction.strip().lower()
    if way not in {"up", "down", "top", "bottom"}:
        raise ToolError(f"direction must be up, down, top or bottom; got {direction!r}")

    # A small model sends "2" as often as 2. A TypeError here would reach it
    # as something it cannot act on.
    try:
        distance = float(screens)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"screens must be a number, got {screens!r}") from exc
    if distance <= 0:
        raise ToolError(f"screens must be positive, got {screens!r}")

    _require_chrome()

    if way == "top":
        _javascript("window.scrollTo({top: 0}); 'done'")
        return "Scrolled to the top of the page."
    if way == "bottom":
        _javascript("window.scrollTo({top: document.body.scrollHeight}); 'done'")
        return "Scrolled to the bottom of the page."

    sign = "" if way == "down" else "-"
    _javascript(f"window.scrollBy(0, {sign}window.innerHeight * {distance}); 'done'")
    return f"Scrolled {way} {distance:g} screen{'s' if distance != 1 else ''}."


@register(
    description="Type text into whatever is focused in Chrome, such as a message box.",
    safety="mutating",
)
def type_text(text: str) -> str:
    """Type `text` as if on the keyboard.

    Keystrokes rather than setting a value directly: a chat box built in
    React ignores a value it was handed and only notices real typing.

    Args:
        text: What to type.
    """
    typed = text.strip()
    if not typed:
        raise ToolError("nothing was given to type")

    # A newline is a Return, and in a chat box Return sends. Typing a
    # two-line message would post the first line and report both as typed.
    if "\n" in typed or "\r" in typed:
        raise ToolError(
            "this types one line at a time. A line break would press Return, which "
            "sends the message. Type each line, then use press_key('enter') on purpose."
        )

    _require_chrome()
    _type(f"keystroke {json.dumps(typed)}")
    return f"Typed {typed!r}."


@register(
    description="Press a single key in Chrome, such as Enter, Escape or Tab.",
    safety="destructive",
)
def press_key(key: str) -> str:
    """Press one key.

    ``destructive``, which reads oddly for a keypress until you notice that
    the key people ask for is Enter, and in a chat box Enter is what sends
    the message. Sending something outward is exactly what that class is
    for, so it stays off unless the user turns it on.

    Args:
        key: enter, escape, tab, space, up, down, left, right, delete.
    """
    name = key.strip().lower()
    code = _KEY_CODES.get(name)
    if code is None:
        raise ToolError(f"unknown key {key!r}. Known: {', '.join(sorted(_KEY_CODES))}")

    _require_chrome()
    _type(f"key code {code}")
    return f"Pressed {name}."


FOCUS_SETTLE = 0.2
"""Seconds to let the window come forward before typing into it.

Bringing Chrome to the front and typing are two separate things to macOS, and
without a pause the first keystroke can land in whatever was focused before —
or be lost."""


def _type(action: str) -> None:
    """Bring Chrome forward and send one keyboard action to it.

    Both in a single script: run as two, the window can still be coming
    forward when the keys arrive.
    """
    _osascript(
        f'tell application "{BROWSER}" to activate\n'
        f"delay {FOCUS_SETTLE}\n"
        f'tell application "System Events" to {action}'
    )


#: macOS virtual key codes for the keys worth naming aloud.
_KEY_CODES = {
    "enter": 36,
    "return": 36,
    "escape": 53,
    "esc": 53,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
}
