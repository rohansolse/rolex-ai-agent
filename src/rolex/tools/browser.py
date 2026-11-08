"""Seeing and acting on the page in Chrome.

Opening a page and walking away is not much of an assistant. These let rolex
read what is on screen and act on it, which is the difference between "I have
opened WhatsApp for you" and actually opening a conversation.

Everything goes through ``osascript``. Chrome accepts JavaScript over Apple
Events, so no driver, no debugging port and no Python dependency is needed —
but the user has to switch it on once, under View -> Developer -> Allow
JavaScript from Apple Events.

Safety: looking changes nothing, so :func:`read_page`, :func:`page_buttons`
and :func:`scroll_page` run unasked. :func:`click_text`, :func:`click_item` and
:func:`type_text` are ``mutating`` and asked about every time — one wrong click sends a message
to the wrong person. :func:`press_key` is ``destructive``, because the key
people ask for is Enter, and Enter is what sends the message.
"""

import json
import shutil
import subprocess
from typing import Any

from rolex.core.errors import ToolError
from rolex.tools import pointer
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


def _quote(text: str) -> str:
    """`text` as an AppleScript string literal.

    ``ensure_ascii=False`` matters: JSON would otherwise write a non-ASCII
    character as ``\\uXXXX``, which AppleScript does not understand and
    rejects as a syntax error. One em dash in a JavaScript comment broke
    every click until this was found.
    """
    return json.dumps(text, ensure_ascii=False)


def _javascript(source: str) -> str:
    """Run JavaScript in Chrome's active tab and return its result as text.

    JavaScript that returns nothing comes back as AppleScript's "missing
    value", which would otherwise reach the model as if the page had said
    those words.
    """
    quoted = _quote(source)
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


@register(
    description=(
        "List the things that can be clicked on the page in Chrome, numbered. "
        "Use this before clicking when the page has several similar items."
    )
)
def page_buttons() -> str:
    """The clickable things, numbered, by their visible text.

    Numbered because text alone is often ambiguous: asked to click "Rohan
    Solse" on WhatsApp Web, the words appear both in the conversation list
    and inside the open conversation, and guessing picks the wrong one. With
    numbers the model can look first and then say exactly which.
    """
    _require_chrome()
    raw = _javascript(_LIST_CLICKABLE)

    # One round trip: a second call for the title could straddle a tab
    # change and head the listing with a different page's name.
    try:
        found = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("could not read what is on this page") from exc

    where = str(found.get("page") or "this page")
    key = str(found.get("key") or where)
    listing = str(found.get("items") or "")

    if not listing:
        _remember(key, [])
        return f"Nothing clickable was found on {where}."

    lines = listing.splitlines()
    marked = _mark_new(key, lines)
    _remember(key, lines)

    return f"Clickable on {where} (click_item takes the number):\n" + "\n".join(marked)


#: The last listing, keyed by the page it came from, so a later one can say
#: what is new. A click usually changes the page, and "what appeared?" is
#: the question that follows. Keyed by page because otherwise listing a
#: second page marks every one of its items as new.
#:
#: None means nothing has been listed for that page yet, which is not the
#: same as having listed it and found nothing.
_PREVIOUS: dict[str, list[str]] = {}


def _label_of(line: str) -> str:
    """A listed line without its number, which changes as things appear."""
    _, _, rest = line.partition(". ")
    return rest.strip()


def _mark_new(page: str, lines: list[str]) -> list[str]:
    """The same lines, with a star against anything that was not there before."""
    previous = _PREVIOUS.get(page)
    if previous is None:
        return lines

    before = {_label_of(line) for line in previous}
    return [f"{line}  *new" if _label_of(line) not in before else line for line in lines]


def _remember(page: str, lines: list[str]) -> None:
    _PREVIOUS[page] = lines


def forget_listings() -> None:
    """Drop what has been listed.

    For tests, which would otherwise leak into each other and see stray
    *new markers depending on the order they ran in.
    """
    _PREVIOUS.clear()


#: Shared by listing and clicking, so the two can never disagree about what
#: is on the page. A non-zero box is not enough on its own: a closed menu
#: keeps its size while being `visibility: hidden` or fully transparent, and
#: offering those for clicking means clicking something nobody can see.
_CLICKABLE_JS = """
  const LIMIT = 40;
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

  // Where on the page a thing sits. "Rohan Solse" appears both in a chat
  // list and inside the open conversation, and a name alone cannot say
  // which was meant. Naming the region lets the model, and the user
  // reading the list, tell them apart.
  const regionOf = (node) => {
    if (node.closest('[role=dialog], dialog')) return 'dialog';
    if (node.closest('#pane-side, [role=complementary], aside')) return 'list';
    if (node.closest('#main, main, [role=main]')) return 'conversation';
    if (node.closest('header, [role=banner]')) return 'header';
    if (node.closest('nav, [role=navigation]')) return 'nav';
    return 'page';
  };

  // What kind of thing it is, in words a person would use.
  const kindOf = (node) => {
    const role = (node.getAttribute('role') || '').toLowerCase();
    if (role === 'listitem' || role === 'row') return 'item';
    if (role === 'link' || node.tagName === 'A') return 'link';
    // Only input[type=submit] is ever listed, and calling it a field makes
    // the model try to type into it.
    if (node.tagName === 'INPUT') return 'button';
    return 'button';
  };

  const describe = (node) => regionOf(node) + '/' + kindOf(node);

  // Where on the screen a node sits, so a real click can be sent to it.
  //
  // Synthetic DOM events were tried first and are not enough: WhatsApp Web
  // reported every click as successful and opened nothing. It listens for
  // genuine input, so rolex sends a genuine click through macOS and
  // JavaScript only says where to aim. Coordinates are read after
  // scrolling, because scrolling moves them.
  const whereIs = (node) => {
    // 'instant': with scroll-behavior: smooth the page is still moving when
    // the rectangle is read, and the click lands on whatever slid into that
    // spot instead.
    node.scrollIntoView({block: 'center', behavior: 'instant'});
    const box = node.getBoundingClientRect();

    // Page zoom scales CSS pixels but not screenX, screenY or outerHeight,
    // so the two have to be brought into the same units before adding. At
    // 150% zoom, ignoring this puts the click a long way from the target
    // and still calls it a success.
    const zoom = window.innerWidth ? window.outerWidth / window.innerWidth : 1;
    const chromeHeight = window.outerHeight - window.innerHeight * zoom;

    return {
      x: Math.round(window.screenX + (box.left + box.width / 2) * zoom),
      y: Math.round(window.screenY + chromeHeight + (box.top + box.height / 2) * zoom),
    };
  };
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
    const where = describe(node);
    // Keyed on region as well as words: the same name in a sidebar and in
    // the open conversation are two different things, and collapsing them
    // is exactly the ambiguity the region was added to remove.
    const key = where + '|' + label;
    if (!label || seen.has(key)) continue;
    seen.add(key);
    out.push((out.length + 1) + '. [' + where + '] ' + label);
    if (out.length >= LIMIT) break;
  }
  // Keyed on the address, not the title: WhatsApp puts the unread count
  // in its title, so "(1) WhatsApp" and "WhatsApp" would be two pages and
  // the "what is new?" comparison would reset exactly when something new
  // arrived.
  return JSON.stringify({
    page: document.title || location.host,
    key: location.origin + location.pathname,
    items: out.join('\\n')
  });
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
    _focus()

    # Located and clicked in two steps, because the click is a real one: the
    # page says where, and macOS does the clicking. Synthetic DOM events were
    # tried first and WhatsApp Web ignored every one of them while reporting
    # success.
    found = _javascript(_CLICK.replace("__WANTED__", _quote(wanted)))
    spot = _position(found, repr(wanted))
    pointer.click(float(spot["x"]), float(spot["y"]))
    return f"Clicked {spot.get('label') or wanted}."


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
    const at = whereIs(node);
    return JSON.stringify({x: at.x, y: at.y, label: label.slice(0, 60)});
  }
  return JSON.stringify({missing: wanted});
})()
"""
)


@register(
    description=(
        "Click a numbered item from page_buttons. Use this instead of click_text "
        "when several things on the page say something similar."
    ),
    safety="mutating",
)
def click_item(number: int, expected: str = "") -> str:
    """Click the thing page_buttons listed as `number`.

    Args:
        number: Its number in the most recent page_buttons listing.
        expected: What that line said, if known. Numbers go stale the moment
            a click or a scroll changes what is visible, and clicking the
            wrong conversation and then typing into it sends a message to
            the wrong person. Given this, rolex checks before clicking.
    """
    try:
        wanted = int(number)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"number must be a whole number, got {number!r}") from exc
    if wanted < 1:
        raise ToolError(f"numbering starts at 1, got {wanted}")

    _require_chrome()
    _focus()

    found = _javascript(_LOCATE_NTH.replace("__INDEX__", str(wanted - 1)))
    spot = _position(found, f"item {wanted}")
    label = str(spot.get("label") or "")

    # The listed line carries a "[region/kind]" prefix, a number, and
    # sometimes a "*new" marker. A model passing back "what that line said"
    # includes them, so they are stripped before comparing — otherwise
    # every check fails and the tool becomes unusable.
    if not _same_thing(expected, label):
        raise ToolError(
            f"item {wanted} is now {label!r}, not {expected!r}. The page has changed; "
            f"call page_buttons again and use the new number."
        )

    pointer.click(float(spot["x"]), float(spot["y"]))
    return f"Clicked {label or f'item {wanted}'}."


def _focus() -> None:
    """Bring Chrome forward and wait for it to get there.

    The click is a real one and goes wherever the pointer is told, so if
    Chrome is still behind the terminal when it lands, the terminal gets
    clicked instead.
    """
    _osascript(f'tell application "{BROWSER}" to activate\ndelay {FOCUS_SETTLE}')


def _same_thing(expected: str, label: str) -> bool:
    """Whether a label the caller remembered still names this item."""
    wanted = _bare(expected)
    return not wanted or wanted in _bare(label)


def _bare(line: str) -> str:
    """A listed line stripped of its number, region and "new" marker."""
    text = line.strip()
    number, dot, rest = text.partition(". ")
    if dot and number.strip().isdigit():
        text = rest
    if text.startswith("["):
        _, closed, rest = text.partition("] ")
        if closed:
            text = rest
    return " ".join(text.removesuffix("*new").split()).casefold().strip()


def _position(found: str, what: str) -> dict[str, Any]:
    """Read where something is, or say it was not there."""
    try:
        parsed: object = json.loads(found)
    except json.JSONDecodeError as exc:
        raise ToolError(f"could not look for {what} on the page") from exc

    if not isinstance(parsed, dict):
        raise ToolError(f"could not look for {what} on the page")
    if "missing" in parsed:
        raise ToolError(f"there is no {what} on this page. Use page_buttons to see what is there.")
    return parsed


_LOCATE_NTH = (
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
    const key = describe(node) + '|' + label;
    if (!label || seen.has(key)) continue;
    seen.add(key);
    out.push([node, '[' + describe(node) + '] ' + label]);
    if (out.length >= LIMIT) break;
  }
  const chosen = out[__INDEX__];
  if (!chosen) return JSON.stringify({missing: __INDEX__});
  const at = whereIs(chosen[0]);
  return JSON.stringify({x: at.x, y: at.y, label: chosen[1].slice(0, 60)});
})()
"""
)


@register(description="List the tabs open in Chrome, numbered.")
def list_tabs() -> str:
    """Every tab in the front Chrome window, with the open one marked."""
    _require_chrome()
    # AppleScript has no inline conditional, so the marker is built with a
    # statement rather than an expression.
    listing = _osascript(
        "set out to {}\n"
        f'tell application "{BROWSER}" to tell front window\n'
        "  set open_one to active tab index\n"
        "  repeat with i from 1 to count tabs\n"
        '    set mark to ""\n'
        '    if i = open_one then set mark to "[open] "\n'
        '    set end of out to (i as text) & ". " & mark & (title of tab i)\n'
        "  end repeat\n"
        "end tell\n"
        # Set outside the Chrome block: `text item delimiters` belongs to
        # AppleScript itself, and Chrome rejects being asked for it.
        "set AppleScript's text item delimiters to linefeed\n"
        "return ((count of out) as text) & linefeed & (out as text)"
    )
    if not listing:
        return "No tabs are open."
    total_text, _, rows = listing.partition("\n")
    try:
        total = int(total_text)
        if total < 0:
            raise ValueError("negative tab count")
    except ValueError as exc:
        raise ToolError("Chrome returned an invalid tab count; try listing tabs again") from exc
    if total == 0:
        return "No tabs are open."
    return (
        f"Tabs in the front Chrome window: {total} "
        f"(close_tab and switch_tab take the number):\n{rows}"
    )


@register(
    description="Close a Chrome tab by its number from list_tabs.",
    safety="mutating",
)
def close_tab(number: int, expected_title: str = "") -> str:
    """Close one tab.

    Args:
        number: Its number in the list_tabs listing.
        expected_title: Title shown when the user approved closing this tab.
    """
    wanted, title = _tab(number)
    approved_title = expected_title or title
    if title != approved_title:
        raise ToolError("the tab changed; list tabs and confirm again")
    status = _osascript(
        f'tell application "{BROWSER}"\n'
        '  if (count of windows) is 0 then return "changed"\n'
        "  tell front window\n"
        f'    if (count of tabs) < {wanted} then return "changed"\n'
        f"    set target_tab to tab {wanted}\n"
        f'    if title of target_tab is not {_quote(approved_title)} then return "changed"\n'
        "    set target_id to id of target_tab\n"
        "    close target_tab\n"
        "  end tell\n"
        "  repeat with w in windows\n"
        "    repeat with t in tabs of w\n"
        '      if id of t = target_id then return "still_open"\n'
        "    end repeat\n"
        "  end repeat\n"
        '  return "closed"\n'
        "end tell"
    )
    if status != "closed":
        raise ToolError("Chrome did not confirm the tab was closed; list tabs and try again")
    return f"Closed the tab showing {title!r}."


@register(
    description="Switch to a Chrome tab by its number from list_tabs.",
    safety="mutating",
)
def switch_tab(number: int) -> str:
    """Bring one tab to the front.

    Args:
        number: Its number in the list_tabs listing.
    """
    wanted, title = _tab(number)
    _osascript(f'tell application "{BROWSER}" to set active tab index of front window to {wanted}')
    return f"Switched to {title!r}."


def _tab(number: int) -> tuple[int, str]:
    """Check a tab number and return it with that tab's title.

    Checked rather than trusted: a number from an earlier listing can point
    at a different tab once one has been closed, and closing the wrong tab
    loses whatever was in it.
    """
    try:
        wanted = int(number)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"number must be a whole number, got {number!r}") from exc
    if wanted < 1:
        raise ToolError(f"tabs are numbered from 1, got {wanted}")

    _require_chrome()
    count = _osascript(f'tell application "{BROWSER}" to count tabs of front window')
    try:
        total = int(count)
    except ValueError as exc:
        raise ToolError(f"{BROWSER} has no window open") from exc

    if wanted > total:
        raise ToolError(f"there is no tab {wanted}; Chrome has {total}. Use list_tabs to see them.")

    title = _osascript(f'tell application "{BROWSER}" to title of tab {wanted} of front window')
    return wanted, title


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
    _type(f"keystroke {_quote(typed)}")
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
