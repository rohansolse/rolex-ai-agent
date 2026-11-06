"""The tools that read and act on the page in Chrome.

``osascript`` is never actually run: a fake records the script it was handed.
What is tested is the script we build and how its result is read, which is
where the mistakes are. Nothing opens, clicks or types on the machine running
the suite.
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from rolex.core.errors import ToolError
from rolex.tools import browser, get


@dataclass
class FakeRun:
    """Stands in for subprocess.run, answering scripts from a script."""

    answers: list[str] = field(default_factory=list)
    returncode: int = 0
    stderr: str = ""
    raises: Exception | None = None
    scripts: list[str] = field(default_factory=list)

    def __call__(self, argv: list[str], **_: Any) -> "FakeRun":
        self.scripts.append(argv[-1])
        if self.raises:
            raise self.raises
        self.stdout = self.answers.pop(0) if self.answers else "true"
        return self

    stdout: str = "true"

    @property
    def javascript(self) -> list[str]:
        """Just the JavaScript sent, pulled back out of the AppleScript."""
        found = []
        for script in self.scripts:
            _, sep, tail = script.partition("javascript ")
            if sep:
                found.append(json.loads(tail))
        return found


@pytest.fixture
def chrome(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr("subprocess.run", fake)
    return fake


# --- reading --------------------------------------------------------------


def page(title: str = "My Page", url: str = "https://example.com", text: str = "Hello") -> str:
    return json.dumps({"title": title, "url": url, "text": text})


def test_read_page_reports_title_url_and_text(chrome: FakeRun) -> None:
    chrome.answers = ["true", page(text="Hello   there\n\nfriend")]

    report = browser.read_page()

    assert "My Page" in report
    assert "https://example.com" in report
    assert "Hello there friend" in report


def test_the_page_is_read_in_one_round_trip(chrome: FakeRun) -> None:
    """The active tab can change between calls, and a report stitched from
    several describes no page that ever existed."""
    chrome.answers = ["true", page()]
    browser.read_page()
    assert len(chrome.javascript) == 1


def test_an_unreadable_answer_is_reported(chrome: FakeRun) -> None:
    chrome.answers = ["true", "not json at all"]
    with pytest.raises(ToolError, match="could not read the page"):
        browser.read_page()


def test_read_page_is_capped(chrome: FakeRun) -> None:
    """A chat page runs to tens of thousands of characters, and a 3B model's
    context does not."""
    chrome.answers = ["true", page()]
    browser.read_page()
    assert f"slice(0, {browser.PAGE_LIMIT})" in chrome.javascript[-1]


def test_a_page_with_no_text_says_so(chrome: FakeRun) -> None:
    chrome.answers = ["true", page(text="")]
    assert "no readable text" in browser.read_page()


def test_page_buttons_lists_what_was_found(chrome: FakeRun) -> None:
    chrome.answers = ["true", "- MBRJ self group\n- Send"]
    listed = browser.page_buttons()
    assert "MBRJ self group" in listed
    assert "Send" in listed


def test_page_buttons_says_so_when_there_is_nothing(chrome: FakeRun) -> None:
    chrome.answers = ["true", ""]
    assert "Nothing clickable" in browser.page_buttons()


def test_reading_never_asks_permission() -> None:
    """Looking at a page cannot do any harm."""
    assert get("read_page").safety == "read_only"
    assert get("page_buttons").safety == "read_only"


# --- clicking -------------------------------------------------------------


def test_clicking_sends_the_wanted_text(chrome: FakeRun) -> None:
    chrome.answers = ["true", "MBRJ self group"]
    assert browser.click_text("MBRJ self group") == "Clicked MBRJ self group."
    assert json.dumps("mbrj self group") in chrome.javascript[-1].lower()


def test_clicking_something_absent_says_what_to_do(chrome: FakeRun) -> None:
    chrome.answers = ["true", "missing:nowhere"]
    with pytest.raises(ToolError, match="page_buttons"):
        browser.click_text("nowhere")


def test_clicking_nothing_is_refused(chrome: FakeRun) -> None:
    with pytest.raises(ToolError, match="nothing was given to click"):
        browser.click_text("   ")
    assert chrome.scripts == []


def test_clicking_asks_permission_every_time() -> None:
    """One wrong click sends a message to the wrong person."""
    assert get("click_text").safety == "mutating"


# --- scrolling ------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("top", "scrollTo({top: 0})"), ("bottom", "document.body.scrollHeight")],
)
def test_scrolling_to_an_end(chrome: FakeRun, direction: str, expected: str) -> None:
    browser.scroll_page(direction)
    assert expected in chrome.javascript[-1]


def test_scrolling_down_moves_a_screenful(chrome: FakeRun) -> None:
    browser.scroll_page("down")
    assert "window.scrollBy(0, window.innerHeight * 1.0)" in chrome.javascript[-1]


def test_scrolling_up_moves_the_other_way(chrome: FakeRun) -> None:
    browser.scroll_page("up")
    assert "-window.innerHeight" in chrome.javascript[-1]


def test_scrolling_accepts_part_of_a_screen(chrome: FakeRun) -> None:
    browser.scroll_page("down", 0.5)
    assert "window.innerHeight * 0.5" in chrome.javascript[-1]


@pytest.mark.parametrize("direction", ["sideways", "", "DOWNWARDS"])
def test_an_unknown_direction_is_refused(chrome: FakeRun, direction: str) -> None:
    with pytest.raises(ToolError, match="up, down, top or bottom"):
        browser.scroll_page(direction)


@pytest.mark.parametrize("screens", [0, -1])
def test_a_meaningless_distance_is_refused(chrome: FakeRun, screens: float) -> None:
    with pytest.raises(ToolError, match="must be positive"):
        browser.scroll_page("down", screens)


def test_scrolling_never_asks_permission() -> None:
    assert get("scroll_page").safety == "read_only"


# --- typing ---------------------------------------------------------------


def test_typing_sends_the_text_as_keystrokes(chrome: FakeRun) -> None:
    """Keystrokes, not a value assignment: a chat box built in React ignores
    a value it was handed."""
    browser.type_text("hello there")
    assert f"keystroke {json.dumps('hello there')}" in chrome.scripts[-1]


def test_typing_brings_chrome_forward_in_the_same_script(chrome: FakeRun) -> None:
    """Two scripts and the window can still be coming forward when the keys
    arrive, so the first character lands somewhere else."""
    browser.type_text("hello")
    last = chrome.scripts[-1]
    assert "activate" in last
    assert "keystroke" in last
    assert f"delay {browser.FOCUS_SETTLE}" in last


def test_typing_nothing_is_refused(chrome: FakeRun) -> None:
    with pytest.raises(ToolError, match="nothing was given to type"):
        browser.type_text("  ")


@pytest.mark.parametrize(("key", "code"), [("enter", 36), ("Escape", 53), ("tab", 48)])
def test_a_named_key_is_sent_as_its_code(chrome: FakeRun, key: str, code: int) -> None:
    browser.press_key(key)
    assert f"key code {code}" in chrome.scripts[-1]


def test_an_unknown_key_lists_the_known_ones(chrome: FakeRun) -> None:
    with pytest.raises(ToolError, match="enter"):
        browser.press_key("hyperspace")


def test_typing_asks_permission() -> None:
    assert get("type_text").safety == "mutating"


def test_pressing_a_key_counts_as_destructive() -> None:
    """The key people ask for is Enter, and in a chat box Enter sends the
    message. Sending something outward is what that class is for."""
    assert get("press_key").safety == "destructive"


# --- when the browser will not cooperate ----------------------------------


def test_chrome_not_running_says_to_open_it(chrome: FakeRun) -> None:
    chrome.answers = ["false"]
    with pytest.raises(ToolError, match="not running"):
        browser.read_page()


def test_a_blocked_automation_permission_explains_itself() -> None:
    message = browser._explain("Not allowed to send Apple events to Google Chrome. (-1743)")
    assert "Privacy & Security" in message
    assert "Automation" in message


def test_javascript_being_switched_off_explains_itself() -> None:
    message = browser._explain("Executing JavaScript through AppleScript is disabled.")
    assert "Allow JavaScript from Apple Events" in message


def test_no_window_is_explained() -> None:
    assert "no window" in browser._explain("Can't get window 1.")


def test_a_slow_page_is_reported(chrome: FakeRun) -> None:
    chrome.raises = subprocess.TimeoutExpired(cmd="osascript", timeout=browser.TIMEOUT)
    with pytest.raises(ToolError, match="did not respond"):
        browser.read_page()


def test_a_machine_without_osascript_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ToolError, match="only works on macOS"):
        browser.read_page()


def test_javascript_that_returns_nothing_reads_as_empty(chrome: FakeRun) -> None:
    """AppleScript prints "missing value" for undefined, which would reach
    the model as if the page had said those words."""
    chrome.answers = [browser.MISSING]
    assert browser._javascript("whatever") == ""


# --- the Phase 5 review's findings ----------------------------------------


@pytest.mark.parametrize("text", ["first line\nsecond line", "one\r\ntwo"])
def test_a_multi_line_message_is_refused(chrome: FakeRun, text: str) -> None:
    """A newline is a Return, and in a chat box Return sends. Typing two
    lines would post the first and report both as typed."""
    with pytest.raises(ToolError, match="press_key"):
        browser.type_text(text)
    assert chrome.scripts == []


def test_an_exact_label_is_preferred_over_a_longer_one(chrome: FakeRun) -> None:
    """Asked for "Ram", the first substring match in DOM order is "Rama
    Sharma" — a different person."""
    chrome.answers = ["true", "Ram"]
    browser.click_text("Ram")
    script = chrome.javascript[-1]
    assert "lower === wanted" in script
    assert script.index("lower === wanted") < script.index("lower.includes(wanted)")


def test_listing_and_clicking_judge_visibility_the_same_way(chrome: FakeRun) -> None:
    """A closed menu keeps its size while being hidden, so a non-zero box is
    not enough. The two must agree, or rolex offers what it cannot click."""
    chrome.answers = ["true", "- a"]
    browser.page_buttons()
    listing = chrome.javascript[-1]

    chrome.scripts.clear()
    chrome.answers = ["true", "a"]
    browser.click_text("a")
    clicking = chrome.javascript[-1]

    for script in (listing, clicking):
        assert "visibility" in script
        assert "opacity" in script
        assert "getComputedStyle" in script


@pytest.mark.parametrize("screens", ["2", "0.5"])
def test_a_distance_sent_as_text_still_works(chrome: FakeRun, screens: Any) -> None:
    """A small model sends "2" as often as 2."""
    browser.scroll_page("down", screens)
    assert f"window.innerHeight * {float(screens)}" in chrome.javascript[-1]


def test_a_distance_that_is_not_a_number_is_refused(chrome: FakeRun) -> None:
    bad: Any = "lots"
    with pytest.raises(ToolError, match="must be a number"):
        browser.scroll_page("down", bad)


def test_the_prompt_tells_the_model_typing_does_not_send() -> None:
    """Without this it types the message and reports success, having never
    pressed Return."""
    from rolex.core.config import Config
    from rolex.interface.chat import system_prompt

    prompt = system_prompt(Config())
    assert "press_key" in prompt
    assert "read_page" in prompt
