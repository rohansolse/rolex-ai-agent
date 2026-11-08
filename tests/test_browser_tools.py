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
def clicks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    """Records the real clicks that would have been sent, sending none."""
    sent: list[tuple[float, float]] = []
    monkeypatch.setattr("rolex.tools.pointer.click", lambda x, y: sent.append((x, y)))
    return sent


@pytest.fixture
def chrome(monkeypatch: pytest.MonkeyPatch, clicks: list[tuple[float, float]]) -> FakeRun:
    # Listings are remembered between calls so a later one can say what is
    # new. Left alone, that leaks between tests and a stray *new appears
    # depending on the order they ran in.
    browser.forget_listings()
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
    chrome.answers = [
        "true",
        page_list("1. [list/item] MBRJ self group\n2. [page/button] Send", "WhatsApp"),
    ]
    listed = browser.page_buttons()
    assert "MBRJ self group" in listed
    assert "Send" in listed


def test_page_buttons_says_so_when_there_is_nothing(chrome: FakeRun) -> None:
    chrome.answers = ["true", page_list("")]
    assert "Nothing clickable" in browser.page_buttons()


def test_reading_never_asks_permission() -> None:
    """Looking at a page cannot do any harm."""
    assert get("read_page").safety == "read_only"
    assert get("page_buttons").safety == "read_only"


# --- clicking -------------------------------------------------------------


def spot(label: str = "MBRJ self group", x: int = 100, y: int = 200) -> str:
    return json.dumps({"x": x, "y": y, "label": label})


def page_list(items: str, page: str = "A Page") -> str:
    """What the listing script returns: the page's name and its items."""
    return json.dumps({"page": page, "items": items})


def test_clicking_sends_the_wanted_text(chrome: FakeRun, clicks: list[tuple[float, float]]) -> None:
    chrome.answers = ["true", "", spot()]
    assert browser.click_text("MBRJ self group") == "Clicked MBRJ self group."
    assert json.dumps("mbrj self group") in chrome.javascript[-1].lower()


def test_clicking_sends_a_real_click_at_the_right_place(
    chrome: FakeRun, clicks: list[tuple[float, float]]
) -> None:
    """Synthetic DOM events and System Events both reported success and did
    nothing on WhatsApp Web. Only a Quartz event opens a conversation."""
    chrome.answers = ["true", "", spot(x=1771, y=629)]
    browser.click_text("MBRJ self group")
    assert clicks == [(1771.0, 629.0)]


def test_clicking_something_absent_says_what_to_do(chrome: FakeRun) -> None:
    chrome.answers = ["true", "", json.dumps({"missing": "nowhere"})]
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


def test_an_exact_label_is_preferred_over_a_longer_one(
    chrome: FakeRun, clicks: list[tuple[float, float]]
) -> None:
    """Asked for "Ram", the first substring match in DOM order is "Rama
    Sharma" — a different person."""
    chrome.answers = ["true", "", spot("Ram")]
    browser.click_text("Ram")
    script = chrome.javascript[-1]
    assert "lower === wanted" in script
    assert script.index("lower === wanted") < script.index("lower.includes(wanted)")


def test_listing_and_clicking_judge_visibility_the_same_way(chrome: FakeRun) -> None:
    """A closed menu keeps its size while being hidden, so a non-zero box is
    not enough. The two must agree, or rolex offers what it cannot click."""
    chrome.answers = ["true", page_list("1. [list/item] a")]
    browser.page_buttons()
    listing = chrome.javascript[0]

    chrome.scripts.clear()
    chrome.answers = ["true", "", spot("a")]
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
    assert "page_buttons" in prompt
    assert "click_item" in prompt


# --- clicking by number ---------------------------------------------------


def test_the_listing_is_numbered(chrome: FakeRun) -> None:
    """Text alone is ambiguous: asked to click "Rohan Solse" on WhatsApp
    Web, those words appear both in the chat list and inside the open
    conversation. Numbers let the model look first and then say which."""
    chrome.answers = ["true", page_list("1. Home Sweet Home\n2. MBRJ | Self", "WhatsApp")]
    listed = browser.page_buttons()
    assert "click_item" in listed
    assert "2. MBRJ | Self" in listed


def test_clicking_a_number_clicks_that_item(
    chrome: FakeRun, clicks: list[tuple[float, float]]
) -> None:
    chrome.answers = ["true", "", spot("MBRJ | Self", x=280, y=632)]
    assert browser.click_item(22) == "Clicked MBRJ | Self."
    assert clicks == [(280.0, 632.0)]


def test_the_number_is_counted_from_one(chrome: FakeRun) -> None:
    """page_buttons prints 1, 2, 3; the list behind it starts at 0."""
    chrome.answers = ["true", "", spot()]
    browser.click_item(1)
    assert "out[0]" in chrome.javascript[-1]


@pytest.mark.parametrize("number", [0, -3])
def test_a_number_below_one_is_refused(chrome: FakeRun, number: int) -> None:
    with pytest.raises(ToolError, match="starts at 1"):
        browser.click_item(number)


def test_a_number_sent_as_text_still_works(
    chrome: FakeRun, clicks: list[tuple[float, float]]
) -> None:
    """A small model sends "3" as often as 3."""
    chrome.answers = ["true", "", spot()]
    browser.click_item("3")
    assert "out[2]" in chrome.javascript[-1]


def test_something_that_is_not_a_number_is_refused(chrome: FakeRun) -> None:
    with pytest.raises(ToolError, match="whole number"):
        browser.click_item("the third one")


def test_a_number_past_the_end_says_to_look_again(chrome: FakeRun) -> None:
    chrome.answers = ["true", "", json.dumps({"missing": 99})]
    with pytest.raises(ToolError, match="page_buttons"):
        browser.click_item(100)


def test_clicking_by_number_asks_permission() -> None:
    assert get("click_item").safety == "mutating"


def test_a_stale_number_is_refused_rather_than_clicked(chrome: FakeRun) -> None:
    """Numbers go stale the moment a click changes what is visible, and
    clicking the wrong conversation then typing sends a message to the
    wrong person."""
    chrome.answers = ["true", "", spot("Rohan Solse")]
    with pytest.raises(ToolError, match="page has changed"):
        browser.click_item(22, expected="MBRJ | Self")


def test_a_number_that_still_matches_is_clicked(
    chrome: FakeRun, clicks: list[tuple[float, float]]
) -> None:
    chrome.answers = ["true", "", spot("MBRJ | Self 12:09 am")]
    browser.click_item(22, expected="MBRJ | Self")
    assert clicks


def test_clicking_waits_for_chrome_to_come_forward(chrome: FakeRun) -> None:
    """A real click goes wherever the pointer is told. If Chrome is still
    behind the terminal when it lands, the terminal gets clicked."""
    chrome.answers = ["true", "", spot()]
    browser.click_text("anything")
    assert any("activate" in s and "delay" in s for s in chrome.scripts)


def test_listing_and_locating_share_one_limit(chrome: FakeRun) -> None:
    """Without the same cap, click_item(41) clicks something the user was
    never shown."""
    chrome.answers = ["true", page_list("1. [list/item] a")]
    browser.page_buttons()
    listing = chrome.javascript[0]

    chrome.scripts.clear()
    chrome.answers = ["true", "", spot()]
    browser.click_item(1)
    locating = chrome.javascript[-1]

    for script in (listing, locating):
        assert "LIMIT" in script


def test_coordinates_account_for_page_zoom(chrome: FakeRun) -> None:
    """Zoom scales CSS pixels but not screenX or outerHeight; ignoring it
    put the click a long way off and still called it a success."""
    chrome.answers = ["true", "", spot()]
    browser.click_text("anything")
    assert "outerWidth / window.innerWidth" in chrome.javascript[-1]


def test_scrolling_into_view_is_instant(chrome: FakeRun) -> None:
    """With scroll-behavior: smooth the page is still moving when the
    rectangle is read."""
    chrome.answers = ["true", "", spot()]
    browser.click_text("anything")
    assert "behavior: 'instant'" in chrome.javascript[-1]


# --- tabs -----------------------------------------------------------------


def test_tabs_are_listed_with_numbers_and_the_open_one_marked(chrome: FakeRun) -> None:
    chrome.answers = ["true", "1. Colab\n2. [open] WhatsApp"]
    listed = browser.list_tabs()
    assert "close_tab" in listed
    assert "[open] WhatsApp" in listed


def test_no_tabs_says_so(chrome: FakeRun) -> None:
    chrome.answers = ["true", ""]
    assert browser.list_tabs() == "No tabs are open."


def test_closing_a_tab_names_what_was_closed(chrome: FakeRun) -> None:
    chrome.answers = ["true", "3", "Example Domain", ""]
    assert browser.close_tab(3) == "Closed the tab showing 'Example Domain'."
    assert any("close tab 3" in script for script in chrome.scripts)


def test_switching_tab_names_where_it_went(chrome: FakeRun) -> None:
    chrome.answers = ["true", "3", "Colab", ""]
    assert browser.switch_tab(1) == "Switched to 'Colab'."
    assert any("active tab index of front window to 1" in s for s in chrome.scripts)


def test_a_tab_number_past_the_end_is_refused(chrome: FakeRun) -> None:
    """A number from an earlier listing points elsewhere once a tab has been
    closed, and closing the wrong tab loses what was in it."""
    chrome.answers = ["true", "2"]
    with pytest.raises(ToolError, match="Chrome has 2"):
        browser.close_tab(99)
    assert not any("close tab" in script for script in chrome.scripts)


@pytest.mark.parametrize("number", [0, -1])
def test_a_tab_number_below_one_is_refused(chrome: FakeRun, number: int) -> None:
    with pytest.raises(ToolError, match="numbered from 1"):
        browser.close_tab(number)


def test_something_that_is_not_a_tab_number_is_refused(chrome: FakeRun) -> None:
    with pytest.raises(ToolError, match="whole number"):
        browser.close_tab("two")


def test_no_window_open_is_reported(chrome: FakeRun) -> None:
    chrome.answers = ["true", "not a number"]
    with pytest.raises(ToolError, match="no window open"):
        browser.close_tab(1)


def test_listing_tabs_never_asks_permission() -> None:
    assert get("list_tabs").safety == "read_only"


def test_closing_and_switching_tabs_ask_permission() -> None:
    assert get("close_tab").safety == "mutating"
    assert get("switch_tab").safety == "mutating"


def test_the_prompt_separates_a_tab_from_an_application() -> None:
    """Asked to close a tab, rolex reached for close_app and reported that
    "WhatsApp Web is not running"."""
    from rolex.core.config import Config
    from rolex.interface.chat import system_prompt

    prompt = system_prompt(Config())
    assert "close_tab" in prompt
    assert "close_app" in prompt


# --- opening a WhatsApp chat ----------------------------------------------


def whatsapp(
    monkeypatch: pytest.MonkeyPatch, answers: list[str]
) -> tuple[list[str], list[tuple[float, float]]]:
    """Fake the page and the pointer; return the scripts run and the clicks."""
    from rolex.tools import chat_page

    scripts: list[str] = []
    clicked: list[tuple[float, float]] = []
    replies = iter(answers)

    def run(script: str) -> str:
        scripts.append(script)
        return next(replies)

    monkeypatch.setattr(chat_page, "_require_chrome", lambda: None)
    monkeypatch.setattr(chat_page, "_focus", lambda: None)
    monkeypatch.setattr(chat_page, "_javascript", run)
    monkeypatch.setattr(chat_page, "POLL", 0)
    monkeypatch.setattr("rolex.tools.pointer.click", lambda x, y: clicked.append((x, y)))
    return scripts, clicked


def test_a_chat_is_found_by_searching(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat list is virtualised: a chat far down it is not in the page
    at all until something scrolls to it. Searching is what a person does."""
    from rolex.tools import chat_page

    found = json.dumps({"x": 280, "y": 632, "label": "testing ai"})
    _, clicked = whatsapp(
        monkeypatch,
        ["https://web.whatsapp.com/", "ok", found, found, "testing ai", ""],
    )

    assert chat_page.open_whatsapp_chat("testing ai") == (
        "Opened the WhatsApp chat with testing ai."
    )
    assert clicked == [(280.0, 632.0)]


def test_the_search_box_may_be_an_input_or_a_contenteditable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WhatsApp changed it from one to the other, which is what broke this."""
    from rolex.tools import chat_page

    spot = json.dumps({"x": 1, "y": 2, "label": "a"})
    scripts, _ = whatsapp(monkeypatch, ["https://web.whatsapp.com/", "ok", spot, spot, "a", ""])
    chat_page.open_whatsapp_chat("a")

    typing = scripts[1]
    assert "input[role=textbox]" in typing
    assert "contenteditable" in typing


def test_a_name_that_never_comes_up_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from rolex.tools import chat_page

    whatsapp(
        monkeypatch,
        ["https://web.whatsapp.com/", "ok", *["waiting"] * 40],
    )

    with pytest.raises(ToolError, match="came up in the search"):
        chat_page.open_whatsapp_chat("nobody")


def test_a_chat_that_will_not_open_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clicking is not the same as the conversation appearing."""
    from rolex.tools import chat_page

    found = json.dumps({"x": 1, "y": 2, "label": "a"})
    whatsapp(
        monkeypatch,
        ["https://web.whatsapp.com/", "ok", found, found, *["something else"] * 40, ""],
    )

    with pytest.raises(ToolError, match="did not open"):
        chat_page.open_whatsapp_chat("a")


def test_an_empty_name_is_refused() -> None:
    from rolex.tools import chat_page

    with pytest.raises(ToolError, match="give the chat's name"):
        chat_page.open_whatsapp_chat("   ")


def test_it_refuses_when_whatsapp_is_not_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from rolex.tools import chat_page

    whatsapp(monkeypatch, ["https://example.com/"])

    with pytest.raises(ToolError, match="not the page in front"):
        chat_page.open_whatsapp_chat("a")


def test_the_search_is_cleared_afterwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search left in place hides every other chat from the next listing."""
    from rolex.tools import chat_page

    found = json.dumps({"x": 1, "y": 2, "label": "a"})
    scripts, _ = whatsapp(monkeypatch, ["https://web.whatsapp.com/", "ok", found, found, "a", ""])
    chat_page.open_whatsapp_chat("a")

    assert "delete" in scripts[-1] or "''" in scripts[-1]


# --- saying where each thing is -------------------------------------------


def test_each_item_says_where_it_sits(chrome: FakeRun) -> None:
    """ "Rohan Solse" appears in the chat list and inside the open
    conversation. A name alone cannot say which was meant."""
    chrome.answers = ["true", page_list("1. [list/item] Rohan Solse", "WhatsApp")]
    listed = browser.page_buttons()
    assert "[list/item] Rohan Solse" in listed


def test_the_regions_are_named_in_the_script(chrome: FakeRun) -> None:
    chrome.answers = ["true", page_list("1. [list/item] a", "WhatsApp")]
    browser.page_buttons()
    script = chrome.javascript[0]
    for region in ("dialog", "list", "conversation", "header", "nav"):
        assert f"'{region}'" in script


def test_the_page_is_named(chrome: FakeRun) -> None:
    chrome.answers = ["true", page_list("1. [page/button] Send", "WhatsApp")]
    assert "WhatsApp" in browser.page_buttons()


# --- saying what is new ---------------------------------------------------


def test_the_first_listing_marks_nothing_as_new(chrome: FakeRun) -> None:
    chrome.answers = ["true", page_list("1. [list/item] a\n2. [list/item] b")]
    assert "*new" not in browser.page_buttons()


def test_what_appeared_since_last_time_is_marked(chrome: FakeRun) -> None:
    """A click usually changes the page, and "what appeared?" is the next
    question."""
    chrome.answers = ["true", page_list("1. [list/item] a")]
    browser.page_buttons()

    chrome.answers = ["true", page_list("1. [list/item] a\n2. [page/button] Send")]
    listed = browser.page_buttons()

    assert "Send  *new" in listed
    assert "] a\n" in listed or listed.rstrip().endswith("] a")


def test_a_renumbered_item_is_not_called_new(chrome: FakeRun) -> None:
    """Numbers shift as things appear; the label is what identifies a thing."""
    chrome.answers = ["true", page_list("1. [list/item] b")]
    browser.page_buttons()

    chrome.answers = ["true", page_list("1. [page/button] Back\n2. [list/item] b")]
    listed = browser.page_buttons()

    assert "] b" in listed
    assert "b  *new" not in listed


@pytest.mark.parametrize(
    "remembered",
    [
        "22. [list/item] MBRJ | Self  *new",
        "[list/item] MBRJ | Self",
        "MBRJ | Self",
        "22. MBRJ | Self",
    ],
)
def test_a_line_copied_from_the_listing_is_accepted(
    chrome: FakeRun, clicks: list[tuple[float, float]], remembered: str
) -> None:
    """A model passing back "what that line said" includes the number, the
    region and any *new marker. Compared literally, every check failed."""
    chrome.answers = ["true", "", spot("[list/item] MBRJ | Self 12:09 am")]
    browser.click_item(22, expected=remembered)
    assert clicks


def test_a_different_item_is_still_refused(chrome: FakeRun) -> None:
    chrome.answers = ["true", "", spot("[list/item] Rohan Solse")]
    with pytest.raises(ToolError, match="page has changed"):
        browser.click_item(22, expected="22. [list/item] MBRJ | Self")


def test_the_same_name_in_two_regions_is_listed_twice(chrome: FakeRun) -> None:
    """Collapsing them is exactly the ambiguity the region was added to
    remove: "Rohan Solse" is both a chat in the sidebar and a line inside
    the open conversation."""
    chrome.answers = ["true", page_list("1. [list/item] a", "A Page")]
    browser.page_buttons()
    script = chrome.javascript[0]
    assert "describe(node) + '|' + label" in script or "where + '|' + label" in script


def test_locating_keys_on_the_region_too(chrome: FakeRun) -> None:
    """Listing and locating must agree, or a number means two things."""
    chrome.answers = ["true", "", spot()]
    browser.click_item(1)
    assert "describe(node) + '|' + label" in chrome.javascript[-1]


def test_a_submit_input_is_called_a_button(chrome: FakeRun) -> None:
    """Only input[type=submit] is ever listed; calling it a field makes the
    model try to type into it."""
    chrome.answers = ["true", page_list("1. [page/button] Send")]
    browser.page_buttons()
    script = chrome.javascript[0]
    assert "'field'" not in script


def test_a_second_page_does_not_mark_everything_new(chrome: FakeRun) -> None:
    """Remembered without a page, listing another one made every item look
    as if it had just appeared."""
    chrome.answers = ["true", page_list("1. [list/item] a", "Page A")]
    browser.page_buttons()

    chrome.answers = ["true", page_list("1. [list/item] x", "Page B")]
    assert "*new" not in browser.page_buttons()


def test_going_back_to_a_page_remembers_it(chrome: FakeRun) -> None:
    chrome.answers = ["true", page_list("1. [list/item] a", "Page A")]
    browser.page_buttons()
    chrome.answers = ["true", page_list("1. [list/item] x", "Page B")]
    browser.page_buttons()

    chrome.answers = ["true", page_list("1. [list/item] a\n2. [page/button] b", "Page A")]
    listed = browser.page_buttons()

    assert "b  *new" in listed
    assert "a  *new" not in listed


def test_having_listed_nothing_is_not_the_same_as_never_listing(
    chrome: FakeRun,
) -> None:
    """A page still loading lists nothing; what turns up next is new."""
    chrome.answers = ["true", page_list("", "Slow Page")]
    browser.page_buttons()

    chrome.answers = ["true", page_list("1. [page/button] now here", "Slow Page")]
    assert "*new" in browser.page_buttons()


def test_the_heading_and_the_items_come_from_one_call(chrome: FakeRun) -> None:
    """A second call for the title could straddle a tab change and head the
    listing with a different page's name."""
    chrome.answers = ["true", page_list("1. [page/button] Send", "WhatsApp")]
    browser.page_buttons()
    assert len(chrome.javascript) == 1


def test_a_moving_result_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Results animate into place, and a click sent while they are sliding
    lands on whatever was at that spot a moment earlier."""
    from rolex.tools import chat_page

    moving = json.dumps({"x": 280, "y": 583, "label": "a"})
    settled = json.dumps({"x": 280, "y": 397, "label": "a"})
    _, clicked = whatsapp(
        monkeypatch,
        ["https://web.whatsapp.com/", "ok", moving, settled, settled, "a", ""],
    )

    chat_page.open_whatsapp_chat("a")

    assert clicked == [(280.0, 397.0)]


def test_a_name_missing_its_punctuation_still_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat saved as "MBRJ | Self" is asked for as "MBRJ Self", and both
    WhatsApp's search and a literal check miss it."""
    from rolex.tools import chat_page

    spot = json.dumps({"x": 1, "y": 2, "label": "MBRJ | Self"})
    whatsapp(
        monkeypatch,
        ["https://web.whatsapp.com/", "ok", spot, spot, "MBRJ | Self \U0001f468", ""],
    )

    assert "MBRJ Self" in chat_page.open_whatsapp_chat("MBRJ Self")


def test_the_search_uses_one_distinctive_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """WhatsApp's own search matches the phrase, so punctuation in the real
    name makes a whole-phrase search find nothing."""
    from rolex.tools import chat_page

    assert chat_page._search_term("MBRJ Self") == "MBRJ"
    assert chat_page._search_term("testing ai") == "testing"
    assert chat_page._search_term("ab") == "ab"


def test_the_matching_accepts_the_words_in_any_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rolex.tools import chat_page

    spot = json.dumps({"x": 1, "y": 2, "label": "a"})
    scripts, _ = whatsapp(monkeypatch, ["https://web.whatsapp.com/", "ok", spot, spot, "a", ""])
    chat_page.open_whatsapp_chat("a")

    locating = scripts[2]
    assert "words.every" in locating


def test_chrome_is_brought_forward_before_clicking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real click goes where the pointer is told; a background Chrome
    never receives it."""
    from rolex.tools import chat_page

    focused: list[bool] = []

    spot = json.dumps({"x": 1, "y": 2, "label": "a"})
    whatsapp(monkeypatch, ["https://web.whatsapp.com/", "ok", spot, spot, "a", ""])
    monkeypatch.setattr(chat_page, "_focus", lambda: focused.append(True))
    chat_page.open_whatsapp_chat("a")

    assert focused == [True]
