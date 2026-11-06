"""The tools that act on the Mac.

``open`` is never actually run: a fake replaces the subprocess call, so the
tests say what argv we build and what we refuse. Nothing opens on the machine
running the suite.
"""

import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from rolex.core.errors import ToolError
from rolex.tools import mac


@dataclass
class FakeRun:
    """Stands in for subprocess.run and records the argv it was given."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    raises: Exception | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str], **_: Any) -> "FakeRun":
        self.calls.append(list(argv))
        if self.raises:
            raise self.raises
        return self

    @property
    def argv(self) -> list[str]:
        assert len(self.calls) == 1, f"expected one call, got {self.calls}"
        return self.calls[0]


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/open")
    monkeypatch.setattr("subprocess.run", fake)
    return fake


# --- open_app -------------------------------------------------------------


def test_open_app_builds_the_right_command(run: FakeRun) -> None:
    assert mac.open_app("Safari") == "Opened Safari."
    assert run.argv == ["/usr/bin/open", "-a", "Safari"]


def test_open_app_trims_whitespace(run: FakeRun) -> None:
    mac.open_app("  Notes  ")
    assert run.argv[-1] == "Notes"


def test_open_app_refuses_an_empty_name(run: FakeRun) -> None:
    with pytest.raises(ToolError, match="no application name"):
        mac.open_app("   ")
    assert run.calls == []


def test_a_name_with_shell_metacharacters_stays_one_argument(run: FakeRun) -> None:
    """The name comes from a model. It must never become a command."""
    mac.open_app("Safari; rm -rf ~")
    assert run.argv == ["/usr/bin/open", "-a", "Safari; rm -rf ~"]


def test_a_failure_from_open_is_reported(run: FakeRun) -> None:
    run.returncode = 1
    run.stderr = "Unable to find application named 'Nope'"
    with pytest.raises(ToolError, match="Unable to find application"):
        mac.open_app("Nope")


def test_a_failure_with_no_message_still_says_something(run: FakeRun) -> None:
    run.returncode = 3
    with pytest.raises(ToolError, match="exit code 3"):
        mac.open_app("Nope")


def test_a_hanging_open_is_reported(run: FakeRun) -> None:
    run.raises = subprocess.TimeoutExpired(cmd="open", timeout=mac.TIMEOUT)
    with pytest.raises(ToolError, match="did not finish"):
        mac.open_app("Safari")


def test_a_machine_without_open_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ToolError, match="only works on macOS"):
        mac.open_app("Safari")


# --- open_url -------------------------------------------------------------


def test_open_url_passes_an_https_address(run: FakeRun) -> None:
    mac.open_url("https://example.com/page")
    assert run.argv == ["/usr/bin/open", "https://example.com/page"]


def test_http_is_allowed(run: FakeRun) -> None:
    mac.open_url("http://example.com")
    assert run.argv[-1] == "http://example.com"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "x-apple-systempreferences://",
        "javascript:alert(1)",
        "example.com",
        "https://",
        "   ",
    ],
)
def test_anything_that_is_not_an_http_url_is_refused(run: FakeRun, url: str) -> None:
    """`open` acts on file:// and app schemes too, and a model should not
    be able to reach either."""
    with pytest.raises(ToolError, match="not an http or https address"):
        mac.open_url(url)
    assert run.calls == []


# --- search_web -----------------------------------------------------------


def test_search_web_encodes_the_query(run: FakeRun) -> None:
    result = mac.search_web("best pav bhaji in pune")
    assert run.argv[-1] == "https://duckduckgo.com/?q=best+pav+bhaji+in+pune"
    assert "best pav bhaji in pune" in result


def test_search_web_encodes_characters_that_would_break_the_url(run: FakeRun) -> None:
    mac.search_web("a & b ?c=1 #d")
    assert run.argv[-1] == "https://duckduckgo.com/?q=a+%26+b+%3Fc%3D1+%23d"


def test_search_web_collapses_whitespace(run: FakeRun) -> None:
    mac.search_web("  two   words\n")
    assert run.argv[-1].endswith("two+words")


def test_search_web_refuses_an_empty_query(run: FakeRun) -> None:
    with pytest.raises(ToolError, match="nothing to search for"):
        mac.search_web("   ")
    assert run.calls == []


# --- how they are declared ------------------------------------------------


@pytest.mark.parametrize("name", ["open_app", "open_url", "search_web"])
def test_every_mac_tool_is_mutating(name: str) -> None:
    """None of these only look; each one makes a window appear."""
    from rolex.tools import get

    assert get(name).safety == "mutating"


# --- matching what the user said to what is installed ---------------------

APPS = ["Google Chrome", "Safari", "Notes", "Terminal", "Visual Studio Code", "Chess"]


def test_an_exact_name_is_used_as_is() -> None:
    assert mac.resolve_app("Safari", APPS) == "Safari"


def test_matching_ignores_case() -> None:
    assert mac.resolve_app("safari", APPS) == "Safari"


def test_a_partial_name_finds_the_app() -> None:
    """People and models say "Chrome"; the app is "Google Chrome". A correct
    request used to fail on a detail the user had no way to know."""
    assert mac.resolve_app("chrome", APPS) == "Google Chrome"


def test_a_leading_word_finds_the_app() -> None:
    assert mac.resolve_app("google", APPS) == "Google Chrome"


def test_an_exact_match_beats_a_longer_one() -> None:
    assert mac.resolve_app("Notes", [*APPS, "Notes Plus"]) == "Notes"


TWO_CHROMES = ["Google Chrome", "Google Chrome Canary"]


def test_an_ambiguous_name_refuses_rather_than_guessing() -> None:
    """Opening the wrong app is worse than asking which one."""
    with pytest.raises(ToolError, match="matches several applications"):
        mac.resolve_app("chrome", TWO_CHROMES)


def test_the_ambiguity_error_names_the_candidates_so_the_model_can_ask() -> None:
    with pytest.raises(ToolError, match="Google Chrome Canary"):
        mac.resolve_app("chrome", TWO_CHROMES)


def test_an_ambiguous_prefix_also_refuses() -> None:
    with pytest.raises(ToolError, match="matches several"):
        mac.resolve_app("google", TWO_CHROMES)


def test_an_exact_name_wins_over_an_ambiguous_substring() -> None:
    assert mac.resolve_app("Google Chrome", TWO_CHROMES) == "Google Chrome"


def test_an_unknown_name_is_passed_through() -> None:
    """`open` knows about apps in places this does not look."""
    assert mac.resolve_app("Setapp Thing", APPS) == "Setapp Thing"


def test_open_app_opens_the_resolved_name(run: FakeRun, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mac, "installed_apps", lambda: APPS)
    result = mac.open_app("chrome")
    assert run.argv == ["/usr/bin/open", "-a", "Google Chrome"]
    assert "Google Chrome" in result
    assert "chrome" in result


def test_open_app_says_nothing_extra_when_the_name_was_already_right(
    run: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mac, "installed_apps", lambda: APPS)
    assert mac.open_app("Safari") == "Opened Safari."


def test_installed_apps_skips_directories_that_are_not_there() -> None:
    """Not every Mac has ~/Applications."""
    assert isinstance(mac.installed_apps(), list)
