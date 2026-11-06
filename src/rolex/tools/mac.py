"""Tools that act on the Mac.

Everything here shells out to ``open``, the macOS command for "do the normal
thing with this". Arguments are passed as a list and never through a shell, so
a name the model invented cannot turn into a command.
"""

import shutil
import subprocess
import urllib.parse
from collections.abc import Callable, Sequence
from pathlib import Path

from rolex.core.errors import ToolError
from rolex.tools.registry import register

TIMEOUT = 15.0
"""Seconds to wait for ``open``. It returns as soon as the app is launching,
so anything slower than this has gone wrong."""

SEARCH_URL = "https://duckduckgo.com/?q={}"
"""Chosen over Google because it does not demand a consent click first."""

APP_DIRECTORIES = (
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)
"""Where to look for an installed app. Covers the normal places; anything
elsewhere is still tried by name, since ``open -a`` knows more than this."""


def _open(*arguments: str) -> None:
    """Run macOS ``open`` with the given arguments.

    Never through a shell: `arguments` come from a model, and a list argv
    means a name containing ``;`` or ``$()`` stays a name.
    """
    executable = shutil.which("open")
    if executable is None:
        raise ToolError("this only works on macOS: the 'open' command is not available")

    try:
        # An argv list, never a shell string: see _open's docstring.
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"'open' did not finish within {TIMEOUT:.0f} seconds") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise ToolError(detail)


@register(
    description="Open a Mac application by name, for example Safari or Notes.",
    safety="mutating",
)
def open_app(name: str) -> str:
    """Launch an application.

    The name is matched against what is installed before being handed to
    ``open``, because people and models say "Chrome" where the app is called
    "Google Chrome". Without this, a correct request failed on a detail the
    user had no way to know.

    Args:
        name: The application's name, as said rather than as filed.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ToolError("no application name was given")

    resolved = resolve_app(cleaned, installed_apps())
    _open("-a", resolved)
    return f"Opened {resolved}." if resolved == cleaned else f"Opened {resolved} ({cleaned})."


def installed_apps() -> list[str]:
    """Names of the applications found in the usual places."""
    found: set[str] = set()
    for directory in APP_DIRECTORIES:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        found.update(entry.stem for entry in entries if entry.suffix == ".app")
    return sorted(found)


def resolve_app(wanted: str, apps: Sequence[str]) -> str:
    """The installed app `wanted` refers to, or `wanted` itself.

    Tried in order: the exact name, a name starting with it, a name containing
    it. An ambiguous request raises rather than guessing, so the model can ask
    which one instead of opening the wrong thing.
    """
    needle = wanted.casefold()

    tests: tuple[Callable[[str], bool], ...] = (
        lambda app: app == needle,
        lambda app: app.startswith(needle),
        lambda app: needle in app,
    )

    for matches in tests:
        found = [app for app in apps if matches(app.casefold())]
        if len(found) == 1:
            return found[0]
        if found:
            return _one_of(wanted, found)

    # Not installed anywhere we looked. Hand it to `open` regardless: it
    # knows about apps in places this does not.
    return wanted


def _one_of(wanted: str, candidates: Sequence[str]) -> str:
    """Refuse to guess between several apps, naming them all."""
    raise ToolError(
        f"'{wanted}' matches several applications: {', '.join(candidates)}. "
        f"Ask the user which one they meant."
    )


@register(
    description="Open a web page in the default browser. Needs a full URL.",
    safety="mutating",
)
def open_url(url: str) -> str:
    """Open a URL in whichever browser is the default.

    Args:
        url: A full http or https address.
    """
    cleaned = url.strip()
    parsed = urllib.parse.urlparse(cleaned)

    # Only http(s). `open` will happily act on file:// and on custom app
    # schemes, and neither is something a model should be able to reach for.
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError(f"'{cleaned}' is not an http or https address")

    _open(cleaned)
    return f"Opened {cleaned} in the browser."


@register(
    description="Search the web in the browser. Use when asked to look something up online.",
    safety="mutating",
)
def search_web(query: str) -> str:
    """Open a search for `query` in the default browser.

    Args:
        query: What to search for.
    """
    cleaned = " ".join(query.split())
    if not cleaned:
        raise ToolError("nothing to search for")

    _open(SEARCH_URL.format(urllib.parse.quote_plus(cleaned)))
    return f"Searched the web for '{cleaned}' in the browser."
