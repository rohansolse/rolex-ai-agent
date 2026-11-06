"""Clicking where a human would click.

macOS offers several ways to click and only one of them convinces a web
application. They were tried in order:

- ``element.click()`` in JavaScript. WhatsApp Web reported every click as
  successful and opened nothing: it listens for genuine input.
- A full sequence of synthetic ``pointerdown``/``mousedown``/``click``
  events. Same result.
- AppleScript's ``System Events to click at {x, y}``. That asks the
  accessibility layer to press whatever is at the point, and a web page's
  internals are not accessible elements, so nothing happens — silently, with
  the permission granted and no error.
- A Quartz event, which is what a mouse itself produces. This works.

Quartz is reached through :mod:`ctypes`, so this still adds no dependency.
It needs Accessibility permission for whatever is running rolex, under
System Settings -> Privacy & Security -> Accessibility.
"""

import ctypes
import ctypes.util
import time

from rolex.core.errors import ToolError

#: Quartz event types and the tap to post them to.
_MOUSE_MOVED = 5
_LEFT_DOWN = 1
_LEFT_UP = 2
_HID_TAP = 0
_LEFT_BUTTON = 0

SETTLE = 0.05
"""Seconds between moving, pressing and releasing.

A press and release in the same instant is not what a mouse does, and some
interfaces treat it as noise."""


class CGPoint(ctypes.Structure):
    """A point on screen, in the coordinates Quartz uses."""

    _fields_ = (("x", ctypes.c_double), ("y", ctypes.c_double))


def _quartz() -> ctypes.CDLL:
    """The ApplicationServices library, with the signatures we need."""
    path = ctypes.util.find_library("ApplicationServices")
    if path is None:
        raise ToolError("this only works on macOS: ApplicationServices was not found")

    library = ctypes.CDLL(path)
    library.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    library.CGEventCreateMouseEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        CGPoint,
        ctypes.c_uint32,
    ]
    library.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    library.CFRelease.argtypes = [ctypes.c_void_p]
    return library


def allowed() -> bool:
    """Whether this process may post input events.

    Creating an event succeeds without permission; it is posting that is
    silently dropped. Asking first is the only way to tell the difference
    between a click that worked and one that went nowhere while reporting
    success.
    """
    path = ctypes.util.find_library("ApplicationServices")
    if path is None:
        return False
    library = ctypes.CDLL(path)
    try:
        library.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(library.AXIsProcessTrusted())
    except AttributeError:
        # Old enough macOS not to have it. Assume yes rather than refuse.
        return True


def click(x: float, y: float) -> None:
    """Click once at a point on the screen, as a mouse would.

    Raises:
        ToolError: the click cannot be sent, which on macOS means
            Accessibility permission has not been given.
    """
    if not allowed():
        raise ToolError(
            "clicks are not permitted. Give the terminal permission under "
            "System Settings -> Privacy & Security -> Accessibility, then start "
            "rolex again. Without it a click is silently thrown away."
        )

    library = _quartz()

    for kind in (_MOUSE_MOVED, _LEFT_DOWN, _LEFT_UP):
        event = library.CGEventCreateMouseEvent(None, kind, CGPoint(x, y), _LEFT_BUTTON)
        if not event:
            raise ToolError(
                "the click could not be sent. Give the terminal permission under "
                "System Settings -> Privacy & Security -> Accessibility."
            )
        try:
            library.CGEventPost(_HID_TAP, event)
        finally:
            library.CFRelease(event)
        time.sleep(SETTLE)
