"""Sending a real mouse click.

The library is faked, so nothing on the machine is clicked. What is tested
is that a click is a move, a press and a release, posted in that order, and
that each event is released afterwards.
"""

from typing import Any

import pytest

from rolex.core.errors import ToolError
from rolex.tools import pointer


class FakeQuartz:
    """Stands in for ApplicationServices, recording what was posted."""

    def __init__(self, *, creates: bool = True) -> None:
        self.creates = creates
        self.posted: list[tuple[int, float, float]] = []
        self.released = 0
        self.handles = 0

        for name in ("CGEventCreateMouseEvent", "CGEventPost", "CFRelease"):
            setattr(self, f"_{name}", None)

    @property
    def CGEventCreateMouseEvent(self) -> Any:  # noqa: N802 - mirrors the C name
        def create(source: Any, kind: int, point: Any, button: int) -> int:
            if not self.creates:
                return 0
            self.handles += 1
            self.last = (kind, point.x, point.y)
            return self.handles

        return _Signature(create)

    @property
    def CGEventPost(self) -> Any:  # noqa: N802 - mirrors the C name
        def post(tap: int, event: int) -> None:
            self.posted.append(self.last)

        return _Signature(post)

    @property
    def CFRelease(self) -> Any:  # noqa: N802 - mirrors the C name
        def release(event: int) -> None:
            self.released += 1

        return _Signature(release)


class _Signature:
    """A callable that tolerates having restype/argtypes assigned to it."""

    def __init__(self, call: Any) -> None:
        self._call = call
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *args: Any) -> Any:
        return self._call(*args)


@pytest.fixture
def quartz(monkeypatch: pytest.MonkeyPatch) -> FakeQuartz:
    fake = FakeQuartz()
    monkeypatch.setattr(pointer, "_quartz", lambda: fake)
    monkeypatch.setattr(pointer, "allowed", lambda: True)
    monkeypatch.setattr(pointer, "SETTLE", 0)
    return fake


def test_a_click_is_a_move_a_press_and_a_release(quartz: FakeQuartz) -> None:
    pointer.click(100, 200)
    assert [kind for kind, _, _ in quartz.posted] == [
        pointer._MOUSE_MOVED,
        pointer._LEFT_DOWN,
        pointer._LEFT_UP,
    ]


def test_every_event_lands_at_the_same_point(quartz: FakeQuartz) -> None:
    pointer.click(1771, 629)
    assert {(x, y) for _, x, y in quartz.posted} == {(1771.0, 629.0)}


def test_nothing_is_leaked(quartz: FakeQuartz) -> None:
    """Each Quartz event has to be released, or the process grows."""
    pointer.click(10, 10)
    assert quartz.released == len(quartz.posted)


def test_without_permission_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Posting an event without Accessibility is dropped in silence, so the
    click has to be refused up front rather than reported as done."""
    fake = FakeQuartz()
    monkeypatch.setattr(pointer, "allowed", lambda: False)
    monkeypatch.setattr(pointer, "_quartz", lambda: fake)

    with pytest.raises(ToolError, match="Accessibility"):
        pointer.click(1, 1)

    assert fake.posted == []


def test_an_event_that_cannot_be_made_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pointer, "allowed", lambda: True)
    monkeypatch.setattr(pointer, "_quartz", lambda: FakeQuartz(creates=False))
    monkeypatch.setattr(pointer, "SETTLE", 0)

    with pytest.raises(ToolError, match="Accessibility"):
        pointer.click(1, 1)


def test_a_machine_without_the_library_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pointer, "allowed", lambda: True)
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    with pytest.raises(ToolError, match="only works on macOS"):
        pointer.click(1, 1)


def test_permission_is_unknown_without_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    assert pointer.allowed() is False
