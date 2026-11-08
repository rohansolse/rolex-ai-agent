"""Deterministic browser state for end-to-end command workflows and failure injection."""

from dataclasses import replace

from rolex.core.errors import ToolError
from rolex.tools.session import AmbiguousContactError, Contact, SendUncertainError, Target


class FixtureBrowser:
    def __init__(self) -> None:
        self.window = 41
        self.count = 3
        self.current = Target(41, 7, "about:blank", "", "")
        self.contacts = [
            Contact("nisha", "Nisha Shah", "Test fixture", 1, 1),
            Contact("sam-work", "Sam", "Work fixture", 2, 2),
            Contact("sam-home", "Sam", "Home fixture", 3, 3),
        ]
        self.sent: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str]] = []
        self.lose_ack = False
        self.submit_then_fail = False
        self.selection_fails = False
        self.focuses = 0

    def observe(self) -> Target:
        return self.current

    def focus(self) -> Target:
        self.focuses += 1
        self.current = Target(self.window, 8, "https://web.whatsapp.com/", "", "", signed_in=True)
        return self.current

    def tabs(self) -> tuple[int, int]:
        return self.window, self.count

    def open_tabs(self, count: int, window: int) -> tuple[int, int]:
        assert window == self.window
        self.count += count
        return self.window, self.count

    def select(self, name: str, expected: Target, choice: Contact | None = None) -> Target:
        if self.selection_fails:
            raise ToolError("Injected selection failure")
        if expected.identity != self.current.identity:
            raise ToolError("Target changed")
        hits = [contact for contact in self.contacts if name.casefold() in contact.label.casefold()]
        if choice:
            hits = [contact for contact in hits if contact.key == choice.key]
        if len(hits) > 1:
            raise AmbiguousContactError(hits)
        if not hits:
            raise ToolError("No fixture contact")
        self.current = Target(
            self.window,
            8,
            "https://web.whatsapp.com/",
            hits[0].label,
            "",
            signed_in=True,
            conversation_id=hits[0].key,
        )
        return self.current

    def write(self, expected: Target, text: str) -> Target:
        assert self.current == expected
        self.writes.append((expected.conversation_id, text))
        self.current = replace(expected, draft=text)
        return self.current

    def send(self, expected: Target) -> Target:
        assert expected == self.current
        if not self.lose_ack:
            self.sent.append((expected.conversation_id, expected.draft))
            self.current = replace(
                expected,
                draft="",
                outgoing=(*expected.outgoing, (str(len(self.sent)), expected.draft)),
            )
        if self.lose_ack or self.submit_then_fail:
            raise SendUncertainError("Acknowledgement lost")
        return self.current
