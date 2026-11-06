"""Deciding whether a tool may run.

The model chooses what to do; this decides whether it is allowed to. Keeping
the two apart matters: a model talked into asking for something destructive
still has to get past a rule it cannot see or argue with.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rolex.tools.base import Safety, Tool

Asker = Callable[[Tool, Mapping[str, Any]], bool]
"""Puts the question to the user and returns their answer."""


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a tool may run, and why not if it may not."""

    allowed: bool
    reason: str = ""


ALLOWED = Decision(allowed=True)


@dataclass(frozen=True, slots=True)
class Policy:
    """What rolex may do without being asked each time.

    Defaults are deliberately cautious. ``read_only`` runs freely because
    looking at something cannot go wrong; ``mutating`` asks, because the user
    should see "open Safari?" before it happens; ``destructive`` is refused
    outright until explicitly switched on, because a model that has
    misunderstood should not be one confirmation away from deleting something.
    """

    ask: Asker | None = None
    """How to ask. ``None`` means nothing can ask, so anything that would need
    asking is refused — which is the right answer for a non-interactive run."""

    auto_approve: frozenset[Safety] = frozenset({"read_only"})
    allow_destructive: bool = False

    def decide(self, tool: Tool, arguments: Mapping[str, Any]) -> Decision:
        """Whether `tool` may run with `arguments`."""
        if tool.safety == "destructive" and not self.allow_destructive:
            return Decision(
                allowed=False,
                reason=(
                    f"Not done: '{tool.name}' can destroy or send something, and rolex is "
                    f"configured to refuse that. Tell the user it was not done, and that "
                    f"they can allow it with ROLEX_ALLOW_DESTRUCTIVE=true."
                ),
            )

        if tool.safety in self.auto_approve:
            return ALLOWED

        if self.ask is None:
            return Decision(
                allowed=False,
                reason=(
                    f"Not done: '{tool.name}' needs the user's go-ahead and there is no way "
                    f"to ask them right now. Tell the user what you wanted to do."
                ),
            )

        if self.ask(tool, arguments):
            return ALLOWED

        return Decision(
            allowed=False,
            reason=(
                f"Not done: the user refused permission to run '{tool.name}'. "
                f"Tell them it was not done. Do not try again and do not claim "
                f"it already happened."
            ),
        )


def always_allow() -> Policy:
    """A policy that permits everything. For tests and scripted runs."""
    return Policy(
        auto_approve=frozenset({"read_only", "mutating", "destructive"}), allow_destructive=True
    )
