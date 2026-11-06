"""What a tool is.

A tool is a plain function plus enough description for a model to decide to
call it. Nothing here imports :mod:`rolex.llm`: tools must stay usable, and
testable, without a model anywhere near them. The agent is what translates a
tool into the schema a provider wants.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Safety = Literal["read_only", "mutating", "destructive"]
"""How much damage running this could do.

- ``read_only`` — looks at something and changes nothing. Runs unasked.
- ``mutating`` — changes the machine in a way the user would notice but would
  not mind undoing. Opening an app.
- ``destructive`` — deletes, overwrites, or sends something outward.

Phase 2 ships only a ``read_only`` tool. The policy that acts on this arrives
with the registry in Phase 3; the field exists now so no tool is ever written
without its author having to think about the answer.
"""


@dataclass(frozen=True, slots=True)
class Tool:
    """One thing rolex can do."""

    name: str
    description: str
    """Written for the model, not for a developer. It is the only thing the
    model has to decide whether this tool is the right one."""

    run: Callable[[Mapping[str, Any]], str]
    """Does the work and returns what the model should be told.

    Returns a string because that is what goes back into the conversation.
    """

    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON Schema for the arguments."""

    safety: Safety = "read_only"
