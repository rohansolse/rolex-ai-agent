"""Telling the time.

The simplest possible real tool, and the one Phase 2 uses to prove the agent
loop works: a model cannot know the time, so an answer that is right can only
have come from calling this.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rolex.tools.base import Tool


def current_time(_: Mapping[str, Any], *, now: datetime | None = None) -> str:
    """The local date and time, written out for a model to read aloud.

    Args:
        _: No arguments. Present because every tool is called the same way.
        now: Override the clock. Tests pass a fixed time; nothing else should.
    """
    moment = datetime.now().astimezone() if now is None else now
    return moment.strftime("%A %d %B %Y, %I:%M %p").replace(" 0", " ")


get_time = Tool(
    name="get_time",
    description=(
        "Get the current date and time on this computer. "
        "Use this whenever the user asks what time or what day it is."
    ),
    run=current_time,
    safety="read_only",
)
