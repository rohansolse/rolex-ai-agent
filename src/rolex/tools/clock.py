"""Telling the time.

A model cannot know the time, so an answer that is right can only have come
from calling this.
"""

from datetime import datetime

from rolex.tools.registry import register

DESCRIPTION = "Get the current date and time. Use only when asked what time or what day it is."
"""Short on purpose. A longer description with "do not use it for..."
warnings scored no better at getting the model to choose correctly, and a 3B
model that does not understand the question sometimes answers by copying the
description back at the user. Measured: the long version did exactly that."""


@register(description=DESCRIPTION)
def get_time() -> str:
    """The local date and time, written out for a model to read aloud."""
    return format_time(datetime.now().astimezone())


def format_time(moment: datetime) -> str:
    """`moment` written the way rolex says it. Separate so it can be tested."""
    return moment.strftime("%A %d %B %Y, %I:%M %p").replace(" 0", " ")
