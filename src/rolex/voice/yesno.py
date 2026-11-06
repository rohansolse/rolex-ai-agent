"""Reading a spoken yes or no.

Used when rolex asks permission out loud. Deliberately strict: anything that
is not clearly a yes is not a yes. Mishearing "no" as "yes" opens an app the
user did not want; mishearing "yes" as unclear only means asking again.
"""

import re

YES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "ya",
        "sure",
        "ok",
        "okay",
        "please",
        "fine",
        "go",
        "do",
        "open",
        "allow",
        "correct",
        "right",
        "affirmative",
    }
)
"""Words that mean yes on their own. "open" and "do" are here because "do it"
and "open it" are what people actually say to an assistant."""

NO = frozenset(
    {
        "no",
        "nope",
        "nah",
        "never",
        "stop",
        "cancel",
        "don",
        "dont",
        "negative",
        "wait",
        "not",
    }
)
"""Words that mean no. "don" is there because whisper writes "don't" as
"don t" once the apostrophe is stripped."""

_WORDS = re.compile(r"[a-z]+")


def understand(said: str) -> bool | None:
    """True for yes, False for no, None when it was not clear.

    A no anywhere in the sentence beats a yes, because "no, don't open it"
    contains neither word first and must not be read as consent.
    """
    words = set(_WORDS.findall(said.casefold()))
    if not words:
        return None

    if words & NO:
        return False
    if words & YES:
        return True
    return None
