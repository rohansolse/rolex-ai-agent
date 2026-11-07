"""Commands rolex carries out without asking a model what to do.

Measured, not assumed: on seven ordinary instructions, `qwen2.5:3b` and
`qwen2.5:7b` both chose the right tool five times, and the 7B needed fifty
seconds to get one of them wrong. "Close Chrome" does not need a model's
judgement — it is a rule. Routing the handful of things the user says most
through a rule makes them work every time, and leaves the model the jobs it
is actually good at.

Anything not matched here falls through to the agent loop as before.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

#: What was said, reduced for matching: lower case, no punctuation.
_TIDY = re.compile(r"[^a-z0-9\s]+")


@dataclass(frozen=True, slots=True)
class Command:
    """One thing rolex recognises without thinking about it."""

    name: str
    patterns: tuple[str, ...]
    """Regular expressions, tried against the tidied sentence."""

    must_be_about: tuple[str, ...] = ()
    """Words that must NOT follow, because they mean something else.

    "Reply to this" is a command; "answer my question about python" is a
    question. Without this the router claims both."""

    takes_rest: bool = False
    """Whether what follows the match is passed on as an instruction."""


@dataclass(frozen=True, slots=True)
class Match:
    """A recognised command and anything the user added to it."""

    name: str
    rest: str = ""


#: What the command is allowed to be followed by. Anything else is a
#: sentence that merely contains the word "reply", and belongs to the model.
_THIS_CHAT = r"(?:this|that|them|him|her|the\s+client|the\s+chat|the\s+message)"

COMMANDS: tuple[Command, ...] = (
    Command(
        name="reply",
        patterns=(
            # "draft a reply", "write me a response" — and nothing after the
            # noun except an instruction about what to say.
            # "draft a reply", "write me a response", optionally "to this".
            # Anything else after the noun — "an answer explaining recursion"
            # — is a different request and belongs to the model.
            rf"^(?:please\s+)?(?:draft|write|make|give)\s+(?:me\s+)?(?:a|an|the)?\s*"
            rf"(?:reply|response|answer)(?:\s+(?:to\s+)?{_THIS_CHAT})?"
            rf"(?=$|\s+(?:saying|with|and\s+say|that)\b)",
            # "reply to this", "respond to them", "answer this". The target
            # has to be this conversation: "answer my question about python"
            # and "reply to the email from john" are not this command.
            rf"^(?:reply|respond|answer)(?:\s+to)?\s+{_THIS_CHAT}(?=\s|$)",
            # "reply" / "respond" on its own, with nothing or an instruction
            # after it.
            r"^(?:reply|respond)(?=\s+(?:saying|with|and\s+say)\b|$)",
            # "what should I reply", "how should I respond"
            # "what should I reply", and only about this chat. The target
            # stays optional but then nothing may follow: "what should I
            # reply to the email from john" is not this command.
            rf"^(?:what|how)\s+should\s+i\s+(?:reply|respond|answer)"
            rf"(?:\s+(?:to\s+)?{_THIS_CHAT})?(?=$|\s+(?:saying|with)\b)",
        ),
        takes_rest=True,
    ),
)


def understand(said: str) -> Match | None:
    """The command `said` is, or None to let the agent loop handle it."""
    tidied, offsets = _tidy(said)
    if not tidied:
        return None

    for command in COMMANDS:
        for pattern in command.patterns:
            found = re.match(pattern, tidied)
            if found is None:
                continue
            rest = _rest_of(said, offsets, found.end()) if command.takes_rest else ""
            return Match(name=command.name, rest=_tidy_rest(rest))
    return None


def _tidy(said: str) -> tuple[str, list[int]]:
    """`said` reduced for matching, and where each kept character came from.

    The offsets matter: the instruction the user added is sliced out of the
    *original*, not the tidied text. Taking it from the tidied text loses
    punctuation and splits numbers — "the price is Rs 2,500" became "rs 2
    500", which is the exact kind of mangled figure this module exists to
    keep out of a client's chat.
    """
    kept: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(said.casefold()):
        reduced = character if (character.isalnum() and character.isascii()) else " "
        if reduced == " " and kept and kept[-1] == " ":
            continue
        if reduced == " " and not kept:
            continue
        kept.append(reduced)
        offsets.append(index)

    while kept and kept[-1] == " ":
        kept.pop()
        offsets.pop()
    return "".join(kept), offsets


def _rest_of(said: str, offsets: list[int], position: int) -> str:
    """What the user added after the command, in their own words."""
    if position >= len(offsets):
        return ""
    return said[offsets[position] :].strip()


def _tidy_rest(rest: str) -> str:
    """Drop the joining words that start an added instruction."""
    for opener in ("saying that", "saying", "and say that", "and say", "that", "to"):
        if rest.startswith(f"{opener} "):
            return rest[len(opener) + 1 :].strip()
    return rest


def handles(said: str) -> bool:
    """Whether this would be handled without the model choosing."""
    return understand(said) is not None


#: Filled in by the interface layer, which knows how to run each one.
Runner = Callable[[Match], str]
