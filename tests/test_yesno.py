"""Reading a spoken yes or no.

Strict on purpose: mishearing a refusal as consent opens something the user
did not want, while mishearing consent as unclear only costs them repeating
themselves.
"""

import pytest

from rolex.voice.yesno import understand

YES = [
    "yes",
    "Yes.",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "Okay!",
    "go ahead",
    "do it",
    "open it",
    "please do",
    "yes please",
    "that's right",
]

NO = [
    "no",
    "No.",
    "nope",
    "nah",
    "don t",
    "dont do that",
    "no do not open it",
    "cancel",
    "stop",
    "wait",
    "not now",
    "never mind",
]

UNCLEAR = ["", "   ", "hmm", "maybe", "what time is it", "I was talking to someone else", "..."]


@pytest.mark.parametrize("said", YES)
def test_a_yes_is_understood(said: str) -> None:
    assert understand(said) is True


@pytest.mark.parametrize("said", NO)
def test_a_no_is_understood(said: str) -> None:
    assert understand(said) is False


@pytest.mark.parametrize("said", UNCLEAR)
def test_anything_unclear_is_neither(said: str) -> None:
    assert understand(said) is None


def test_a_no_anywhere_beats_a_yes() -> None:
    """ "no, please don't" contains "please". It is still a no."""
    assert understand("no please don't") is False
    assert understand("yeah no don't") is False


def test_case_and_punctuation_do_not_matter() -> None:
    assert understand("YES!!!") is True
    assert understand("  No,  ") is False
