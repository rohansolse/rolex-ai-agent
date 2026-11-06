"""Spotting the wake word.

The cases are the spellings whisper actually produced when spoken to, plus the
ordinary words most likely to begin a sentence addressed to a computer. Both
lists matter: one guards against ignoring the user, the other against waking
at nothing.
"""

import pytest

from rolex.voice.wake import WAKE_SKELETON, heard, resembles, skeleton, strip

MISHEARINGS = [
    "rolex",
    "Rolex",
    "ROLEX",
    "rulics",
    "relax",
    "rollex",
    "roleks",
    "rolax",
    "rolix",
    "rollix",
    "ruleks",
    "roelex",
]

NOT_THE_WAKE_WORD = [
    "open",
    "what",
    "hello",
    "search",
    "time",
    "rules",
    "relaxed",
    "close",
    "play",
    "rohan",
    "robot",
    "right",
    "really",
    "ready",
    "release",
    "reluctant",
    "rolling",
    "rock",
]


@pytest.mark.parametrize("word", MISHEARINGS)
def test_every_observed_mishearing_wakes_it(word: str) -> None:
    assert resembles(word), f"{word!r} reduces to {skeleton(word)!r}, not {WAKE_SKELETON!r}"


@pytest.mark.parametrize("word", NOT_THE_WAKE_WORD)
def test_ordinary_words_do_not_wake_it(word: str) -> None:
    assert not resembles(word)


def test_the_skeleton_drops_vowels_and_collapses_doubles() -> None:
    assert skeleton("rolex") == "rlks"
    assert skeleton("rollex") == "rlks"
    assert skeleton("RULICS") == "rlks"


# --- pulling the command out of what was heard ---------------------------


@pytest.mark.parametrize(
    ("said", "command"),
    [
        ("rolex open chrome", "open chrome"),
        ("Rolex, open Chrome.", "open Chrome."),
        ("Rulics, what time is it?", "what time is it?"),
        ("relax how much battery do I have", "how much battery do I have"),
        ("roll ex open notes", "open notes"),
        ("rolex   —  search the web", "search the web"),
    ],
)
def test_the_wake_word_is_removed(said: str, command: str) -> None:
    assert strip(said) == command


def test_the_wake_word_alone_means_it_is_listening() -> None:
    """Different from not being called: the user has its attention."""
    assert strip("rolex") == ""
    assert strip("Rolex.") == ""


@pytest.mark.parametrize(
    "said",
    [
        "open chrome",
        "I bought a Rolex watch",
        "what time is it",
        "",
        "   ",
        "...",
    ],
)
def test_speech_that_was_not_addressed_to_rolex_is_ignored(said: str) -> None:
    assert strip(said) is None
    assert not heard(said)


@pytest.mark.parametrize(
    ("said", "command"),
    [
        ("hello rolex what is the time today", "what is the time today"),
        ("hey rolex open chrome", "open chrome"),
        ("okay rolex open notes", "open notes"),
        ("Hello Rolex, what is the time?", "what is the time?"),
    ],
)
def test_a_greeting_before_the_wake_word_is_allowed(said: str, command: str) -> None:
    """People say "hello rolex" at least as often as "rolex"."""
    assert strip(said) == command


@pytest.mark.parametrize(
    "said",
    [
        "I bought a Rolex watch",
        "my friend has a rolex",
        "so I told him rolex is great",
    ],
)
def test_the_wake_word_must_still_come_near_the_start(said: str) -> None:
    """Otherwise it is a keyword, not a wake word."""
    assert strip(said) is None


def test_a_two_word_mishearing_does_not_eat_only_half() -> None:
    assert strip("roll ex open notes") == "open notes"
