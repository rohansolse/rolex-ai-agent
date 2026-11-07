"""Recognising the few things the user says most often.

These go round the model on purpose. Measured over seven instructions,
`qwen2.5:3b` and `qwen2.5:7b` each picked the right tool five times, and the
7B took fifty seconds to get one wrong. "Reply to this" is a rule.
"""

import pytest

from rolex.agent.commands import handles, understand

REPLY = [
    "reply to this",
    "reply to them",
    "reply to the client",
    "respond to this",
    "answer this",
    "draft a reply",
    "write a reply",
    "make a response",
    "give me an answer",
    "please draft a reply",
    "what should I reply",
    "how should i respond",
]

NOT_A_COMMAND = [
    # "say" is too broad on its own: "what should I say to my wife" is not
    # about the chat in front of us.
    "what should I say",
    "what should I say to my wife",
    "answer my question about python",
    "answer me this what is 2 plus 2",
    "respond to the email from john",
    "reply to the email from john",
    "open chrome",
    "what time is it",
    "close the whatsapp tab",
    "how much battery do I have",
    "who are you",
    "",
    "   ",
    "tell me a joke about replies",
]


@pytest.mark.parametrize("said", REPLY)
def test_asking_for_a_reply_is_recognised(said: str) -> None:
    match = understand(said)
    assert match is not None
    assert match.name == "reply"


@pytest.mark.parametrize("said", NOT_A_COMMAND)
def test_everything_else_goes_to_the_model(said: str) -> None:
    """Falling through is the default: only what is certain is routed."""
    assert understand(said) is None
    assert not handles(said)


@pytest.mark.parametrize(
    ("said", "instruction"),
    [
        ("write a reply saying we can do it by friday", "we can do it by friday"),
        ("answer this saying yes", "yes"),
        ("reply to this and say the price is fixed", "the price is fixed"),
        ("draft a reply that we are closed sunday", "we are closed sunday"),
        ("reply to this", ""),
    ],
)
def test_what_the_user_added_is_kept(said: str, instruction: str) -> None:
    """ "Say we can do it by Friday" is the whole point of the sentence."""
    match = understand(said)
    assert match is not None
    assert match.rest == instruction


def test_punctuation_and_case_do_not_matter() -> None:
    """It arrives from speech, where both are whisper's guess."""
    assert understand("Reply to this.") is not None
    assert understand("REPLY TO THEM!") is not None


@pytest.mark.parametrize(
    ("said", "instruction"),
    [
        ("reply saying the price is Rs 2,500", "the price is Rs 2,500"),
        ("write a reply saying Friday after 5:30pm", "Friday after 5:30pm"),
        ("answer this saying it is 60% done", "it is 60% done"),
    ],
)
def test_the_instruction_keeps_the_users_own_words(said: str, instruction: str) -> None:
    """Taken from the tidied sentence instead, "Rs 2,500" became "rs 2 500"
    — a mangled figure, in the one module written to keep invented numbers
    out of a client's chat."""
    match = understand(said)
    assert match is not None
    assert match.rest == instruction


# --- opening a chat by name -----------------------------------------------


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("open testing ai group", "testing ai"),
        ("open the testing ai group", "testing ai"),
        ("open chat with Priya Deshmukh", "Priya Deshmukh"),
        ("go to the MBRJ Self chat", "MBRJ Self"),
        ("switch to testing ai group", "testing ai"),
        ("show me the testing ai conversation", "testing ai"),
        ("open the whatsapp group testing ai", "testing ai"),
    ],
)
def test_opening_a_chat_is_recognised(said: str, name: str) -> None:
    """The model was clicking numbered items at random instead of
    searching, and the wrong conversation is how a message reaches the
    wrong person."""
    match = understand(said)
    assert match is not None
    assert match.name == "open_chat"
    assert match.rest == name


@pytest.mark.parametrize(
    "said",
    [
        "open chrome",
        "open safari",
        "open https://example.com",
        "open the browser",
        "open",
    ],
)
def test_opening_other_things_is_left_to_the_model(said: str) -> None:
    match = understand(said)
    assert match is None or match.name != "open_chat"


def test_the_word_group_is_not_part_of_the_name() -> None:
    """WhatsApp does not have "group" in the name."""
    match = understand("open the testing ai group")
    assert match is not None
    assert "group" not in match.rest
