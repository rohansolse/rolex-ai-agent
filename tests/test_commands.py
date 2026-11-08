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


# --- sending --------------------------------------------------------------


@pytest.mark.parametrize("said", ["send it", "send", "send that", "send the message"])
def test_sending_what_is_already_typed_is_recognised(said: str) -> None:
    match = understand(said)
    assert match is not None
    assert match.name == "send"
    assert not match.who


@pytest.mark.parametrize(
    ("said", "who", "text"),
    [
        ("send hi to rohan solse", "rohan solse", "hi"),
        ("send a message to priya saying hello how are you", "priya", "hello how are you"),
        ("message Priya saying hi", "Priya", "hi"),
        ("tell rohan that we are closed tomorrow", "rohan", "we are closed tomorrow"),
        ("whatsapp the testing ai group saying done", "testing ai", "done"),
    ],
)
def test_who_and_what_are_separated(said: str, who: str, text: str) -> None:
    match = understand(said)
    assert match is not None
    assert match.name == "send_to"
    assert match.who == who
    assert match.rest == text


@pytest.mark.parametrize("said", ["send something", "message", "tell"])
def test_half_a_sentence_is_left_to_the_model(said: str) -> None:
    """Guessing here would send the wrong words to the wrong person."""
    match = understand(said)
    assert match is None or match.name != "send_to"


@pytest.mark.parametrize(
    ("said", "who", "text"),
    [
        # The person comes first after "tell" and "message". Read the other
        # way round, this becomes a message to somebody called "call me".
        ("tell rohan to call me", "rohan", "call me"),
        ("message rohan to confirm tomorrow", "rohan", "confirm tomorrow"),
        # And after "send" the message comes first.
        ("send hi to rohan solse", "rohan solse", "hi"),
    ],
)
def test_the_order_depends_on_the_verb(said: str, who: str, text: str) -> None:
    match = understand(said)
    assert match is not None
    assert (match.who, match.rest) == (who, text)


@pytest.mark.parametrize("said", ["send that to rohan", "send it to priya"])
def test_sending_what_is_typed_to_someone_named(said: str) -> None:
    """Falling through to the bare "send" would drop the recipient and
    post it to whichever chat happened to be in front."""
    match = understand(said)
    assert match is not None
    assert match.name == "send_to"
    assert match.who
    assert match.rest == "", "the words are whatever is in the box"


@pytest.mark.parametrize(
    "said",
    ["tell me what to do", "tell me how to open chrome", "message me the details"],
)
def test_tell_me_is_a_question_not_a_message(said: str) -> None:
    match = understand(said)
    assert match is None or match.name != "send_to"


@pytest.mark.parametrize(
    "said",
    [
        "yes pleas send it on whatapp",
        "yes please send it on WhatsApp",
        "can you send it",
        "type it there and send it",
    ],
)
def test_natural_followup_sending_is_routed(said: str) -> None:
    match = understand(said)
    assert match is not None
    assert match.name == "send"
    assert match.rest == ""


def test_named_type_followup_keeps_the_recipient() -> None:
    match = understand("can you type this msg in Testing AI")
    assert match is not None
    assert match.name == "type_previous"
    assert match.who == "Testing AI"


@pytest.mark.parametrize(
    "said",
    [
        "send it but do not actually send anything",
        "type this Python expression",
        "what does send it mean",
    ],
)
def test_unrelated_or_qualified_followups_are_not_sent(said: str) -> None:
    match = understand(said)
    assert match is None or match.name not in {"send", "send_to", "type_previous"}


@pytest.mark.parametrize(
    "said, who, text",
    [
        ("can you say hi in that chat", "", "hi"),
        ("say Hi! in this chat", "", "Hi!"),
        ("can you say hi in whatsapp chat of rohan solse", "rohan solse", "hi"),
        (
            "please say Happy Diwali! in the WhatsApp chat with Rohan Solse",
            "Rohan Solse",
            "Happy Diwali!",
        ),
        ("say The price is Rs 2,500. in that chat", "", "The price is Rs 2,500."),
    ],
)
def test_saying_a_message_in_a_chat_is_a_send_request(said: str, who: str, text: str) -> None:
    match = understand(said)
    assert match is not None
    assert match.name == ("send_to" if who else "send")
    assert match.who == who
    assert match.rest == text


@pytest.mark.parametrize(
    "said",
    [
        "can you say hi",
        "say hi in French",
        "can you say hi in that chat without sending",
    ],
)
def test_saying_hi_without_a_chat_send_target_is_conversation(said: str) -> None:
    assert understand(said) is None


@pytest.mark.parametrize(
    "said",
    [
        "please open new tab in whatapp chat",
        "open a new tab in WhatsApp chat",
        "can you open another tab in WhatsApp",
        "open the first tab in Chrome chat",
    ],
)
def test_browser_tabs_are_not_chat_names(said: str) -> None:
    match = understand(said)
    assert match is None or match.name != "open_chat"
