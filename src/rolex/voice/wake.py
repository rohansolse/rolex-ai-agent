"""Spotting the wake word in something that was heard.

Whisper does not hear "rolex" reliably: it is not an English word, and a
three-second clip gives little to go on. Measured against synthesised speech
it produced "Rolex", "Rulics" and "relax" for the same word, so a fixed list
of spellings does not hold — the next microphone produces one that is not on
it.

Two approaches were measured. Plain string similarity failed in both
directions at once: "rulics" scored 0.36 while "rules" scored 0.60, so no
threshold separated them. Comparing *consonant skeletons* did: "rolex",
"rulics", "relax", "rollex", "roleks", "rolax", "rolix", "rollix", "ruleks"
and "roelex" all reduce to ``rlks``, while "rules", "release", "rolling",
"robot", "rohan", "right", "ready", "rock", "open", "search" and "relaxed" do
not. That is what is used.

Being wrong is cheap in one direction and expensive in the other: a false wake
means rolex listens for a command that never comes, while a missed wake means
it ignored the user.
"""

import re

WAKE_WORD = "rolex"

#: Letters that carry no information about how a word sounds here. Vowels
#: because whisper guesses them from almost nothing, 'h' because it is silent
#: in the middle of a word, 'y' because it behaves as a vowel.
_SILENT = re.compile(r"[aeiouyh]")

#: Spellings of the same sound. Applied before vowels are dropped.
_SOUNDS = (("x", "ks"), ("c", "k"), ("q", "k"), ("ph", "f"), ("z", "s"))

#: A run of the same letter sounds like one of it: "rollex" and "rolex".
_DOUBLED = re.compile(r"(.)\1+")

_WORDS = re.compile(r"[a-z]+")


def skeleton(word: str) -> str:
    """`word` reduced to the consonants that decide how it sounds."""
    reduced = word.casefold()
    for spelling, sound in _SOUNDS:
        reduced = reduced.replace(spelling, sound)
    reduced = _SILENT.sub("", reduced)
    return _DOUBLED.sub(r"\1", reduced)


#: Computed once; every heard word is compared against this.
WAKE_SKELETON = skeleton(WAKE_WORD)

#: Heard as two words rather than one. Checked first, longest first, so
#: "roll ex open notes" does not consume only "roll".
PHRASES = ("roll ex", "role x", "raw lex", "rule ex", "row lex")


def heard(text: str) -> bool:
    """Whether `text` begins with something that sounds like the wake word."""
    return strip(text) is not None


def resembles(word: str) -> bool:
    """Whether a single heard word sounds like the wake word."""
    return skeleton(word) == WAKE_SKELETON


def strip(text: str) -> str | None:
    """`text` with the wake word removed, or None if it was not there.

    Only the beginning counts: "rolex, open chrome" wakes it while "I bought a
    Rolex" does not, which is the difference between a wake word and a
    keyword.

    Returns an empty string when the wake word was all that was said. The user
    has rolex's attention and has not asked for anything yet, which is not the
    same as never having called it.
    """
    words = _WORDS.findall(text.casefold())
    if not words:
        return None

    for phrase in sorted(PHRASES, key=lambda p: -len(p.split())):
        parts = phrase.split()
        if words[: len(parts)] == parts:
            return _rest(text, len(parts))

    if resembles(words[0]):
        return _rest(text, 1)

    return None


def _rest(text: str, used: int) -> str:
    """What is left of `text` after its first `used` words."""
    remaining = text
    for _ in range(used):
        match = _WORDS.search(remaining.casefold())
        if match is None:
            return ""
        remaining = remaining[match.end() :]
    return remaining.lstrip(" ,.:;-—").strip()
