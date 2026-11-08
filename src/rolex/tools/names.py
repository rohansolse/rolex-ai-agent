"""Conservative Devanagari-to-Latin search spelling for contact names.

Only contact search keys are transliterated. Dictated messages stay verbatim.
A failed search still requires clarification rather than selecting a different name.
"""

import re

_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "ळ": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
}
_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
_SIGNS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॅ": "a",
    "ॉ": "o",
}


def search_name(name: str) -> str:
    if not re.search(r"[\u0900-\u097f]", name):
        return name
    words: list[str] = []
    for word in name.split():
        pieces: list[str] = []
        for character in word:
            if character in _CONSONANTS:
                pieces.append(_CONSONANTS[character] + "a")
            elif character in _VOWELS:
                pieces.append(_VOWELS[character])
            elif character in _SIGNS and pieces:
                pieces[-1] = pieces[-1].removesuffix("a") + _SIGNS[character]
            elif character == "्" and pieces:
                pieces[-1] = pieces[-1].removesuffix("a")
            elif character in {"ं", "ँ"}:
                pieces.append("n")
            elif character not in {"़", "\u0903"}:
                pieces.append(character)
        if word and word[-1] in _CONSONANTS and pieces:
            pieces[-1] = pieces[-1].removesuffix("a")
        words.append(re.sub(r"([aiu])\1+", r"\1", "".join(pieces)))
    return " ".join(words)
