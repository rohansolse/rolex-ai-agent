"""Recognize explicit launch requests against installed application names."""

import re
from collections.abc import Sequence

_PREFIX = re.compile(
    r"^(?:(?:hello|hi|hey|please|pls)[,\s]+)*"
    r"(?:(?:i\s+(?:asked|aked|said|told\s+you)(?:\s+you)?\s+to)\s+)?",
    re.I,
)
_FORWARD = re.compile(
    r"^(?:open|launch|start|bring\s+up|खोलो|खोल|उघडा|उघड)\s+"
    r"(?:the\s+)?(?P<name>.+?)(?:\s+(?:app|application))?(?:\s+please)?$",
    re.I,
)
_REVERSE = re.compile(
    r"^(?P<name>.+?)\s+(?:open|launch|start|chalu|ughad|खोलो|खोल|उघडा|उघड|चालू)"
    r"(?:\s+(?:kar|kara|karo|कर|करा|करो))?$",
    re.I,
)


def requested_application(said: str) -> str | None:
    """Extract a launch target; questions, negation and compound steps don't match."""
    source = _PREFIX.sub("", said.strip().rstrip(".!?।"))
    match = _FORWARD.fullmatch(source) or _REVERSE.fullmatch(source)
    if match is None:
        return None
    target = match["name"].strip()
    if re.search(r"\b(?:and|then|aur|ani|aani)\b|[,;\n]|आणि|और", target, re.I):
        return None
    return target


def matches_installed(name: str, installed: Sequence[str]) -> bool:
    """Match whole name words, so `chat` cannot launch an app named ChatGPT."""
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
    return any(pattern.search(app) for app in installed)
