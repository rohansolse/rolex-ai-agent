"""Exception hierarchy for rolex.

Every error rolex raises on purpose inherits from :class:`RolexError`, so a
caller can catch one type and know the failure came from us rather than from a
library or the interpreter.
"""


class RolexError(Exception):
    """Base class for every error rolex raises deliberately."""


class ConfigError(RolexError):
    """Configuration is missing, malformed, or contradictory."""


class ProviderError(RolexError):
    """An LLM provider could not be reached or returned something unusable."""


class ToolError(RolexError):
    """A tool failed while running, or was called with bad arguments."""


class UnreachableError(ProviderError):
    """The backend's server could not be contacted at all.

    Separate from :class:`ProviderError` because it is the one failure with an
    obvious fix — start the server — and callers add that advice.
    """
