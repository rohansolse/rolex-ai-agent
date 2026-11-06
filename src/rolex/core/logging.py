"""Logging setup.

One function, called once from the entrypoint. Modules elsewhere just do
``logging.getLogger(__name__)`` and say nothing about handlers or formats.
"""

import logging
import sys

from rolex.core.errors import ConfigError


def configure_logging(level: str = "INFO") -> None:
    """Send rolex's logs to stderr at `level`.

    stderr, not stdout, so a piped conversation stays clean.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ConfigError(f"log level must be one of DEBUG, INFO, WARNING, ERROR; got {level!r}")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger("rolex")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False
