"""Logging setup.

One function, called once from the entrypoint. Modules elsewhere just do
``logging.getLogger(__name__)`` and say nothing about handlers or formats.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from rolex.core.errors import ConfigError


def configure_logging(level: str = "INFO", *, structured: bool = False) -> None:
    """Send rolex's logs to stderr at `level`.

    stderr, not stdout, so a piped conversation stays clean.
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ConfigError(f"log level must be one of DEBUG, INFO, WARNING, ERROR; got {level!r}")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if structured else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    logger = logging.getLogger("rolex")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False


class JsonFormatter(logging.Formatter):
    """Stable diagnostic fields without conversation contents or arguments."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for name in ("round", "tool", "outcome", "elapsed_ms", "error_type", "messages"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        return json.dumps(payload, ensure_ascii=False)
