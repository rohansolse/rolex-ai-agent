"""Tools the agent can call. Must not import rolex.llm.

Importing this package declares every tool: each module's ``@register``
decorators run on import, and :func:`rolex.tools.registry.registered` then
knows about them. Add a module here when you add one.
"""

from rolex.tools import clock, mac, system  # noqa: F401 - imported to register
from rolex.tools.base import Safety, Tool
from rolex.tools.registry import get, registered, with_safety

__all__ = ["Safety", "Tool", "get", "registered", "with_safety"]
