"""The tool type and the clock tool."""

from datetime import UTC, datetime
from typing import Any

from rolex.tools.base import Tool
from rolex.tools.clock import current_time, get_time


def test_a_tool_defaults_to_an_empty_object_schema() -> None:
    tool = Tool(name="t", description="d", run=lambda _: "x")
    assert tool.parameters == {"type": "object", "properties": {}}


def test_a_tool_defaults_to_read_only() -> None:
    assert Tool(name="t", description="d", run=lambda _: "x").safety == "read_only"


def test_tools_do_not_share_a_schema_dict() -> None:
    """A mutable default shared between tools would be a nasty surprise."""
    first = Tool(name="a", description="d", run=lambda _: "")
    second = Tool(name="b", description="d", run=lambda _: "")
    first.parameters["properties"]["x"] = {}
    assert second.parameters["properties"] == {}


def test_current_time_formats_a_fixed_moment() -> None:
    moment = datetime(2026, 11, 6, 16, 30, tzinfo=UTC)
    assert current_time({}, now=moment) == "Friday 6 November 2026, 4:30 PM"


def test_current_time_drops_the_leading_zero_on_the_hour() -> None:
    moment = datetime(2026, 1, 2, 9, 5, tzinfo=UTC)
    assert current_time({}, now=moment) == "Friday 2 January 2026, 9:05 AM"


def test_get_time_runs_and_returns_a_string() -> None:
    result: Any = get_time.run({})
    assert isinstance(result, str)
    assert result


def test_get_time_is_read_only() -> None:
    assert get_time.safety == "read_only"


def test_get_time_description_tells_the_model_when_to_use_it() -> None:
    assert "time" in get_time.description.lower()
