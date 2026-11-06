"""The tool type and the individual tools."""

from datetime import UTC, datetime

from rolex.tools import get
from rolex.tools.base import Tool
from rolex.tools.clock import format_time, get_time


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


# --- the clock ------------------------------------------------------------


def test_time_is_formatted_for_reading_aloud() -> None:
    moment = datetime(2026, 11, 6, 16, 30, tzinfo=UTC)
    assert format_time(moment) == "Friday 6 November 2026, 4:30 PM"


def test_the_leading_zero_on_the_hour_is_dropped() -> None:
    moment = datetime(2026, 1, 2, 9, 5, tzinfo=UTC)
    assert format_time(moment) == "Friday 2 January 2026, 9:05 AM"


def test_get_time_returns_a_string() -> None:
    result = get_time()
    assert isinstance(result, str)
    assert result


def test_get_time_is_registered_and_read_only() -> None:
    tool = get("get_time")
    assert tool.safety == "read_only"
    assert "time" in tool.description.lower()
    assert tool.parameters == {"type": "object", "properties": {}}
