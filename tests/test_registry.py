"""Declaring tools, and the schema derived from their signatures."""

import pytest

from rolex.core.errors import ToolError
from rolex.tools.base import Tool
from rolex.tools.registry import Registry, schema_of, with_safety


@pytest.fixture
def registry() -> Registry:
    """A registry of this test's own, so the real one is never touched."""
    return Registry()


# --- declaring ------------------------------------------------------------


def test_a_decorated_function_becomes_a_tool(registry: Registry) -> None:
    @registry.register(description="does a thing")
    def thing() -> str:
        return "done"

    tool = registry.get("thing")
    assert tool.name == "thing"
    assert tool.description == "does a thing"
    assert tool.run({}) == "done"


def test_the_function_is_returned_unchanged(registry: Registry) -> None:
    """It must stay directly callable, so tests need not go via the registry."""

    @registry.register(description="d")
    def thing() -> str:
        return "called directly"

    assert thing() == "called directly"


def test_the_name_can_be_overridden(registry: Registry) -> None:
    @registry.register(description="d", name="nicer_name")
    def ugly_internal_name() -> str:
        return ""

    assert registry.get("nicer_name").name == "nicer_name"


def test_safety_defaults_to_read_only_and_can_be_set(registry: Registry) -> None:
    @registry.register(description="d")
    def safe() -> str:
        return ""

    @registry.register(description="d", safety="destructive")
    def risky() -> str:
        return ""

    assert registry.get("safe").safety == "read_only"
    assert registry.get("risky").safety == "destructive"


def test_a_duplicate_name_is_refused(registry: Registry) -> None:
    @registry.register(description="d")
    def twice() -> str:
        return ""

    with pytest.raises(ToolError, match="two tools are called 'twice'"):

        @registry.register(description="d", name="twice")
        def other() -> str:
            return ""


def test_an_unknown_name_is_an_error(registry: Registry) -> None:
    with pytest.raises(ToolError, match="no tool called 'ghost'"):
        registry.get("ghost")


def test_registered_lists_in_declaration_order(registry: Registry) -> None:
    @registry.register(description="d")
    def first() -> str:
        return ""

    @registry.register(description="d")
    def second() -> str:
        return ""

    assert [tool.name for tool in registry.tools()] == ["first", "second"]


# --- schema from the signature -------------------------------------------


def test_no_parameters_gives_an_empty_object() -> None:
    def nothing() -> str:
        return ""

    assert schema_of(nothing) == {"type": "object", "properties": {}}


def test_each_python_type_maps_to_a_json_type() -> None:
    def every(text: str, count: int, ratio: float, flag: bool) -> str:
        return ""

    assert schema_of(every)["properties"] == {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
    }


def test_a_parameter_without_a_default_is_required() -> None:
    def mixed(needed: str, optional: str = "x") -> str:
        return ""

    assert schema_of(mixed)["required"] == ["needed"]


def test_no_required_key_when_everything_has_a_default() -> None:
    def all_optional(a: str = "x") -> str:
        return ""

    assert "required" not in schema_of(all_optional)


def test_an_optional_type_is_nullable_and_not_required() -> None:
    def maybe(value: str | None = None) -> str:
        return ""

    schema = schema_of(maybe)
    assert schema["properties"]["value"]["type"] == "string"
    assert "required" not in schema


def test_an_undescribable_type_is_refused_at_declaration() -> None:
    """Better the author sees this than the model does."""

    def weird(payload: dict[str, int]) -> str:
        return ""

    with pytest.raises(ToolError, match="cannot be described"):
        schema_of(weird)


def test_a_missing_type_hint_is_refused() -> None:
    def untyped(value) -> str:  # type: ignore[no-untyped-def]
        return ""

    with pytest.raises(ToolError, match="has no type hint"):
        schema_of(untyped)


def test_varargs_are_refused() -> None:
    def splat(*args: str) -> str:
        return ""

    with pytest.raises(ToolError, match=r"\*args"):
        schema_of(splat)


# --- calling --------------------------------------------------------------


def test_arguments_are_passed_through_by_name(registry: Registry) -> None:
    @registry.register(description="d")
    def greet(who: str, loudly: bool = False) -> str:
        return f"hi {who}{'!' if loudly else ''}"

    assert registry.get("greet").run({"who": "Rohan", "loudly": True}) == "hi Rohan!"


def test_a_default_is_used_when_the_argument_is_absent(registry: Registry) -> None:
    @registry.register(description="d")
    def greet(who: str = "world") -> str:
        return f"hi {who}"

    assert registry.get("greet").run({}) == "hi world"


def test_an_unexpected_argument_is_reported_usefully(registry: Registry) -> None:
    """A small model invents argument names; the message has to teach it."""

    @registry.register(description="d")
    def greet(who: str) -> str:
        return ""

    with pytest.raises(ToolError, match="unexpected argument"):
        registry.get("greet").run({"who": "x", "volume": "loud"})


def test_the_error_lists_what_the_tool_actually_takes(registry: Registry) -> None:
    @registry.register(description="d")
    def greet(who: str) -> str:
        return ""

    with pytest.raises(ToolError, match="It takes: who"):
        registry.get("greet").run({"nope": "x"})


def test_a_missing_required_argument_is_reported(registry: Registry) -> None:
    @registry.register(description="d")
    def greet(who: str) -> str:
        return ""

    with pytest.raises(ToolError, match="called wrongly"):
        registry.get("greet").run({})


# --- filtering ------------------------------------------------------------


def test_with_safety_keeps_only_the_allowed_classes() -> None:
    tools = [
        Tool(name="look", description="d", run=lambda _: "", safety="read_only"),
        Tool(name="change", description="d", run=lambda _: "", safety="mutating"),
        Tool(name="destroy", description="d", run=lambda _: "", safety="destructive"),
    ]

    kept = with_safety(tools, ["read_only", "mutating"])

    assert [tool.name for tool in kept] == ["look", "change"]
