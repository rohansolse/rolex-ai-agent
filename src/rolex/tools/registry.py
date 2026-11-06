"""Registering tools, and describing them to a model.

A tool is declared by decorating a function. The JSON Schema the model needs
is derived from that function's signature, so the description the model sees
and the code that runs can never drift apart — which they do immediately when
a schema is written out by hand beside the function.

    @register(description="Open an app by name.", safety="mutating")
    def open_app(name: str) -> str:
        ...
"""

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Union, get_args, get_origin, get_type_hints

from rolex.core.errors import ToolError
from rolex.tools.base import Safety, Tool

#: Python type to JSON Schema type. Anything else is refused at declaration
#: time rather than confusing the model at run time.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class Registry:
    """A collection of declared tools.

    A class rather than module-level state so a test can have its own and
    never touch the real one. The application uses the single :data:`REGISTRY`
    below, which is what :func:`register` writes to.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        *,
        description: str,
        safety: Safety = "read_only",
        name: str | None = None,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Declare a function as a tool.

        The function is returned unchanged, so it stays directly callable and
        directly testable without going near a registry.

        Args:
            description: What the model is told. Short: a long one gets copied
                back at the user by a small model that is out of its depth.
            safety: How much damage calling it could do. See :data:`Safety`.
            name: Override the tool name. Defaults to the function's own.
        """

        def declare(function: Callable[..., str]) -> Callable[..., str]:
            tool_name = name or function.__name__
            if tool_name in self._tools:
                raise ToolError(f"two tools are called '{tool_name}'")

            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                run=_adapter(function, tool_name),
                parameters=schema_of(function),
                safety=safety,
            )
            return function

        return declare

    def tools(self) -> list[Tool]:
        """Every declared tool, in declaration order."""
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        """One tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"no tool called '{name}'") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


#: The one every tool module declares into.
REGISTRY = Registry()

register = REGISTRY.register
"""Declare a tool. See :meth:`Registry.register`."""


def registered() -> list[Tool]:
    """Every tool rolex knows about."""
    return REGISTRY.tools()


def get(name: str) -> Tool:
    """One of rolex's tools, by name."""
    return REGISTRY.get(name)


def schema_of(function: Callable[..., Any]) -> dict[str, Any]:
    """The JSON Schema for a function's arguments.

    A parameter with no default is required. ``str | None`` becomes a plain
    optional ``string``: ``None`` is dropped rather than emitted as a nullable
    type, because telling a small model it may pass null invites it to. Omit
    the argument instead and the default applies.

    Anything whose type is not a plain string, number or boolean is refused
    here, where the author sees it, rather than at run time.
    """
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except Exception as exc:
        raise ToolError(f"{function.__name__}: could not read its type hints: {exc}") from exc

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise ToolError(f"{function.__name__}: *args and **kwargs cannot be described")

        if parameter.name not in hints:
            raise ToolError(f"{function.__name__}: parameter '{parameter.name}' has no type hint")

        properties[parameter.name] = _property(function.__name__, parameter.name, hints)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _property(tool: str, field: str, hints: Mapping[str, Any]) -> dict[str, Any]:
    """One parameter's schema entry."""
    annotation = hints[field]
    options = _unwrap_optional(annotation)

    kinds = []
    for option in options:
        json_type = _JSON_TYPES.get(option)
        if json_type is None:
            raise ToolError(
                f"{tool}: parameter '{field}' is {_name_of(option)}, which cannot be described "
                f"to a model. Use one of: {', '.join(sorted(t.__name__ for t in _JSON_TYPES))}."
            )
        kinds.append(json_type)

    return {"type": kinds[0] if len(kinds) == 1 else kinds}


def _unwrap_optional(annotation: Any) -> list[Any]:
    """The members of a union, or just the type itself."""
    if get_origin(annotation) in {Union, type(int | str)}:
        return [arg for arg in get_args(annotation) if arg is not type(None)] or [type(None)]
    return [annotation]


def _name_of(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _adapter(function: Callable[..., str], tool_name: str) -> Callable[[Mapping[str, Any]], str]:
    """Turn a function into something that takes one arguments mapping.

    Unexpected and missing arguments are reported as :class:`ToolError`, which
    the agent loop hands back to the model as a result it can learn from — a
    small model gets argument names wrong often, and that is recoverable.
    """
    signature = inspect.signature(function)
    accepted = set(signature.parameters)

    def call(arguments: Mapping[str, Any]) -> str:
        if unknown := sorted(set(arguments) - accepted):
            expected = ", ".join(sorted(accepted)) or "none"
            raise ToolError(
                f"'{tool_name}' got unexpected argument(s) {', '.join(unknown)}. "
                f"It takes: {expected}."
            )
        try:
            bound = signature.bind(**arguments)
        except TypeError as exc:
            raise ToolError(f"'{tool_name}' was called wrongly: {exc}") from exc
        return function(*bound.args, **bound.kwargs)

    return call


def with_safety(tools: Iterable[Tool], allowed: Iterable[Safety]) -> list[Tool]:
    """Only the tools whose safety class is in `allowed`.

    For narrowing what a particular run may do — a scheduled job given
    read-only tools and nothing else. Not used by ``rolex chat``, which offers
    everything and lets the policy decide at call time, so the model can say
    "I could do that but was refused" rather than claiming not to know how.
    """
    permitted = set(allowed)
    return [tool for tool in tools if tool.safety in permitted]
