"""ToolRegistry: the set of tools a router can dispatch to."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from toolrouter.errors import DuplicateToolError, ToolDefinitionError
from toolrouter.schema import tool_schema
from toolrouter.tool import Tool


class ToolRegistry:
    """Named tools in registration order.

    Registration fails fast: a duplicate name, a non-Tool, or an args model that cannot be
    exported as a schema (e.g. recursive) is rejected here, not on first use. Build the
    registry before serving calls; mutating it while calls are in flight is not supported.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise ToolDefinitionError(
                f"register() expects a Tool, got {type(tool).__name__}. "
                "Did you forget the @tool decorator?"
            )
        if tool.name in self._tools:
            raise DuplicateToolError(tool.name)
        tool_schema(tool)  # fail fast on schemas that can't be exported
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> Tool:
        try:
            return self._tools.pop(name)
        except KeyError:
            raise KeyError(f"No tool named {name!r}") from None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"No tool named {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(list(self._tools.values()))

    def names(self) -> list[str]:
        return list(self._tools)

    def schema(self, name: str) -> dict[str, Any]:
        """Normalized input schema for one tool (a fresh copy)."""
        return tool_schema(self[name])
