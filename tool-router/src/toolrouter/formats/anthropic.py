"""Anthropic Messages API tool definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from toolrouter.schema import tool_schema
from toolrouter.tool import Tool


def tool(t: Tool) -> dict[str, Any]:
    return {"name": t.name, "description": t.description, "input_schema": tool_schema(t)}


def tools(source: Iterable[Tool]) -> list[dict[str, Any]]:
    """`tools=` payload for `client.messages.create(...)`. Accepts a registry or any iterable."""
    return [tool(t) for t in source]
