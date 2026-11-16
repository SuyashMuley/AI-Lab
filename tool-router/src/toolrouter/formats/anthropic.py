"""Anthropic Messages API tool definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from toolrouter.calls import ToolCall
from toolrouter.formats._util import call_id, get, raw_arguments, tool_name
from toolrouter.schema import tool_schema
from toolrouter.tool import Tool


def tool(t: Tool) -> dict[str, Any]:
    return {"name": t.name, "description": t.description, "input_schema": tool_schema(t)}


def tools(source: Iterable[Tool]) -> list[dict[str, Any]]:
    """`tools=` payload for `client.messages.create(...)`. Accepts a registry or any iterable."""
    return [tool(t) for t in source]


def parse_calls(content: Iterable[Any]) -> list[ToolCall]:
    """Tool calls from response `content` blocks (SDK objects or dicts); other blocks ignored."""
    calls: list[ToolCall] = []
    for block in content:
        if get(block, "type") != "tool_use":
            continue
        calls.append(
            ToolCall(
                id=call_id(get(block, "id"), len(calls)),
                name=tool_name(get(block, "name")),
                arguments=raw_arguments(get(block, "input")),
            )
        )
    return calls
