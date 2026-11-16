"""Anthropic Messages API tool definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from toolrouter.calls import ToolCall, ToolResult
from toolrouter.feedback import model_text
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


def result_block(result: ToolResult) -> dict[str, Any]:
    """One `tool_result` content block. Failures set `is_error`."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": model_text(result),
    }
    if not result.ok:
        block["is_error"] = True
    return block


def result_messages(results: Iterable[ToolResult]) -> list[dict[str, Any]]:
    """Messages to append after the assistant turn.

    Anthropic wants every result for a turn's `tool_use` blocks together in ONE user message,
    in the same order; `[]` when there are no results. To add your own text, append text
    blocks after these (tool results must come first).
    """
    blocks = [result_block(r) for r in results]
    return [{"role": "user", "content": blocks}] if blocks else []
