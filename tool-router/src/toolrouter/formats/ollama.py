"""Ollama `/api/chat` tool definitions (same envelope as OpenAI Chat Completions)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from toolrouter.calls import ToolCall
from toolrouter.formats._util import get, raw_arguments, tool_name
from toolrouter.schema import tool_schema
from toolrouter.tool import Tool


def tool(t: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": tool_schema(t),
        },
    }


def tools(source: Iterable[Tool]) -> list[dict[str, Any]]:
    """`tools=` payload for `ollama.chat(...)`. Accepts a registry or any iterable."""
    return [tool(t) for t in source]


def parse_calls(message: Any) -> list[ToolCall]:
    """Tool calls from a chat response message (object or dict), or from a bare list of calls.

    Ollama calls carry no ids, so deterministic ones (`call_0`, `call_1`, ...) are generated.
    """
    raw = message if isinstance(message, list | tuple) else get(message, "tool_calls")
    calls: list[ToolCall] = []
    for item in raw or []:
        fn = get(item, "function")
        calls.append(
            ToolCall(
                id=f"call_{len(calls)}",
                name=tool_name(get(fn, "name")),
                arguments=raw_arguments(get(fn, "arguments")),
            )
        )
    return calls
