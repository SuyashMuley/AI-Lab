"""OpenAI tool definitions (Chat Completions and Responses shapes).

`strict` structured-output mode is intentionally not emitted yet: it requires every property
to be required and optionals to be nullable, which needs a null-handling story in validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from toolrouter.calls import ToolCall, ToolResult
from toolrouter.feedback import model_text
from toolrouter.formats._util import call_id, get, raw_arguments, tool_name
from toolrouter.schema import tool_schema
from toolrouter.tool import Tool


def tool(t: Tool, api: Literal["chat", "responses"] = "chat") -> dict[str, Any]:
    params = tool_schema(t)
    if api == "chat":
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": params},
        }
    return {"type": "function", "name": t.name, "description": t.description, "parameters": params}


def tools(
    source: Iterable[Tool], api: Literal["chat", "responses"] = "chat"
) -> list[dict[str, Any]]:
    """`tools=` payload. `api="chat"` for Chat Completions, `"responses"` for Responses."""
    return [tool(t, api) for t in source]


def parse_calls(source: Any, api: Literal["chat", "responses"] = "chat") -> list[ToolCall]:
    """Tool calls from a response.

    chat: a message (with `tool_calls`) or the list of tool calls itself.
    responses: a response (with `output`) or the list of output items; only `function_call`
    items are used. Arguments stay as the JSON text the API returned.
    """
    calls: list[ToolCall] = []
    if api == "chat":
        items = source if isinstance(source, list | tuple) else get(source, "tool_calls")
        for item in items or []:
            fn = get(item, "function")
            calls.append(
                ToolCall(
                    id=call_id(get(item, "id"), len(calls)),
                    name=tool_name(get(fn, "name")),
                    arguments=raw_arguments(get(fn, "arguments")),
                )
            )
        return calls
    items = source if isinstance(source, list | tuple) else get(source, "output")
    for item in items or []:
        if get(item, "type") != "function_call":
            continue
        calls.append(
            ToolCall(
                id=call_id(get(item, "call_id"), len(calls)),
                name=tool_name(get(item, "name")),
                arguments=raw_arguments(get(item, "arguments")),
            )
        )
    return calls


def result_messages(
    results: Iterable[ToolResult], api: Literal["chat", "responses"] = "chat"
) -> list[dict[str, Any]]:
    """Items to append after the assistant turn, one per result, in order.

    chat: `{"role": "tool", "tool_call_id", "content"}` messages.
    responses: `{"type": "function_call_output", "call_id", "output"}` input items.
    Failures are prefixed `Error (<code>): ` because these APIs have no error flag.
    """
    if api == "chat":
        return [
            {"role": "tool", "tool_call_id": r.call_id, "content": model_text(r)} for r in results
        ]
    return [
        {"type": "function_call_output", "call_id": r.call_id, "output": model_text(r)}
        for r in results
    ]
