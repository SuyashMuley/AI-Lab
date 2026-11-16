"""OpenAI tool definitions (Chat Completions and Responses shapes).

`strict` structured-output mode is intentionally not emitted yet: it requires every property
to be required and optionals to be nullable, which needs a null-handling story in validation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

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
