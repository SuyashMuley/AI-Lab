"""The bridge to `toolrouter`: what the model is offered, and what a result means to the log."""

from __future__ import annotations

from llmcli.core.models import ToolSpec
from toolrouter import ToolRegistry, ToolResult
from toolrouter.feedback import model_text
from toolrouter.schema import tool_schema

from localagent.events import ToolOutcome

DEFAULT_SYSTEM = "You are a helpful assistant. Solve the user's task and answer concisely."

TOOL_GUIDANCE = (
    "You can call tools when they help. Call a tool only with arguments that match its "
    "signature. If a tool returns an error, read it, fix your call, and try again. "
    "When you have what you need, answer directly without calling a tool."
)


def effective_system(system: str | None, has_tools: bool) -> str:
    """The system prompt a run uses: the caller's own verbatim, else the default (plus tool
    guidance when tools are on offer)."""
    if system is not None:
        return system
    return f"{DEFAULT_SYSTEM} {TOOL_GUIDANCE}" if has_tools else DEFAULT_SYSTEM


def tool_specs(registry: ToolRegistry) -> list[ToolSpec]:
    """The registry as provider-agnostic tool specs (schemas without secret markers)."""
    return [
        ToolSpec(name=t.name, description=t.description, parameters=tool_schema(t))
        for t in registry
    ]


def outcome_from_result(result: ToolResult) -> ToolOutcome:
    return ToolOutcome(
        call_id=result.call_id,
        name=result.name,
        ok=result.ok,
        text=model_text(result),
        error_code=result.error.code.value if result.error is not None else None,
        duration_ms=round(result.duration_ms, 3),
        truncated=result.truncated,
    )
