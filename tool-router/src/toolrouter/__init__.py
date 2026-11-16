"""toolrouter: strict Pydantic tool definitions, validation, and routing for LLM tool calls."""

from toolrouter.argtypes import Confined
from toolrouter.calls import ErrorInfo, ToolCall, ToolCallError, ToolResult, ValidationIssue
from toolrouter.errors import DuplicateToolError, ErrorCode, ToolDefinitionError, ToolError
from toolrouter.models import StrictModel
from toolrouter.registry import ToolRegistry
from toolrouter.results import serialize_result
from toolrouter.router import ToolRouter
from toolrouter.tool import SideEffect, Tool, tool
from toolrouter.validate import DEFAULT_LIMITS, Limits, resolve_coerce, validate_call

__all__ = [
    "DEFAULT_LIMITS",
    "Confined",
    "DuplicateToolError",
    "ErrorCode",
    "ErrorInfo",
    "Limits",
    "SideEffect",
    "StrictModel",
    "Tool",
    "ToolCall",
    "ToolCallError",
    "ToolDefinitionError",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolRouter",
    "ValidationIssue",
    "resolve_coerce",
    "serialize_result",
    "tool",
    "validate_call",
]
__version__ = "0.1.0"
