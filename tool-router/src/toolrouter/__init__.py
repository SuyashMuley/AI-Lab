"""toolrouter: strict Pydantic tool definitions, validation, and routing for LLM tool calls."""

from toolrouter.errors import DuplicateToolError, ToolDefinitionError
from toolrouter.models import StrictModel
from toolrouter.registry import ToolRegistry
from toolrouter.tool import SideEffect, Tool, tool

__all__ = [
    "DuplicateToolError",
    "SideEffect",
    "StrictModel",
    "Tool",
    "ToolDefinitionError",
    "ToolRegistry",
    "tool",
]
__version__ = "0.1.0"
