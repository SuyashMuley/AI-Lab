"""toolrouter: strict Pydantic tool definitions, validation, and routing for LLM tool calls."""

from toolrouter.errors import ToolDefinitionError
from toolrouter.models import StrictModel
from toolrouter.tool import SideEffect, Tool, tool

__all__ = ["SideEffect", "StrictModel", "Tool", "ToolDefinitionError", "tool"]
__version__ = "0.1.0"
