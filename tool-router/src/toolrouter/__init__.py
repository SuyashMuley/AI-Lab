"""toolrouter: strict Pydantic tool definitions, validation, and routing for LLM tool calls."""

from toolrouter.argtypes import Confined, Secret
from toolrouter.calls import ErrorInfo, ToolCall, ToolCallError, ToolResult, ValidationIssue
from toolrouter.errors import DuplicateToolError, ErrorCode, ToolDefinitionError, ToolError
from toolrouter.hooks import CallEvent, LoggingHook, RecordingHook, ResultEvent, RouterHook
from toolrouter.models import StrictModel
from toolrouter.policy import (
    ALLOW,
    AllOf,
    AllowAll,
    AllowList,
    DenyList,
    MaxSideEffect,
    Policy,
    PolicyDecision,
    PolicyRequest,
    ReadOnlyPolicy,
    deny,
)
from toolrouter.registry import ToolRegistry
from toolrouter.results import serialize_result
from toolrouter.router import ToolRouter
from toolrouter.tool import SideEffect, Tool, tool
from toolrouter.validate import DEFAULT_LIMITS, Limits, resolve_coerce, validate_call

__all__ = [
    "ALLOW",
    "AllOf",
    "AllowAll",
    "AllowList",
    "CallEvent",
    "DenyList",
    "LoggingHook",
    "MaxSideEffect",
    "Policy",
    "PolicyDecision",
    "PolicyRequest",
    "ReadOnlyPolicy",
    "RecordingHook",
    "ResultEvent",
    "RouterHook",
    "Secret",
    "deny",
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
