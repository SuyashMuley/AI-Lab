"""Errors raised while *defining* tools (programmer errors, surfaced immediately).

Failures caused by a model at call time never raise; they become ToolResult errors (T4).
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable failure codes returned to callers (and, via text, to models)."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_JSON = "invalid_json"
    ARGUMENTS_TOO_LARGE = "arguments_too_large"
    ARGUMENTS_INVALID = "arguments_invalid"
    POLICY_DENIED = "policy_denied"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_FAILED = "tool_failed"
    RESULT_INVALID = "result_invalid"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"  # a bug on our side, e.g. a crashing custom validator


class ToolDefinitionError(ValueError):
    """A tool is mis-defined: bad name, missing description, unsafe or unusable schema."""


class DuplicateToolError(ToolDefinitionError):
    """A tool with this name is already registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"A tool named {name!r} is already registered")
        self.name = name


class ToolError(Exception):
    """Raise from a handler for an *expected* failure (file not found, quota exceeded, ...).

    Its message is written for the model and is passed through to it, so keep it free of
    internals. Any other exception from a handler is reported generically, never verbatim.
    """
