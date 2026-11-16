"""Errors raised while *defining* tools (programmer errors, surfaced immediately).

Failures caused by a model at call time never raise; they become ToolResult errors (T4).
"""


class ToolDefinitionError(ValueError):
    """A tool is mis-defined: bad name, missing description, unsafe or unusable schema."""
