"""Errors raised while *defining* tools (programmer errors, surfaced immediately).

Failures caused by a model at call time never raise; they become ToolResult errors (T4).
"""


class ToolDefinitionError(ValueError):
    """A tool is mis-defined: bad name, missing description, unsafe or unusable schema."""


class DuplicateToolError(ToolDefinitionError):
    """A tool with this name is already registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"A tool named {name!r} is already registered")
        self.name = name
