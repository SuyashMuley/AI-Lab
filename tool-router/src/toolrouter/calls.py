"""Tool calls going in, structured errors coming out."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from toolrouter.errors import ErrorCode

# Should the model try again (possibly with changed arguments)?
RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.UNKNOWN_TOOL: True,
    ErrorCode.INVALID_JSON: True,
    ErrorCode.ARGUMENTS_TOO_LARGE: True,
    ErrorCode.ARGUMENTS_INVALID: True,
    ErrorCode.POLICY_DENIED: False,
    ErrorCode.TOOL_TIMEOUT: True,
    ErrorCode.TOOL_FAILED: True,
    ErrorCode.RESULT_INVALID: False,
    ErrorCode.CANCELLED: False,
    ErrorCode.INTERNAL_ERROR: False,
}


class ToolCall(BaseModel):
    """One tool call as emitted by a model. `arguments` is raw: JSON text or a parsed dict."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: str | dict[str, Any] = "{}"


class ValidationIssue(BaseModel):
    """One problem with the arguments, in a form a model can act on."""

    model_config = ConfigDict(frozen=True)

    loc: list[str | int]  # path to the offending field, e.g. ["address", "zip"]
    msg: str
    type: str  # Pydantic error type, e.g. "missing", "int_type", "extra_forbidden"
    got: str | None = None  # short repr of the offending value (never for missing fields)


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str  # safe for the model to read; no internals
    issues: list[ValidationIssue] = []
    retryable: bool


def make_error(code: ErrorCode, message: str, issues: Iterable[ValidationIssue] = ()) -> ErrorInfo:
    return ErrorInfo(code=code, message=message, issues=list(issues), retryable=RETRYABLE[code])


class ToolResult(BaseModel):
    """The outcome of one tool call. Exactly one of `content` / `error` is set."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    call_id: str
    name: str
    ok: bool
    content: str | None = None  # serialized success value
    error: ErrorInfo | None = None
    truncated: bool = False
    duration_ms: float = 0.0
    # The underlying exception for logs/hooks. Never serialized, never shown to a model.
    cause: BaseException | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _consistent(self) -> ToolResult:
        if self.ok and (self.error is not None or self.content is None):
            raise ValueError("an ok result needs content and no error")
        if not self.ok and (self.error is None or self.content is not None):
            raise ValueError("a failed result needs an error and no content")
        return self

    @property
    def text(self) -> str:
        """Plain text for the model: the content, or the error message."""
        return (
            self.content
            if self.ok and self.content is not None
            else (self.error.message if self.error else "")
        )


class ToolCallError(Exception):
    """Internal control flow: a call failed a pipeline stage. The router converts it to a result."""

    def __init__(self, info: ErrorInfo) -> None:
        super().__init__(info.message)
        self.info = info
