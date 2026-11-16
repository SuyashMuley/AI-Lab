"""Turn a handler's return value into the text a model reads.

Rules (the value is first validated against the tool's return annotation, strictly, so a
handler bug surfaces as RESULT_INVALID instead of garbage reaching the model):

- an exact `str` is passed through as-is
- everything else becomes JSON text (Pydantic models, lists, dicts, numbers, datetimes, enums,
  `None` -> `null`, ...)
- an unannotated (or `Any`) return is not validated, only serialized; if it cannot be
  serialized the result is RESULT_INVALID
"""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError, to_json

from toolrouter.calls import ToolCallError, make_error
from toolrouter.errors import ErrorCode
from toolrouter.tool import Tool

_MARKER_RESERVE = 48  # room for "\n[truncated N chars]"


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cap `text` at `limit` characters, *including* a visible truncation marker.

    Returns (text, was_truncated). The result never exceeds `limit`; if `limit` is too small for
    a marker the text is hard-cut. Truncated JSON is no longer valid JSON; the marker says so.
    """
    if len(text) <= limit:
        return text, False
    keep = limit - _MARKER_RESERVE
    if keep > 0:
        out = text[:keep] + f"\n[truncated {len(text) - keep} chars]"
        if len(out) <= limit:
            return out, True
    return text[:limit], True


def _invalid(tool: Tool, cause: Exception) -> ToolCallError:
    # Deliberately generic: details (values, types) stay on the exception for logs, not the model.
    err = ToolCallError(
        make_error(
            ErrorCode.RESULT_INVALID,
            f"Tool {tool.name!r} returned a result that could not be used "
            "(this is a bug in the tool, not in your call).",
        )
    )
    err.__cause__ = cause
    return err


def serialize_result(tool: Tool, value: object) -> str:
    """Validated, serialized result text, or ToolCallError(RESULT_INVALID)."""
    adapter = tool.result_adapter
    try:
        if adapter is not None:
            value = adapter.validate_python(value, strict=True)
        if type(value) is str:
            return value
        data = adapter.dump_json(value) if adapter is not None else to_json(value)
    except (ValidationError, PydanticSerializationError, ValueError, TypeError) as exc:
        raise _invalid(tool, exc) from exc  # original stays on __cause__: for logs, not the model
    return data.decode("utf-8")
