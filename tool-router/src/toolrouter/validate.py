"""Parse and strictly validate model-produced arguments.

Everything here is about *untrusted input*: size and nesting are bounded before parsing,
duplicate keys / NaN / non-objects are rejected, and the arguments model is validated in
Pydantic's **JSON mode**. Strict JSON mode accepts what JSON legitimately expresses (a string
for an enum, datetime, UUID or path) while rejecting guesses ("42" for an int, 1 for a bool).
`coerce` switches to lax mode (still no unknown fields, no missing fields).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from toolrouter.calls import ToolCallError, ValidationIssue, make_error
from toolrouter.errors import ErrorCode
from toolrouter.tool import SideEffect, Tool


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 64 * 1024  # serialized arguments
    max_depth: int = 32  # JSON nesting
    max_issues: int = 10  # validation issues reported back
    got_chars: int = 60  # offending-value preview length

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_depth", "max_issues", "got_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Limits.{name} must be a positive int")


DEFAULT_LIMITS = Limits()


def resolve_coerce(tool: Tool, router_coerce: bool) -> bool:
    """Should this tool's arguments be validated leniently?

    An explicit per-tool setting always wins. Otherwise the router default applies, except that
    WRITE tools never inherit a lenient default: they must opt in themselves.
    """
    if tool.coerce is not None:
        return tool.coerce
    if tool.side_effect is SideEffect.WRITE:
        return False
    return router_coerce


# ---- parsing ---------------------------------------------------------------------------------


class _Rejected(ValueError):
    """Raised inside JSON hooks to reject non-standard input."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _Rejected(f"duplicate key {key[:40]!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise _Rejected(f"{name} is not valid JSON")


def _nesting_exceeds(text: str, max_depth: int) -> bool:
    """Cheap pre-parse scan so hostile nesting never reaches a recursive parser."""
    depth = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > max_depth:
                return True
        elif ch in "]}":
            depth -= 1
    return False


def _too_large(what: str, limits: Limits) -> ToolCallError:
    return ToolCallError(
        make_error(
            ErrorCode.ARGUMENTS_TOO_LARGE,
            f"Arguments rejected: {what} (limits: {limits.max_bytes} bytes, "
            f"nesting depth {limits.max_depth}). Send smaller or flatter arguments.",
        )
    )


def _invalid_json(message: str) -> ToolCallError:
    return ToolCallError(make_error(ErrorCode.INVALID_JSON, message))


_JSON_KINDS: dict[type, str] = {
    list: "an array",
    str: "a string",
    bool: "a boolean",
    int: "a number",
    float: "a number",
    type(None): "null",
}


def arguments_text(raw: object, limits: Limits = DEFAULT_LIMITS) -> str:
    """Return checked JSON text for the arguments, or raise ToolCallError.

    Accepts JSON text (empty means "no arguments") or an already-parsed dict (which is
    serialized, so both inputs go through the exact same checks).
    """
    if isinstance(raw, dict):
        try:
            text = json.dumps(raw, allow_nan=False)
        except RecursionError:
            raise _too_large("arguments are nested too deeply", limits) from None
        except (TypeError, ValueError) as exc:
            raise _invalid_json(f"Arguments are not JSON-compatible: {exc}") from None
    elif isinstance(raw, str):
        text = raw if raw.strip() else "{}"
    else:
        raise _invalid_json(
            f"Arguments must be a JSON object (text or dict), got {type(raw).__name__}"
        )

    if len(text) > limits.max_bytes:
        raise _too_large("payload is too large", limits)
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise _invalid_json("Arguments contain invalid unicode (lone surrogate)") from None
    if size > limits.max_bytes:
        raise _too_large("payload is too large", limits)
    if _nesting_exceeds(text, limits.max_depth):
        raise _too_large("arguments are nested too deeply", limits)

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise _invalid_json(
            f"Arguments are not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from None
    except _Rejected as exc:
        raise _invalid_json(f"Arguments are not valid JSON: {exc}") from None
    except RecursionError:
        raise _too_large("arguments are nested too deeply", limits) from None
    if not isinstance(parsed, dict):
        kind = _JSON_KINDS.get(type(parsed), type(parsed).__name__)
        raise _invalid_json(f"Arguments must be a JSON object, got {kind}")
    return text


# ---- validation ------------------------------------------------------------------------------


def _format_loc(loc: tuple[str | int, ...]) -> str:
    if not loc:
        return "<arguments>"
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _preview(value: Any, limit: int) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _issues(exc: ValidationError, limits: Limits) -> tuple[list[ValidationIssue], int]:
    errors = exc.errors(include_url=False, include_context=False, include_input=True)
    issues = [
        ValidationIssue(
            loc=list(e["loc"]),
            msg=e["msg"],
            type=e["type"],
            got=None if e["type"] == "missing" else _preview(e.get("input"), limits.got_chars),
        )
        for e in errors[: limits.max_issues]
    ]
    return issues, max(0, len(errors) - len(issues))


def validate_call(
    tool: Tool,
    arguments: object,
    *,
    router_coerce: bool = False,
    limits: Limits = DEFAULT_LIMITS,
) -> BaseModel:
    """Validated arguments for `tool`, or ToolCallError. Never calls the handler."""
    text = arguments_text(arguments, limits)
    lenient = resolve_coerce(tool, router_coerce)
    try:
        return tool.args_model.model_validate_json(text, strict=not lenient)
    except ValidationError as exc:
        issues, more = _issues(exc, limits)
        detail = "; ".join(
            f"{_format_loc(tuple(i.loc))}: {i.msg}" + (f" (got {i.got})" if i.got else "")
            for i in issues
        )
        extra = f" (+{more} more)" if more else ""
        raise ToolCallError(
            make_error(
                ErrorCode.ARGUMENTS_INVALID,
                f"Invalid arguments for {tool.name!r}: {detail}{extra}",
                issues,
            )
        ) from None
    except Exception as exc:  # a crashing custom validator is our bug, not the model's
        raise ToolCallError(
            make_error(
                ErrorCode.INTERNAL_ERROR, f"Arguments for {tool.name!r} could not be validated"
            )
        ) from exc
