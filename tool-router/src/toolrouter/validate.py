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
from toolrouter.feedback import signature
from toolrouter.redact import REDACTED, secret_at
from toolrouter.schema import redaction_schema
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
            f"Arguments are not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}. "
            "Send a single valid JSON object."
        ) from None
    except _Rejected as exc:
        raise _invalid_json(
            f"Arguments are not valid JSON: {exc}. Send a single valid JSON object."
        ) from None
    except RecursionError:
        raise _too_large("arguments are nested too deeply", limits) from None
    if not isinstance(parsed, dict):
        kind = _JSON_KINDS.get(type(parsed), type(parsed).__name__)
        raise _invalid_json(f"Arguments must be a JSON object, got {kind}")
    return text


# ---- validation ------------------------------------------------------------------------------


_MAX_LOC_PART = 60  # field names come from the model too: never echo them unbounded


def _clip(part: str | int) -> str | int:
    if isinstance(part, str) and len(part) > _MAX_LOC_PART:
        return part[: _MAX_LOC_PART - 1] + "…"
    return part


def _format_loc(loc: tuple[str | int, ...]) -> str:
    if not loc:
        return "<arguments>"
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _preview(value: Any, limit: int) -> str:
    """A short description of a rejected value. Containers are described, never echoed: they may
    hold secrets under keys we know nothing about."""
    if isinstance(value, dict):
        return f"<object, {len(value)} keys>"
    if isinstance(value, list):
        return f"<array, {len(value)} items>"
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _issues(
    exc: ValidationError, limits: Limits, schema: dict[str, Any]
) -> tuple[list[ValidationIssue], int]:
    errors = exc.errors(include_url=False, include_context=False, include_input=True)
    issues = []
    for e in errors[: limits.max_issues]:
        if e["type"] in ("missing", "extra_forbidden"):
            got = None  # nothing to show / an undeclared field's value (we cannot know it is safe)
        elif secret_at(schema, e["loc"]):
            got = REDACTED  # never echo a secret back, not even in an error
        else:
            got = _preview(e.get("input"), limits.got_chars)
        loc = [_clip(part) for part in e["loc"]]
        issues.append(ValidationIssue(loc=loc, msg=e["msg"], type=e["type"], got=got))
    return issues, max(0, len(errors) - len(issues))


def validate_call(
    tool: Tool,
    arguments: object,
    *,
    router_coerce: bool = False,
    limits: Limits = DEFAULT_LIMITS,
) -> BaseModel:
    """Validated arguments for `tool`, or ToolCallError. Never calls the handler."""
    try:
        text = arguments_text(arguments, limits)
    except ToolCallError as exc:
        if exc.info.code is ErrorCode.INVALID_JSON:  # show what a valid call looks like
            base = exc.info.message
            base += "" if base.endswith((".", "!", "?")) else "."
            hint = f"{base} Expected: {signature(tool)}"
            raise ToolCallError(make_error(ErrorCode.INVALID_JSON, hint)) from None
        raise
    lenient = resolve_coerce(tool, router_coerce)
    try:
        return tool.args_model.model_validate_json(text, strict=not lenient)
    except ValidationError as exc:
        issues, more = _issues(exc, limits, redaction_schema(tool))
        detail = "; ".join(
            f"{_format_loc(tuple(i.loc))}: {i.msg}" + (f" (got {i.got})" if i.got else "")
            for i in issues
        )
        extra = f" (+{more} more)" if more else ""
        raise ToolCallError(
            make_error(
                ErrorCode.ARGUMENTS_INVALID,
                f"Invalid arguments for {tool.name!r}: {detail}{extra}. "
                f"Expected: {signature(tool)}",
                issues,
            )
        ) from None
    except Exception as exc:  # a crashing custom validator is our bug, not the model's
        raise ToolCallError(
            make_error(
                ErrorCode.INTERNAL_ERROR, f"Arguments for {tool.name!r} could not be validated"
            )
        ) from exc
