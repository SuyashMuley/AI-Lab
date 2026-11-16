"""Everything a model reads back: repair hints, did-you-mean, and the final result text.

These helpers must never raise: they run while reporting an error, so a bug here would turn a
recoverable tool failure into a crash. Output is bounded however hostile the input.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from functools import cache
from typing import Any

from pydantic import BaseModel

from toolrouter.calls import ToolResult
from toolrouter.schema import args_schema
from toolrouter.tool import Tool

EMPTY_RESULT = "(no output)"  # providers reject or mishandle empty tool results

_MAX_NAME = 64  # tool names are echoed/compared only up to this length
_MAX_SIGNATURE = 300
_MAX_DEPTH = 3
_MAX_ENUM = 8
_MAX_LITERAL = 30


# ---- did you mean --------------------------------------------------------------------------


def _norm(name: str) -> str:
    return name[:_MAX_NAME].lower().replace("-", "_")


def did_you_mean(
    name: str, candidates: Sequence[str], *, limit: int = 3, cutoff: float = 0.6
) -> list[str]:
    """Closest candidate names, best first. Deterministic: ties go to the earlier candidate.

    `Get-Weather` matches `get_weather` exactly; typos and partial names match by similarity.
    """
    target = _norm(name)
    if not target:
        return []
    scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates):
        norm = _norm(candidate)
        ratio = 1.0 if norm == target else difflib.SequenceMatcher(None, target, norm).ratio()
        if ratio >= cutoff:
            scored.append((-ratio, index, candidate))
    scored.sort()
    if scored and scored[0][0] == -1.0:  # an exact match (modulo case and -/_) is the only answer
        scored = [item for item in scored if item[0] == -1.0]
    return [candidate for _, _, candidate in scored[:limit]]


# ---- expected-arguments signature -------------------------------------------------------------


def _lit(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=repr)
    except (TypeError, ValueError, RecursionError):
        text = repr(value)
    return text if len(text) <= _MAX_LITERAL else text[: _MAX_LITERAL - 1] + "…"


def _bound(schema: dict[str, Any], low: str, high: str, xlow: str, xhigh: str) -> str | None:
    lo, hi = schema.get(low), schema.get(high)
    xlo, xhi = schema.get(xlow), schema.get(xhigh)
    parts = []
    if isinstance(lo, int | float) and isinstance(hi, int | float):
        return f"{_lit(lo)}-{_lit(hi)}"
    if isinstance(lo, int | float):
        parts.append(f">={_lit(lo)}")
    if isinstance(hi, int | float):
        parts.append(f"<={_lit(hi)}")
    if isinstance(xlo, int | float):
        parts.append(f">{_lit(xlo)}")
    if isinstance(xhi, int | float):
        parts.append(f"<{_lit(xhi)}")
    return ", ".join(parts) or None


def _scalar(schema: dict[str, Any], kind: Any, depth: int) -> str:
    if kind == "string":
        notes = []
        if (b := _bound(schema, "minLength", "maxLength", "", "")) is not None:
            notes.append(f"{b} chars")
        if isinstance(schema.get("format"), str):
            notes.append(schema["format"])
        return "string" + (f" ({', '.join(notes)})" if notes else "")
    if kind in ("integer", "number"):
        b = _bound(schema, "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
        return f"{kind} ({b})" if b else str(kind)
    if kind in ("boolean", "null"):
        return str(kind)
    if kind == "array":
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list) and prefix:  # a fixed-length tuple
            return "[" + ", ".join(_type_text(p, depth + 1) for p in prefix) + "]"
        item = _type_text(schema.get("items"), depth + 1)
        return f"({item})[]" if " | " in item else f"{item}[]"
    if kind == "object":
        props = schema.get("properties")
        if isinstance(props, dict) and props:
            if depth >= _MAX_DEPTH:
                return "{…}"
            return "{ " + _params(schema, depth + 1) + " }"
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return f"{{ [key: string]: {_type_text(extra, depth + 1)} }}"
        return "object"
    return "any"


def _type_text(schema: Any, depth: int) -> str:
    if not isinstance(schema, dict):
        return "any"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        shown = " | ".join(_lit(v) for v in enum[:_MAX_ENUM])
        return shown + (" | …" if len(enum) > _MAX_ENUM else "")
    if "const" in schema:
        return _lit(schema["const"])
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs:
            parts = list(dict.fromkeys(_type_text(s, depth) for s in subs))
            return " | ".join(parts)
    kind = schema.get("type")
    if isinstance(kind, list) and kind:
        return " | ".join(dict.fromkeys(_scalar(schema, k, depth) for k in kind))
    return _scalar(schema, kind, depth)


def _params(schema: dict[str, Any], depth: int) -> str:
    props = schema.get("properties")
    if not isinstance(props, dict):
        return ""
    required = schema.get("required")
    required_names = (
        {r for r in required if isinstance(r, str)} if isinstance(required, list) else set()
    )
    out = []
    for name, sub in props.items():
        text = f"{name}{'' if name in required_names else '?'}: {_type_text(sub, depth)}"
        if name not in required_names and isinstance(sub, dict) and "default" in sub:
            text += f" = {_lit(sub['default'])}"
        out.append(text)
    return ", ".join(out)


@cache
def _model_params(model: type[BaseModel]) -> str:
    return _params(args_schema(model), 0)


def signature(tool: Tool) -> str:
    """Compact expected-arguments line, e.g. `get_weather(city: string, units?: "c" | "f" = "c")`.

    `?` marks optional parameters; numeric and length limits are shown. Never raises.
    """
    try:
        text = f"{tool.name}({_model_params(tool.args_model)})"
    except Exception:
        return f"{tool.name}(…)"
    return text if len(text) <= _MAX_SIGNATURE else text[: _MAX_SIGNATURE - 2] + "…)"


# ---- messages -----------------------------------------------------------------------------------


def unknown_tool_message(name: str, names: Sequence[str], *, max_listed: int = 20) -> str:
    shown = name[:_MAX_NAME] + ("…" if len(name) > _MAX_NAME else "")
    suggestions = did_you_mean(name, names)
    hint = f" Did you mean {' or '.join(repr(s) for s in suggestions)}?" if suggestions else ""
    if names:
        more = f" (+{len(names) - max_listed} more)" if len(names) > max_listed else ""
        available = f" Available tools: {', '.join(names[:max_listed])}{more}."
    else:
        available = " No tools are available."
    return f"Unknown tool {shown!r}.{hint}{available}"


def model_text(result: ToolResult) -> str:
    """The text to hand a model for this result.

    Success: the content (a placeholder when it is empty). Failure: `Error (<code>): <message>`;
    providers without an error flag (OpenAI, Ollama) need the prefix to tell the two apart.
    """
    if result.ok:
        return result.content or EMPTY_RESULT
    assert result.error is not None  # guaranteed by ToolResult's invariants
    return f"Error ({result.error.code.value}): {result.error.message}"
