"""Normalize Pydantic's JSON schema into a clean, self-contained tool input schema.

Pydantic emits `$defs`/`$ref`, `title` noise, `allOf` wrappers and OpenAPI-only keywords.
Providers handle those unevenly, so we emit one flat, deterministic shape:

- all `$ref`s inlined, no `$defs` (recursive models are rejected: they cannot be inlined)
- `title` dropped everywhere; the root `description` dropped (the tool has its own)
- single-element `allOf` wrappers flattened; OpenAPI `discriminator` dropped
- root is always `{"type": "object", "properties": {...}}`
- `additionalProperties: false` is preserved on every closed object
- secrets: fields marked secret (`SecretStr`, `Annotated[..., Secret()]`) carry `writeOnly: true`
  internally (`keep_markers=True`, used for redaction). Provider-facing schemas drop that marker
  and, more importantly, any `default`/`examples` on a secret: a secret must never be sent to a
  model as a schema default.
"""

from __future__ import annotations

import copy
from functools import cache
from typing import Any

from pydantic import BaseModel

from toolrouter.errors import ToolDefinitionError
from toolrouter.tool import Tool

# Keywords whose value is a mapping of *names* to schemas (keys are field names, not keywords).
_NAME_TO_SCHEMA = ("properties", "patternProperties")
_SCHEMA_VALUED = ("items", "additionalProperties", "not", "if", "then", "else", "contains")
_SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
_DROPPED = frozenset({"title", "$defs", "definitions", "discriminator"})


def _is_secret(node: dict[str, Any]) -> bool:
    return node.get("writeOnly") is True or node.get("format") == "password"


def _finish(node: dict[str, Any], keep: bool) -> dict[str, Any]:
    if keep or not _is_secret(node):
        return node
    hidden = {k: v for k, v in node.items() if k not in ("writeOnly", "default", "examples")}
    if hidden.get("format") == "password":
        del hidden["format"]
    return hidden


def _clean(schema: Any, defs: dict[str, Any], stack: tuple[str, ...], tool: str, keep: bool) -> Any:
    if not isinstance(schema, dict):
        return schema  # booleans (additionalProperties: false) and other scalars

    if "$ref" in schema:
        ref = schema["$ref"]
        name = ref.rsplit("/", 1)[-1] if isinstance(ref, str) else ""
        if not (isinstance(ref, str) and ref.startswith("#/$defs/") and name in defs):
            raise ToolDefinitionError(f"Tool {tool!r}: unsupported schema reference {ref!r}")
        if name in stack:
            raise ToolDefinitionError(
                f"Tool {tool!r}: recursive model {name!r} cannot be used in tool arguments"
            )
        resolved = _clean(defs[name], defs, (*stack, name), tool, keep)
        siblings = {k: v for k, v in schema.items() if k != "$ref"}
        return _finish({**resolved, **_clean(siblings, defs, stack, tool, keep)}, keep)

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _DROPPED:
            continue
        if key in _NAME_TO_SCHEMA and isinstance(value, dict):
            out[key] = {n: _clean(s, defs, stack, tool, keep) for n, s in value.items()}
        elif key in _SCHEMA_VALUED:
            out[key] = _clean(value, defs, stack, tool, keep)
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            out[key] = [_clean(s, defs, stack, tool, keep) for s in value]
        else:
            out[key] = value  # enum/default/const/required/...: data, left untouched

    all_of = out.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        del out["allOf"]
        out = {**all_of[0], **out}  # the wrapper's own keywords win
    return _finish(out, keep)


def normalize_schema(
    raw: dict[str, Any], tool_name: str = "<tool>", *, keep_markers: bool = False
) -> dict[str, Any]:
    """Normalize a raw Pydantic JSON schema (see module docstring)."""
    defs = raw.get("$defs", {})
    root = _clean(raw, defs if isinstance(defs, dict) else {}, (), tool_name, keep_markers)
    root.pop("description", None)
    root["type"] = "object"
    root.setdefault("properties", {})
    return {"type": "object", **{k: v for k, v in root.items() if k != "type"}}


@cache
def _model_schema(model: type[BaseModel], tool_name: str, keep_markers: bool) -> dict[str, Any]:
    return normalize_schema(model.model_json_schema(), tool_name, keep_markers=keep_markers)


def args_schema(
    model: type[BaseModel], tool_name: str = "<tool>", *, keep_markers: bool = False
) -> dict[str, Any]:
    """Normalized input schema for an arguments model (a fresh copy; safe to mutate).

    Provider-facing by default. `keep_markers=True` keeps secret markers for redaction.
    """
    return copy.deepcopy(_model_schema(model, tool_name, keep_markers))


def tool_schema(tool: Tool) -> dict[str, Any]:
    """Normalized, provider-facing input schema for a tool (no secret markers or defaults)."""
    return args_schema(tool.args_model, tool.name)


def redaction_schema(tool: Tool) -> dict[str, Any]:
    """The same schema with secret markers kept; internal, never sent to a provider."""
    return args_schema(tool.args_model, tool.name, keep_markers=True)
