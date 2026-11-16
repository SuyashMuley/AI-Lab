"""Redaction of secret argument values, driven by the tool's JSON schema.

A field is secret when its schema is marked `writeOnly` (the `Secret()` marker) or has
`format: password` (`SecretStr`/`SecretBytes`). Fail-closed rules: if we cannot tell where a
secret might be, we hide more rather than less.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

REDACTED = "[REDACTED]"

_UNIONS = ("anyOf", "oneOf", "allOf")


def is_secret(node: Any) -> bool:
    return isinstance(node, dict) and (
        node.get("writeOnly") is True or node.get("format") == "password"
    )


def contains_secret(node: Any) -> bool:
    """Is any part of this schema secret? Used when a location cannot be resolved precisely."""
    if not isinstance(node, dict):
        return False
    if is_secret(node):
        return True
    for key in ("properties", "patternProperties"):
        children = node.get(key)
        if isinstance(children, dict) and any(contains_secret(c) for c in children.values()):
            return True
    for key in ("items", "additionalProperties"):
        if contains_secret(node.get(key)):
            return True
    for key in (*_UNIONS, "prefixItems"):
        subs = node.get(key)
        if isinstance(subs, list) and any(contains_secret(s) for s in subs):
            return True
    return False


def secret_at(node: Any, loc: Sequence[str | int]) -> bool:
    """Does the value at a Pydantic error location hold (part of) a secret?"""
    if not isinstance(node, dict):
        return False
    if is_secret(node):
        return True
    if not loc:
        return False
    for key in _UNIONS:
        subs = node.get(key)
        if isinstance(subs, list) and any(secret_at(s, loc) for s in subs):
            return True
    head, rest = loc[0], loc[1:]
    if isinstance(head, str):
        props = node.get("properties")
        if isinstance(props, dict) and head in props:
            return secret_at(props[head], rest)
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return secret_at(extra, rest)
    else:
        prefix = node.get("prefixItems")
        if isinstance(prefix, list) and head < len(prefix):
            return secret_at(prefix[head], rest)
        if isinstance(node.get("items"), dict):
            return secret_at(node["items"], rest)
    # Unresolvable step (e.g. a union-branch tag in the location): hide if anything below could
    # be secret.
    return contains_secret(node)


def redact_value(value: Any, schema: Any, *, unknown_fields: bool = True) -> Any:
    """A copy of `value` with secret fields replaced by `[REDACTED]`.

    Unknown fields on a *closed* object are redacted too (they were never declared, so we cannot
    know they are safe). Inside union branches unknown fields are left alone, since a field may
    simply belong to another branch.
    """
    if not isinstance(schema, dict):
        return value
    if is_secret(schema):
        return REDACTED
    for key in _UNIONS:
        subs = schema.get(key)
        if isinstance(subs, list):
            for sub in subs:
                value = redact_value(value, sub, unknown_fields=False)
    if isinstance(value, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        extra = schema.get("additionalProperties")
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in props:
                out[k] = redact_value(v, props[k], unknown_fields=unknown_fields)
            elif isinstance(extra, dict):
                out[k] = redact_value(v, extra, unknown_fields=unknown_fields)
            elif props and extra is False and unknown_fields:
                out[k] = REDACTED
            else:
                out[k] = v
        return out
    if isinstance(value, list):
        prefix = schema.get("prefixItems")
        items = schema.get("items")
        return [
            redact_value(
                v,
                prefix[i] if isinstance(prefix, list) and i < len(prefix) else items,
                unknown_fields=unknown_fields,
            )
            for i, v in enumerate(value)
        ]
    return value


def bound(
    value: Any,
    *,
    max_str: int = 200,
    max_items: int = 20,
    max_depth: int = 6,
    _depth: int = 0,
) -> Any:
    """A size-bounded, JSON-compatible copy: long strings, long collections, deep nesting cut."""
    if isinstance(value, str):
        extra = len(value) - max_str
        return value if extra <= 0 else f"{value[:max_str]}…[+{extra} chars]"
    if isinstance(value, dict | list):
        if _depth >= max_depth:
            return "{…}" if isinstance(value, dict) else "[…]"
        if isinstance(value, dict):
            items = list(value.items())
            out: dict[str, Any] = {
                str(k)[:max_str]: bound(
                    v, max_str=max_str, max_items=max_items, max_depth=max_depth, _depth=_depth + 1
                )
                for k, v in items[:max_items]
            }
            if len(items) > max_items:
                out["…"] = f"+{len(items) - max_items} more keys"
            return out
        listed = [
            bound(v, max_str=max_str, max_items=max_items, max_depth=max_depth, _depth=_depth + 1)
            for v in value[:max_items]
        ]
        if len(value) > max_items:
            listed.append(f"…+{len(value) - max_items} more items")
        return listed
    return value
