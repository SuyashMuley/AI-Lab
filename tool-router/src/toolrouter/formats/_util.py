"""Helpers for reading provider responses (SDK objects or plain dicts) without crashing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def get(obj: Any, key: str, default: Any = None) -> Any:
    """`obj[key]` for mappings, `obj.key` for SDK objects, `default` otherwise."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def call_id(raw: Any, index: int) -> str:
    return raw if isinstance(raw, str) and raw else f"call_{index}"


def tool_name(raw: Any) -> str:
    return raw if isinstance(raw, str) else ""


def raw_arguments(value: Any) -> str | dict[str, Any]:
    """Keep text and dicts as they are; anything else is serialized so validation rejects it."""
    if value is None:
        return {}
    if isinstance(value, str | dict):
        return value
    return json.dumps(value, default=repr)
