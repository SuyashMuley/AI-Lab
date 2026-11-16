"""Strict base model and the schema check that enforces 'no unknown fields' everywhere."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.errors import PydanticInvalidForJsonSchema

from toolrouter.errors import ToolDefinitionError


class StrictModel(BaseModel):
    """Base for tool argument models: unknown fields are rejected, instances are immutable.

    Type strictness (no "42" -> 42) is applied at validation time, so a router or tool can
    opt in to coercion without redefining the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


_SCHEMA_CHILDREN = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_SINGLE = ("items", "additionalProperties", "not", "if", "then", "else", "contains")
_SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")


def _find_open_objects(schema: Any, path: str, problems: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    if "properties" in schema and schema.get("additionalProperties") is not False:
        problems.append(path or "<root>")
    for key in _SCHEMA_CHILDREN:
        children = schema.get(key)
        if isinstance(children, dict):
            for name, sub in children.items():
                _find_open_objects(sub, f"{path}/{name}" if path else name, problems)
    for key in _SCHEMA_SINGLE:
        _find_open_objects(schema.get(key), path, problems)
    for key in _SCHEMA_LISTS:
        for sub in schema.get(key) or []:
            _find_open_objects(sub, path, problems)


def assert_strict_args_model(model: type[BaseModel], tool_name: str) -> None:
    """Reject models that can't be a JSON schema or that accept unknown fields anywhere."""
    try:
        schema = model.model_json_schema()
    except (PydanticInvalidForJsonSchema, TypeError) as exc:
        raise ToolDefinitionError(
            f"Tool {tool_name!r}: arguments cannot be expressed as a JSON schema "
            f"(models can only be shown schemas): {exc}"
        ) from exc
    problems: list[str] = []
    _find_open_objects(schema, "", problems)
    if problems:
        where = ", ".join(sorted(set(problems)))
        raise ToolDefinitionError(
            f"Tool {tool_name!r}: model(s) accept unknown fields: {where}. "
            "Use ConfigDict(extra='forbid') or subclass toolrouter.StrictModel."
        )
