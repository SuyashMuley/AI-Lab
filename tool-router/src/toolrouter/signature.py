"""Build a strict arguments model from a function signature and docstring."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from toolrouter.docstring import parse_docstring
from toolrouter.errors import ToolDefinitionError
from toolrouter.models import StrictModel

_RESERVED = frozenset(dir(BaseModel))


def resolve_hints(fn: Callable[..., Any], tool_name: str) -> dict[str, Any]:
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception as exc:  # NameError, TypeError, ...
        raise ToolDefinitionError(
            f"Tool {tool_name!r}: cannot resolve type hints ({exc}). "
            "Types must be importable at module level."
        ) from exc


def _has_description(annotation: Any) -> bool:
    if get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(m, FieldInfo) and m.description for m in get_args(annotation)[1:])


def build_args_model(
    fn: Callable[..., Any], tool_name: str, hints: dict[str, Any]
) -> type[BaseModel]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ToolDefinitionError(f"Tool {tool_name!r}: cannot read signature: {exc}") from exc

    docs = parse_docstring(getattr(fn, "__doc__", None)).params
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        where = f"Tool {tool_name!r}, parameter {pname!r}"
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolDefinitionError(f"{where}: *args/**kwargs are not allowed (schema is closed)")
        if param.kind is param.POSITIONAL_ONLY:
            raise ToolDefinitionError(f"{where}: positional-only parameters can't be named")
        if pname not in hints:
            raise ToolDefinitionError(f"{where}: missing type annotation")
        if pname.startswith("_") or pname in _RESERVED:
            raise ToolDefinitionError(
                f"{where}: name is reserved by the schema layer; rename it "
                "(or define an explicit model with an alias)"
            )
        annotation = hints[pname]
        description = docs.get(pname)
        if isinstance(param.default, FieldInfo):
            fields[pname] = (annotation, param.default)
            continue
        kwargs: dict[str, Any] = {}
        if description and not _has_description(annotation):
            kwargs["description"] = description
        if param.default is not param.empty:
            kwargs["default"] = param.default
        fields[pname] = (annotation, Field(**kwargs))

    try:
        return create_model(f"{tool_name}_args", __base__=StrictModel, **fields)
    except Exception as exc:
        raise ToolDefinitionError(f"Tool {tool_name!r}: invalid argument types: {exc}") from exc
