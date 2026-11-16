"""Tool definition: the @tool decorator and the immutable Tool it produces."""

from __future__ import annotations

import functools
import inspect
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, overload

from pydantic import BaseModel, TypeAdapter

from toolrouter.docstring import parse_docstring
from toolrouter.errors import ToolDefinitionError
from toolrouter.models import assert_strict_args_model
from toolrouter.signature import build_args_model, resolve_hints

# Valid across Anthropic, OpenAI, and Ollama tool-name rules.
NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


class SideEffect(StrEnum):
    NONE = "none"  # pure
    READ = "read"  # reads external state
    WRITE = "write"  # mutates external state


@dataclass(frozen=True)
class Tool:
    """A validated, ready-to-register tool. Build with @tool; fields are checked on creation."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]
    result_type: Any | None = None  # return annotation; None = unannotated
    side_effect: SideEffect = SideEffect.NONE
    timeout_s: float = 30.0
    max_output_chars: int = 20_000
    coerce: bool | None = None  # None = inherit the router default
    takes_model: bool = False  # True: handler(args_model_instance); False: handler(**fields)
    # Validates + serializes return values; None when the result is unannotated or `Any`.
    result_adapter: TypeAdapter[Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            hint = " (lambdas need an explicit name=)" if self.name == "<lambda>" else ""
            raise ToolDefinitionError(
                f"Invalid tool name {self.name!r}: use 1-64 chars of letters, digits, '_' or '-', "
                f"starting with a letter{hint}"
            )
        if not self.description.strip():
            raise ToolDefinitionError(
                f"Tool {self.name!r} needs a description: add a docstring or pass description="
            )
        if isinstance(self.timeout_s, bool) or not (
            isinstance(self.timeout_s, int | float) and math.isfinite(self.timeout_s)
        ):
            raise ToolDefinitionError(f"Tool {self.name!r}: timeout_s must be a finite number")
        if self.timeout_s <= 0:
            raise ToolDefinitionError(f"Tool {self.name!r}: timeout_s must be > 0")
        if (
            isinstance(self.max_output_chars, bool)
            or not isinstance(self.max_output_chars, int)
            or self.max_output_chars <= 0
        ):
            raise ToolDefinitionError(
                f"Tool {self.name!r}: max_output_chars must be a positive int"
            )
        object.__setattr__(
            self, "result_adapter", _build_result_adapter(self.name, self.result_type)
        )

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.handler)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the original function directly (no validation); handy in tests."""
        return self.handler(*args, **kwargs)

    def bind(self, args: BaseModel) -> Callable[[], Any]:
        """A zero-argument callable that runs the handler with already-validated arguments."""
        if not isinstance(args, self.args_model):
            raise TypeError(
                f"{self.name}: expected {self.args_model.__name__}, got {type(args).__name__}"
            )
        if self.takes_model:
            return functools.partial(self.handler, args)
        kwargs = {n: getattr(args, n) for n in self.args_model.model_fields}
        return functools.partial(self.handler, **kwargs)

    async def invoke(self, args: BaseModel) -> Any:
        """Run the handler inline, no timeout or thread offload (the ToolExecutor adds those)."""
        result = self.bind(args)()
        if inspect.isawaitable(result):
            result = await result
        return result


def _build_result_adapter(name: str, result_type: Any | None) -> TypeAdapter[Any] | None:
    if result_type is None or result_type is Any:
        return None
    try:
        return TypeAdapter(result_type)
    except Exception as exc:
        raise ToolDefinitionError(
            f"Tool {name!r}: return type {result_type!r} cannot be validated by Pydantic. "
            "Annotate with a Pydantic-compatible type, or leave the return unannotated."
        ) from exc


def _explicit_model_param(fn: Callable[..., Any], model: type[BaseModel], name: str) -> None:
    """The handler of an explicit-model tool must take exactly one parameter: the model."""
    params = list(inspect.signature(fn).parameters.values())
    ok_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if len(params) != 1 or params[0].kind not in ok_kinds:
        raise ToolDefinitionError(
            f"Tool {name!r}: with model=, the handler must take exactly one parameter "
            f"(the {model.__name__} instance)"
        )
    ann = resolve_hints(fn, name).get(params[0].name)
    if ann is not None and not (isinstance(ann, type) and (ann is model or issubclass(model, ann))):
        raise ToolDefinitionError(
            f"Tool {name!r}: handler parameter is annotated {ann!r}, expected {model.__name__}"
        )


def _build_tool(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    model: type[BaseModel] | None,
    side_effect: SideEffect,
    timeout_s: float,
    max_output_chars: int,
    coerce: bool | None,
) -> Tool:
    if not callable(fn):
        raise ToolDefinitionError(f"@tool needs a function, got {type(fn).__name__}")
    tool_name = name if name is not None else getattr(fn, "__name__", type(fn).__name__)
    hints = resolve_hints(fn, tool_name)

    if model is not None:
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ToolDefinitionError(f"Tool {tool_name!r}: model= must be a Pydantic model class")
        _explicit_model_param(fn, model, tool_name)
        args_model: type[BaseModel] = model
    else:
        args_model = build_args_model(fn, tool_name, hints)
    assert_strict_args_model(args_model, tool_name)

    desc = description if description is not None else parse_docstring(fn.__doc__).summary
    return Tool(
        name=tool_name,
        description=desc,
        args_model=args_model,
        handler=fn,
        result_type=hints.get("return"),
        side_effect=SideEffect(side_effect),
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        coerce=coerce,
        takes_model=model is not None,
    )


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    model: type[BaseModel] | None = ...,
    side_effect: SideEffect = ...,
    timeout_s: float = ...,
    max_output_chars: int = ...,
    coerce: bool | None = ...,
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    model: type[BaseModel] | None = None,
    side_effect: SideEffect = SideEffect.NONE,
    timeout_s: float = 30.0,
    max_output_chars: int = 20_000,
    coerce: bool | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Define a tool. Use bare (`@tool`) or with options (`@tool(side_effect=...)`).

    The argument schema comes from the function signature (every parameter must be annotated)
    or, with `model=`, from an explicit Pydantic model that the handler receives whole.
    The description comes from `description=` or the docstring summary.
    """

    def build(f: Callable[..., Any]) -> Tool:
        return _build_tool(
            f,
            name=name,
            description=description,
            model=model,
            side_effect=side_effect,
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
            coerce=coerce,
        )

    return build if fn is None else build(fn)
