from typing import Any

from pydantic import create_model

from toolrouter import SideEffect, StrictModel, Tool, tool


def make_tool(
    annotation: Any,
    *,
    coerce: bool | None = None,
    side_effect: SideEffect = SideEffect.NONE,
    name: str = "t",
) -> Tool:
    """A tool with one required argument `x` of the given type."""
    model = create_model("Args", __base__=StrictModel, x=(annotation, ...))
    return tool(model=model, name=name, description="d", coerce=coerce, side_effect=side_effect)(
        lambda args: None
    )
