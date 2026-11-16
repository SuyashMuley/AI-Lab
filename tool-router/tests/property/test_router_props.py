"""Fuzz the whole pipeline: whatever a model sends, dispatch returns a result and never raises."""

import asyncio
import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fakes import router_tools as rt
from toolrouter import ErrorCode, ToolCall, ToolCallError, ToolRegistry, ToolRouter, validate_call

CLIENT_ERRORS = {
    ErrorCode.UNKNOWN_TOOL,
    ErrorCode.INVALID_JSON,
    ErrorCode.ARGUMENTS_TOO_LARGE,
    ErrorCode.ARGUMENTS_INVALID,
}
REGISTRY = ToolRegistry(rt.SAFE)
ROUTER = ToolRouter(REGISTRY)

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4),
    max_leaves=25,
)
names = st.sampled_from(REGISTRY.names()) | st.text(max_size=20)


def arguments_strategy() -> st.SearchStrategy[Any]:
    as_text = json_values.map(lambda v: _dumps(v))
    return (
        as_text
        | st.dictionaries(st.text(max_size=6), json_values, max_size=4)
        | st.text(max_size=80)
    )


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value)
    except (ValueError, RecursionError):
        return "{"


@settings(max_examples=500, deadline=None)
@given(names, arguments_strategy(), st.booleans())
def test_dispatch_never_raises_and_handlers_only_run_after_validation(
    name: str, arguments: Any, coerce: bool
) -> None:
    rt.CALLS.clear()
    call = ToolCall(id="id-1", name=name, arguments=arguments)
    result = asyncio.run(ToolRouter(REGISTRY, coerce=coerce).dispatch(call))

    assert result.call_id == "id-1"
    assert result.name == name
    assert result.duration_ms >= 0

    tool = REGISTRY.get(name)
    if result.ok:
        assert tool is not None
        assert len(rt.CALLS) == 1  # ran exactly once...
        validate_call(tool, arguments, router_coerce=coerce)  # ...and the args are valid
    else:
        assert result.error is not None
        assert result.error.code is not ErrorCode.INTERNAL_ERROR, result.error
        if result.error.code in CLIENT_ERRORS:
            assert rt.CALLS == []  # rejected before any handler ran
        if tool is not None:
            try:
                validate_call(tool, arguments, router_coerce=coerce)
            except ToolCallError:
                assert rt.CALLS == []  # invalid arguments never reach a handler
