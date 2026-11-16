import itertools

import pytest

from tests.fakes.factories import make_tool
from toolrouter import SideEffect, ToolCallError, resolve_coerce, validate_call


@pytest.mark.parametrize(
    ("tool_coerce", "effect", "router", "expected"),
    [
        # explicit per-tool setting always wins
        (True, SideEffect.NONE, False, True),
        (True, SideEffect.READ, False, True),
        (True, SideEffect.WRITE, False, True),  # WRITE may opt in explicitly
        (True, SideEffect.WRITE, True, True),
        (False, SideEffect.NONE, True, False),
        (False, SideEffect.READ, True, False),
        (False, SideEffect.WRITE, True, False),
        # inherit the router default...
        (None, SideEffect.NONE, True, True),
        (None, SideEffect.READ, True, True),
        (None, SideEffect.NONE, False, False),
        (None, SideEffect.READ, False, False),
        # ...except WRITE tools, which never inherit lenient
        (None, SideEffect.WRITE, True, False),
        (None, SideEffect.WRITE, False, False),
    ],
)
def test_resolve_coerce_truth_table(
    tool_coerce: bool | None, effect: SideEffect, router: bool, expected: bool
) -> None:
    t = make_tool(int, coerce=tool_coerce, side_effect=effect)
    assert resolve_coerce(t, router) is expected


def test_truth_table_is_exhaustive() -> None:
    # every combination resolves to a bool without error
    for c, e, r in itertools.product([None, True, False], SideEffect, [True, False]):
        assert isinstance(resolve_coerce(make_tool(int, coerce=c, side_effect=e), r), bool)


def test_router_default_lenient_applies_to_read_but_not_write_tools() -> None:
    read = make_tool(int, side_effect=SideEffect.READ)
    write = make_tool(int, side_effect=SideEffect.WRITE)
    assert validate_call(read, '{"x": "5"}', router_coerce=True).x == 5  # type: ignore[attr-defined]
    with pytest.raises(ToolCallError):
        validate_call(write, '{"x": "5"}', router_coerce=True)


def test_write_tool_can_opt_in_explicitly() -> None:
    write = make_tool(int, side_effect=SideEffect.WRITE, coerce=True)
    assert validate_call(write, '{"x": "5"}').x == 5  # type: ignore[attr-defined]


def test_per_tool_false_overrides_lenient_router() -> None:
    strict = make_tool(int, coerce=False)
    with pytest.raises(ToolCallError):
        validate_call(strict, '{"x": "5"}', router_coerce=True)
