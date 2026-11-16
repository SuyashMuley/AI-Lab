from typing import Any

import pytest

from toolrouter import ToolCall, ToolRegistry, ToolRouter, tool
from toolrouter.results import truncate


def test_text_within_the_limit_is_untouched() -> None:
    assert truncate("abc", 3) == ("abc", False)
    assert truncate("", 10) == ("", False)
    assert truncate("x" * 100, 100) == ("x" * 100, False)  # exactly at the limit


def test_over_the_limit_gets_a_marker_and_stays_within_the_limit() -> None:
    text = "x" * 10_000
    out, was_truncated = truncate(text, 200)
    assert was_truncated
    assert len(out) <= 200
    assert out.startswith("x" * 100)
    assert out.endswith(" chars]")
    assert "\n[truncated " in out
    keep = out.index("\n[truncated ")
    dropped = int(out.split("[truncated ")[1].split(" chars]")[0])
    assert keep + dropped == len(text)  # the marker's count is exact


def test_tiny_limits_hard_cut_without_a_marker() -> None:
    out, was_truncated = truncate("abcdefghij" * 10, 10)
    assert (out, was_truncated) == ("abcdefghij", True)
    assert truncate("abc", 1) == ("a", True)


def test_counts_characters_not_bytes() -> None:
    text = "é" * 500
    out, was_truncated = truncate(text, 100)
    assert was_truncated
    assert len(out) <= 100
    assert out.startswith("é" * 50)


def call(name: str, args: Any = "{}") -> ToolCall:
    return ToolCall(id="c", name=name, arguments=args)


async def test_router_truncates_long_results_and_flags_them() -> None:
    @tool(max_output_chars=100)
    def big() -> str:
        """Large output."""
        return "y" * 50_000

    r = await ToolRouter(ToolRegistry([big])).dispatch(call("big"))
    assert r.ok  # truncation is not an error
    assert r.truncated
    assert r.content is not None
    assert len(r.content) <= 100
    assert "[truncated" in r.content
    assert r.error is None


async def test_router_does_not_flag_results_within_the_limit() -> None:
    @tool(max_output_chars=100)
    def small() -> str:
        """Small output."""
        return "y" * 100

    r = await ToolRouter(ToolRegistry([small])).dispatch(call("small"))
    assert r.content == "y" * 100
    assert not r.truncated


async def test_json_results_are_truncated_too() -> None:
    @tool(max_output_chars=120)
    def many() -> list[int]:
        """Lots of numbers."""
        return list(range(5000))

    r = await ToolRouter(ToolRegistry([many])).dispatch(call("many"))
    assert r.ok
    assert r.truncated
    assert r.content is not None
    assert r.content.startswith("[0,1,2,")
    assert len(r.content) <= 120


@pytest.mark.parametrize("limit", [1, 10, 48, 49, 60, 100, 1000])
async def test_the_cap_holds_for_any_limit(limit: int) -> None:
    @tool(max_output_chars=limit)
    def big() -> str:
        """Large output."""
        return "z" * 5000

    r = await ToolRouter(ToolRegistry([big])).dispatch(call("big"))
    assert r.content is not None
    assert len(r.content) <= limit
    assert r.truncated
