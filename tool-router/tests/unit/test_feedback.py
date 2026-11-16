import datetime
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field

from tests.fakes.factories import make_tool
from tests.fakes.sample_tools import ALL, blob, get_weather, now, search, title_tool
from tests.snapshots import assert_matches_snapshot
from toolrouter import (
    Confined,
    ErrorCode,
    StrictModel,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolRouter,
)
from toolrouter.calls import make_error
from toolrouter.feedback import EMPTY_RESULT, did_you_mean, model_text, signature

# ---- did you mean --------------------------------------------------------------------------

NAMES = ["get_weather", "get_forecast", "search", "now", "ship", "title_tool"]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("get_wether", ["get_weather"]),  # typo
        ("get_weathe", ["get_weather", "get_forecast"]),  # truncated
        ("weather", ["get_weather"]),  # partial name
        ("forecast", ["get_forecast"]),
        ("searh", ["search"]),
        ("SEARCH", ["search"]),  # case
        ("Get-Weather", ["get_weather"]),  # case + separator: exact match is the only answer
        ("get_weather", ["get_weather"]),
        ("zzz", []),
        ("", []),
        ("get", []),  # too vague to guess
    ],
)
def test_did_you_mean(typed: str, expected: list[str]) -> None:
    assert did_you_mean(typed, NAMES) == expected


def test_did_you_mean_is_deterministic_and_bounded() -> None:
    names = [f"tool_{c}" for c in "abcdefgh"]
    assert did_you_mean("tool_x", names) == ["tool_a", "tool_b", "tool_c"]  # ties: earlier first
    assert did_you_mean("tool_x", names, limit=1) == ["tool_a"]
    assert did_you_mean("tool_x", names) == did_you_mean("tool_x", names)
    assert did_you_mean("x", []) == []


def test_did_you_mean_ignores_everything_past_the_name_length_cap() -> None:
    assert did_you_mean("search" + "!" * 10_000, ["search"]) != did_you_mean("search", ["search"])
    huge = "a" * 100_000
    assert did_you_mean(huge, NAMES) == []  # and it returns promptly


# ---- signature -------------------------------------------------------------------------------


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class Addr(StrictModel):
    street: str
    zip: str | None = None


class Leaf(StrictModel):
    d: int


class Low(StrictModel):
    c: Leaf


class Mid(StrictModel):
    b: Low


class Top(StrictModel):
    a: Mid


SIGNATURES: list[tuple[Any, str]] = [
    (int, "t(x: integer)"),
    (float, "t(x: number)"),
    (bool, "t(x: boolean)"),
    (str, "t(x: string)"),
    (list[int], "t(x: integer[])"),
    (list[str | int], "t(x: (string | integer)[])"),
    (dict[str, int], "t(x: { [key: string]: integer })"),
    (dict[str, Any], "t(x: object)"),
    (Color, 't(x: "red" | "blue")'),
    (Literal["a", "b"], 't(x: "a" | "b")'),
    (Literal[1, 2], "t(x: 1 | 2)"),
    (int | None, "t(x: integer | null)"),
    (Annotated[int, Field(ge=1, le=50)], "t(x: integer (1-50))"),
    (Annotated[int, Field(gt=0)], "t(x: integer (>0))"),
    (Annotated[float, Field(le=1.5)], "t(x: number (<=1.5))"),
    (Annotated[str, Field(min_length=1, max_length=200)], "t(x: string (1-200 chars))"),
    (Annotated[str, Field(min_length=3)], "t(x: string (>=3 chars))"),
    (datetime.datetime, "t(x: string (date-time))"),
    (uuid.UUID, "t(x: string (uuid))"),
    (Path, "t(x: string (path))"),
    (Annotated[Path, Confined("/tmp")], "t(x: string (path))"),
    (Addr, "t(x: { street: string, zip?: string | null = null })"),
    (list[Addr], "t(x: ({ street: string, zip?: string | null = null })[])"),
    (tuple[int, str], "t(x: [integer, string])"),
    (set[int], "t(x: integer[])"),
    (
        Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        't(x: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | …)',
    ),
]


@pytest.mark.parametrize(("annotation", "expected"), SIGNATURES, ids=[e for _, e in SIGNATURES])
def test_signature_table(annotation: Any, expected: str) -> None:
    assert signature(make_tool(annotation)) == expected


def test_signature_marks_optional_params_and_defaults() -> None:
    assert signature(get_weather) == 'get_weather(city: string, units?: "c" | "f" = "c")'
    assert signature(now) == "now()"
    assert signature(title_tool) == "title_tool(title: string, properties?: integer = 0)"
    assert signature(blob) == "blob(meta: object)"
    assert signature(search) == (
        "search(query: string (1-200 chars), limit?: integer (1-50) = 10, "
        "tags?: string[] | null = null)"
    )


def test_deep_nesting_is_cut_off() -> None:
    text = signature(make_tool(Top))
    assert "{…}" in text
    assert "d: int" not in text


def test_huge_signatures_are_bounded() -> None:
    from pydantic import create_model

    from toolrouter import tool

    fields: dict[str, Any] = {f"field_number_{i}": (int, ...) for i in range(200)}
    model = create_model("Big", __base__=StrictModel, **fields)
    big = tool(model=model, name="big", description="d")(lambda args: None)
    text = signature(big)
    assert len(text) <= 300
    assert text.startswith("big(field_number_0: integer")
    assert text.endswith("…)")


def test_long_defaults_are_shortened() -> None:
    from toolrouter import tool

    @tool
    def f(blurb: str = "x" * 500) -> str:
        """Doc."""
        return blurb

    text = signature(f)
    assert len(text) < 120
    assert "…" in text


# ---- messages --------------------------------------------------------------------------------


def router() -> ToolRouter:
    return ToolRouter(ToolRegistry(ALL))


async def err(name: str, args: Any = "{}") -> str:
    r = await router().dispatch(ToolCall(id="c", name=name, arguments=args))
    assert r.error is not None
    return r.error.message


async def test_unknown_tool_suggests_and_lists() -> None:
    msg = await err("get_wether")
    assert "Did you mean 'get_weather'?" in msg
    assert "Available tools: get_weather, search, ship, now, title_tool." in msg


async def test_unknown_tool_without_a_close_match_just_lists() -> None:
    msg = await err("zzz")
    assert "Did you mean" not in msg
    assert msg.startswith("Unknown tool 'zzz'. Available tools:")


async def test_invalid_arguments_show_what_was_expected() -> None:
    msg = await err("get_weather", '{"city": "Oslo", "units": "kelvin"}')
    assert "units: Input should be 'c' or 'f' (got 'kelvin')" in msg
    assert msg.endswith('Expected: get_weather(city: string, units?: "c" | "f" = "c")')


async def test_invalid_json_shows_what_was_expected() -> None:
    msg = await err("get_weather", '{"city": ')
    assert "not valid JSON" in msg
    assert "Send a single valid JSON object." in msg
    assert msg.endswith('Expected: get_weather(city: string, units?: "c" | "f" = "c")')


async def test_non_object_json_also_shows_the_signature() -> None:
    msg = await err("search", "[1, 2]")
    assert "got an array" in msg
    assert "Expected: search(query: string (1-200 chars)" in msg


async def test_oversized_arguments_message_is_unchanged_and_has_no_signature() -> None:
    msg = await err("get_weather", '{"city": "' + "x" * 70_000 + '"}')
    assert "Send smaller or flatter arguments" in msg
    assert "Expected:" not in msg


async def test_error_messages_stay_bounded_for_hostile_input() -> None:
    assert len(await err("x" * 100_000)) < 500
    bad = {"title": "x" * 50_000, "properties": "y" * 50_000, "z": "w" * 50_000}
    assert len(await err("title_tool", bad)) < 1500


async def test_snapshot_of_every_kind_of_repair_message() -> None:
    cases: dict[str, tuple[str, Any]] = {
        "unknown_typo": ("get_wether", "{}"),
        "unknown_garbage": ("zzz", "{}"),
        "invalid_json": ("get_weather", '{"city": '),
        "not_an_object": ("search", "[1]"),
        "duplicate_key": ("get_weather", '{"city": "a", "city": "b"}'),
        "missing_field": ("get_weather", "{}"),
        "wrong_type": ("search", '{"query": "x", "limit": "ten"}'),
        "out_of_range": ("search", '{"query": "x", "limit": 0}'),
        "bad_enum": ("get_weather", '{"city": "x", "units": "kelvin"}'),
        "extra_field": ("now", '{"x": 1}'),
        "nested": (
            "ship",
            '{"address": {"street": 1, "city": "c"}, "extras": [{"street": "s"}]}',
        ),
    }
    messages = {k: await err(name, args) for k, (name, args) in cases.items()}
    assert_matches_snapshot("error_messages", messages)


# ---- model_text --------------------------------------------------------------------------------


def test_model_text_success_is_the_content() -> None:
    assert model_text(ToolResult(call_id="1", name="n", ok=True, content="hello")) == "hello"


def test_model_text_empty_success_gets_a_placeholder() -> None:
    assert model_text(ToolResult(call_id="1", name="n", ok=True, content="")) == EMPTY_RESULT


def test_model_text_failure_is_prefixed_with_the_code() -> None:
    err_info = make_error(ErrorCode.TOOL_TIMEOUT, "Tool 'x' did not finish within 5s.")
    r = ToolResult(call_id="1", name="x", ok=False, error=err_info)
    assert model_text(r) == "Error (tool_timeout): Tool 'x' did not finish within 5s."


def test_raw_text_property_is_unchanged_by_formatting() -> None:
    err_info = make_error(ErrorCode.CANCELLED, "bye")
    assert ToolResult(call_id="1", name="x", ok=False, error=err_info).text == "bye"
