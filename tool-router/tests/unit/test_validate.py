import time
from typing import Any

import pytest
from pydantic import field_validator

from tests.fakes.factories import make_tool
from tests.fakes.sample_tools import blob, get_weather, now, search, ship, title_tool
from toolrouter import (
    Limits,
    StrictModel,
    ToolCallError,
    ToolDefinitionError,
    tool,
    validate_call,
)
from toolrouter.calls import RETRYABLE
from toolrouter.errors import ErrorCode


def fail(t: Any, args: Any, **kw: Any) -> Any:
    with pytest.raises(ToolCallError) as ei:
        validate_call(t, args, **kw)
    return ei.value.info


# ---- success paths -------------------------------------------------------------------------


def test_valid_call_returns_args_model_instance() -> None:
    args = validate_call(get_weather, '{"city": "Oslo"}')
    assert isinstance(args, get_weather.args_model)
    assert args.city == "Oslo"  # type: ignore[attr-defined]
    assert args.units == "c"  # type: ignore[attr-defined]


def test_dict_and_text_are_equivalent() -> None:
    a = validate_call(get_weather, {"city": "Oslo", "units": "f"})
    b = validate_call(get_weather, '{"city": "Oslo", "units": "f"}')
    assert a == b


@pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
def test_empty_arguments_mean_no_arguments(empty: str) -> None:
    assert validate_call(now, empty) is not None
    info = fail(get_weather, empty)  # but required fields are still required
    assert info.code is ErrorCode.ARGUMENTS_INVALID
    assert info.issues[0].type == "missing"


def test_no_argument_tool_rejects_extras() -> None:
    info = fail(now, '{"x": 1}')
    assert info.issues[0].type == "extra_forbidden"


def test_unicode_arguments() -> None:
    assert validate_call(get_weather, '{"city": "Zürich 東京 🌦"}').city == "Zürich 東京 🌦"  # type: ignore[attr-defined]


# ---- JSON-level rejections -----------------------------------------------------------------


def test_invalid_json_reports_position_and_is_retryable() -> None:
    info = fail(get_weather, '{"city": ')
    assert info.code is ErrorCode.INVALID_JSON
    assert info.retryable
    assert "line 1 column" in info.message


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("[1, 2]", "an array"),
        ('"hello"', "a string"),
        ("5", "a number"),
        ("true", "a boolean"),
        ("null", "null"),
    ],
)
def test_non_object_json_rejected(text: str, kind: str) -> None:
    info = fail(get_weather, text)
    assert info.code is ErrorCode.INVALID_JSON
    assert f"got {kind}" in info.message


def test_duplicate_keys_rejected() -> None:
    info = fail(get_weather, '{"city": "a", "city": "b"}')
    assert info.code is ErrorCode.INVALID_JSON
    assert "duplicate key 'city'" in info.message


@pytest.mark.parametrize("const", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_constants_rejected(const: str) -> None:
    info = fail(make_tool(float), f'{{"x": {const}}}')
    assert info.code is ErrorCode.INVALID_JSON
    assert const in info.message


@pytest.mark.parametrize("bad", [None, 5, [1], b"{}", object()])
def test_wrong_container_type_rejected(bad: Any) -> None:
    assert fail(get_weather, bad).code is ErrorCode.INVALID_JSON


def test_lone_surrogate_text_rejected_without_crashing() -> None:
    assert fail(get_weather, '{"city": "\ud800"}').code is ErrorCode.INVALID_JSON


def test_non_json_dict_inputs_rejected() -> None:
    assert fail(blob, {"meta": {"x": object()}}).code is ErrorCode.INVALID_JSON
    assert fail(blob, {"meta": {"x": float("nan")}}).code is ErrorCode.INVALID_JSON
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    assert fail(blob, {"meta": cyclic}).code is ErrorCode.INVALID_JSON


def test_non_string_dict_keys_are_stringified_then_checked() -> None:
    info = fail(get_weather, {1: "x", "city": "Oslo"})
    assert info.issues[0].type == "extra_forbidden"
    assert info.issues[0].loc == ["1"]


# ---- limits ----------------------------------------------------------------------------------


def test_size_limit_exact_boundary() -> None:
    limits = Limits(max_bytes=100)
    prefix, suffix = '{"city": "', '"}'
    ok = prefix + "a" * (100 - len(prefix) - len(suffix)) + suffix
    assert len(ok) == 100
    assert validate_call(get_weather, ok, limits=limits) is not None
    assert fail(get_weather, ok + " ", limits=limits).code is ErrorCode.ARGUMENTS_TOO_LARGE


def test_size_limit_counts_bytes_not_characters() -> None:
    limits = Limits(max_bytes=100)
    text = '{"city": "' + "é" * 50 + '"}'  # 62 chars but 112 bytes
    assert len(text) < 100 < len(text.encode())
    assert fail(get_weather, text, limits=limits).code is ErrorCode.ARGUMENTS_TOO_LARGE


def nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def test_depth_limit_exact_boundary() -> None:
    limits = Limits(max_depth=5)
    # {"meta": {"k": [[..]]}} -> root(1) + meta(2) + list depth d
    assert validate_call(blob, '{"meta": {"k": ' + nested(3) + "}}", limits=limits) is not None
    info = fail(blob, '{"meta": {"k": ' + nested(4) + "}}", limits=limits)
    assert info.code is ErrorCode.ARGUMENTS_TOO_LARGE
    assert info.retryable


def test_brackets_inside_strings_do_not_count_as_depth() -> None:
    limits = Limits(max_depth=3)
    text = '{"meta": {"k": "[[[[[[[[[[[ \\" {{{{{{{{ "}}'
    assert validate_call(blob, text, limits=limits) is not None


def test_hostile_nesting_rejected_fast() -> None:
    start = time.perf_counter()
    assert fail(blob, '{"meta": ' + nested(30_000) + "}").code is ErrorCode.ARGUMENTS_TOO_LARGE
    assert fail(blob, nested(60_000)).code is ErrorCode.ARGUMENTS_TOO_LARGE  # over size cap
    deep: Any = {}
    for _ in range(5_000):
        deep = {"k": deep}
    assert fail(blob, {"meta": deep}).code is ErrorCode.ARGUMENTS_TOO_LARGE  # recursion path
    assert time.perf_counter() - start < 2.0


def test_default_limits_allow_reasonable_payloads() -> None:
    big = {"meta": {f"k{i}": "v" * 20 for i in range(1000)}}  # ~30 KB
    assert validate_call(blob, big) is not None


def test_limits_must_be_positive() -> None:
    for field in ("max_bytes", "max_depth", "max_issues", "got_chars"):
        with pytest.raises(ValueError):
            Limits(**{field: 0})
        with pytest.raises(ValueError):
            Limits(**{field: True})


# ---- validation issues ---------------------------------------------------------------------


def test_missing_field_issue() -> None:
    info = fail(get_weather, "{}")
    assert info.code is ErrorCode.ARGUMENTS_INVALID
    (issue,) = info.issues
    assert (issue.loc, issue.type, issue.got) == (["city"], "missing", None)
    assert "city: Field required" in info.message
    assert info.retryable


def test_extra_field_issue_names_the_field_but_never_echoes_its_value() -> None:
    info = fail(get_weather, '{"city": "Oslo", "bogus": "hunter2-maybe-a-secret"}')
    (issue,) = info.issues
    assert (issue.loc, issue.type, issue.got) == (["bogus"], "extra_forbidden", None)
    assert "bogus: Extra inputs are not permitted" in info.message
    assert "hunter2" not in info.message  # an undeclared field may be anything, even a secret


def test_container_values_are_described_not_echoed() -> None:
    info = fail(get_weather, {"city": {"password": "hunter2", "n": 1}})
    assert info.issues[0].got == "<object, 2 keys>"
    assert "hunter2" not in info.message
    info = fail(get_weather, {"city": ["a", "b", "c"]})
    assert info.issues[0].got == "<array, 3 items>"


def test_hostile_field_names_are_clipped() -> None:
    long_key = "k" * 50_000
    info = fail(get_weather, {"city": "Oslo", long_key: 1})
    assert len(info.message) < 600
    assert len(info.issues[0].loc[0]) <= 60


def test_wrong_type_issue_shows_got() -> None:
    info = fail(search, '{"query": "x", "limit": "ten"}')
    (issue,) = info.issues
    assert issue.loc == ["limit"]
    assert issue.got == "'ten'"
    assert "limit: Input should be a valid integer" in info.message
    assert "(got 'ten')" in info.message


def test_constraint_violation_issue() -> None:
    info = fail(search, '{"query": "x", "limit": 0}')
    assert info.issues[0].type == "greater_than_equal"


def test_nested_locations_with_list_indexes() -> None:
    info = fail(ship, '{"address": {"street": 1, "city": "c"}, "extras": [{"street": "s"}]}')
    locs = {tuple(i.loc) for i in info.issues}
    assert ("address", "street") in locs
    assert ("extras", 0, "city") in locs
    assert "address.street" in info.message
    assert "extras[0].city" in info.message


def test_enum_error_lists_choices() -> None:
    info = fail(get_weather, '{"city": "x", "units": "kelvin"}')
    assert "'c' or 'f'" in info.issues[0].msg
    assert info.issues[0].got == "'kelvin'"


def test_issue_cap_and_more_marker() -> None:
    limits = Limits(max_issues=2)
    info = fail(title_tool, '{"title": 1, "properties": "x", "z": 1, "y": 2}', limits=limits)
    assert len(info.issues) == 2
    assert "(+2 more)" in info.message


def test_got_preview_is_truncated() -> None:
    info = fail(search, '{"query": 5, "limit": 1}', limits=Limits(got_chars=5))
    assert info.issues[0].got == "5"
    long = fail(make_tool(int), '{"x": "' + "z" * 500 + '"}', limits=Limits(got_chars=10))
    assert long.issues[0].got is not None
    assert len(long.issues[0].got) == 10
    assert long.issues[0].got.endswith("…")


def test_message_never_echoes_whole_payload() -> None:
    secret = "s3cr3t-" + "x" * 5000
    info = fail(search, '{"query": ' + f'"{secret}"' + ', "limit": "bad"}')
    assert secret not in info.message
    assert all(len(i.got or "") <= 60 for i in info.issues)


# ---- failures inside validators ------------------------------------------------------------


def test_value_error_in_validator_is_a_normal_issue() -> None:
    class A(StrictModel):
        n: int

        @field_validator("n")
        @classmethod
        def positive(cls, v: int) -> int:
            if v < 0:
                raise ValueError("must not be negative")
            return v

    t = tool(model=A, name="a", description="d")(lambda args: None)
    info = fail(t, '{"n": -1}')
    assert info.code is ErrorCode.ARGUMENTS_INVALID
    assert "must not be negative" in info.message


def test_crashing_validator_is_internal_error_not_model_blame() -> None:
    class A(StrictModel):
        n: int

        @field_validator("n")
        @classmethod
        def boom(cls, v: int) -> int:
            raise KeyError("bug in our validator, path /etc/secret")

    t = tool(model=A, name="a", description="d")(lambda args: None)
    info = fail(t, '{"n": 1}')
    assert info.code is ErrorCode.INTERNAL_ERROR
    assert not info.retryable
    assert "/etc/secret" not in info.message


# ---- error taxonomy hygiene -----------------------------------------------------------------


def test_every_error_code_has_a_retryable_flag() -> None:
    assert set(RETRYABLE) == set(ErrorCode)


def test_definition_errors_stay_separate_from_call_errors() -> None:
    assert not issubclass(ToolCallError, ToolDefinitionError)
