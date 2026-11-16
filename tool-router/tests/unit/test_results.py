import pytest

from tests.fakes import router_tools as rt
from toolrouter import ErrorCode, ToolCallError, ToolDefinitionError, serialize_result, tool


def ser(t: object, value: object) -> str:
    return serialize_result(t, value)  # type: ignore[arg-type]


def test_exact_str_passes_through_raw() -> None:
    assert ser(rt.echo, 'hello "world"\n') == 'hello "world"\n'
    assert ser(rt.echo, "") == ""


def test_models_lists_dicts_numbers_become_json() -> None:
    assert ser(rt.report, rt.Report(city="Oslo", temp=21.5)) == '{"city":"Oslo","temp":21.5}'
    assert ser(rt.reports, [rt.Report(city="a", temp=1.0)]) == '[{"city":"a","temp":1.0}]'
    assert ser(rt.stats, {"a": 1}) == '{"a":1}'
    assert ser(rt.add, 5) == "5"


def test_dicts_that_strictly_satisfy_a_declared_model_are_accepted() -> None:
    # Strict mode accepts a dict for a model type, but still validates every field strictly.
    assert ser(rt.report, {"city": "x", "temp": 1.0}) == '{"city":"x","temp":1.0}'
    assert ser(rt.reports, [{"city": "x", "temp": 1}]) == '[{"city":"x","temp":1.0}]'


def test_none_datetime_enum() -> None:
    assert ser(rt.nothing, None) == "null"
    import datetime

    assert ser(rt.when, datetime.datetime(2024, 1, 2, 3, 4, 5)) == '"2024-01-02T03:04:05"'
    assert ser(rt.pick, rt.Color.RED) == '"red"'


def test_optional_str_result_is_raw_text_or_null() -> None:
    assert ser(rt.maybe, "yes") == "yes"
    assert ser(rt.maybe, None) == "null"


def test_unannotated_and_any_are_serialized_not_validated() -> None:
    assert ser(rt.untyped, {"x": 3}) == '{"x":3}'
    assert ser(rt.untyped, rt.Report(city="a", temp=1.0)) == '{"city":"a","temp":1.0}'
    assert ser(rt.any_typed, [1, "a", None]) == '[1,"a",null]'
    assert rt.untyped.result_adapter is None
    assert rt.any_typed.result_adapter is None


@pytest.mark.parametrize(
    ("t", "value"),
    [
        (rt.bad_int, "oops"),  # wrong type
        (rt.bad_int, 5.5),
        (rt.bad_int, True),  # bool is not an int in strict mode
        (rt.report, {"city": 1, "temp": "hot"}),  # dict that does not satisfy the model
        (rt.report, rt.Other(city="x", temp=1.0)),  # a different model class
        (rt.echo, 5),  # int where str is declared
        (rt.reports, [{"city": 1, "temp": 1.0}]),
        (rt.stats, {"a": "1"}),  # strict: no coercion of results
        (rt.pick, "red"),  # python-mode strict wants the enum member
    ],
)
def test_declared_type_mismatch_is_result_invalid(t: object, value: object) -> None:
    with pytest.raises(ToolCallError) as ei:
        ser(t, value)
    info = ei.value.info
    assert info.code is ErrorCode.RESULT_INVALID
    assert not info.retryable
    assert "bug in the tool" in info.message
    assert ei.value.__cause__ is not None  # details kept for logs


def test_result_invalid_message_never_contains_the_value() -> None:
    with pytest.raises(ToolCallError) as ei:
        ser(rt.bad_int, "SECRET-VALUE-123")
    assert "SECRET-VALUE-123" not in ei.value.info.message


def test_unserializable_unannotated_result() -> None:
    with pytest.raises(ToolCallError) as ei:
        ser(rt.unserializable, rt.Opaque())
    assert ei.value.info.code is ErrorCode.RESULT_INVALID


# ---- definition-time handling of return annotations --------------------------------------


def test_unvalidatable_return_annotation_rejected_at_definition() -> None:
    with pytest.raises(ToolDefinitionError, match="return type"):

        @tool
        def f() -> rt.Opaque:
            """Doc."""
            return rt.Opaque()


def test_async_handlers_use_the_declared_result_type() -> None:
    assert rt.slow_add.result_type is int
    assert rt.slow_add.result_adapter is not None
