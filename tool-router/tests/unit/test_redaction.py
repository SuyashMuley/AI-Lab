import json
from typing import Any

import pytest

from tests.fakes import secret_tools as st
from tests.fakes.secret_tools import SENTINEL
from toolrouter import (
    ErrorCode,
    RecordingHook,
    ToolCall,
    ToolCallError,
    ToolRegistry,
    ToolRouter,
    validate_call,
)
from toolrouter.feedback import signature
from toolrouter.formats import anthropic, ollama, openai
from toolrouter.hooks import redacted_arguments
from toolrouter.redact import REDACTED, bound, contains_secret, redact_value, secret_at
from toolrouter.schema import redaction_schema, tool_schema


def walk(node: Any) -> Any:
    """Yield every dict in a nested structure."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


# ---- schemas: markers stay internal, secrets never reach a provider ----


@pytest.mark.parametrize("t", st.ALL, ids=lambda t: t.name)
def test_provider_schemas_carry_no_secret_markers_or_secret_defaults(t: Any) -> None:
    for payload in (
        tool_schema(t),
        anthropic.tool(t),
        openai.tool(t),
        openai.tool(t, api="responses"),
        ollama.tool(t),
    ):
        for node in walk(payload):
            assert "writeOnly" not in node
            assert node.get("format") != "password"
        assert "hunter2-default" not in json.dumps(payload)


def test_internal_schema_keeps_the_markers() -> None:
    props = redaction_schema(st.login)["properties"]
    assert props["password"]["writeOnly"] is True
    assert props["password"]["format"] == "password"
    assert redaction_schema(st.call_api)["properties"]["api_key"]["writeOnly"] is True
    assert redaction_schema(st.with_default)["properties"]["token"]["default"] == "hunter2-default"


def test_a_secret_default_is_not_shown_in_the_signature_either() -> None:
    assert signature(st.with_default) == 'with_default(note?: string = "n", token?: string)'


def test_non_secret_defaults_survive_for_providers() -> None:
    assert tool_schema(st.call_api)["properties"]["retries"]["default"] == 3


# ---- redact_value ----


def red(tool: Any, value: Any) -> Any:
    return redact_value(value, redaction_schema(tool))


def test_simple_secret_fields() -> None:
    assert red(st.login, {"user": "bob", "password": "pw"}) == {"user": "bob", "password": REDACTED}
    out = red(st.call_api, {"url": "http://x", "api_key": "k", "retries": 2})
    assert out == {"url": "http://x", "api_key": REDACTED, "retries": 2}


def test_nested_models_and_lists_of_models() -> None:
    value = {
        "creds": {"user": "u", "password": "pw"},
        "items": [{"name": "a", "token": "t1"}, {"name": "b", "token": "t2"}],
    }
    assert red(st.nested, value) == {
        "creds": {"user": "u", "password": REDACTED},
        "items": [{"name": "a", "token": REDACTED}, {"name": "b", "token": REDACTED}],
    }


def test_optional_secret_whether_set_or_not() -> None:
    assert red(st.optional_secret, {"note": "n", "key": "s"}) == {"note": "n", "key": REDACTED}
    assert red(st.optional_secret, {"note": "n"}) == {"note": "n"}


def test_map_of_secrets_and_list_of_secrets() -> None:
    assert red(st.secret_map, {"headers": {"a": "1", "b": "2"}, "label": "l"}) == {
        "headers": {"a": REDACTED, "b": REDACTED},
        "label": "l",
    }
    assert red(st.secret_list, {"tokens": ["x", "y"]}) == {"tokens": [REDACTED, REDACTED]}


def test_union_branches_each_redact_their_own_secret() -> None:
    assert red(st.union_tool, {"target": {"name": "n", "token": "t"}}) == {
        "target": {"name": "n", "token": REDACTED}
    }
    assert red(st.union_tool, {"target": {"user": "u", "password": "p"}}) == {
        "target": {"user": "u", "password": REDACTED}
    }


def test_free_form_dict_values_are_not_secret() -> None:
    value = {"meta": {"anything": "visible", "n": [1, 2]}, "label": "l"}
    assert red(st.freeform, value) == value


def test_unknown_fields_on_a_closed_object_are_redacted() -> None:
    out = red(st.login, {"user": "u", "password": "p", "api_key": SENTINEL})
    assert out["api_key"] == REDACTED  # never declared, so we cannot know it is safe
    assert SENTINEL not in json.dumps(out)


def test_redact_does_not_mutate_its_input() -> None:
    value = {"user": "u", "password": "pw"}
    red(st.login, value)
    assert value == {"user": "u", "password": "pw"}


def test_wrong_shapes_do_not_crash() -> None:
    for value in (None, 5, "s", [1], {"user": 1, "password": [1]}, {"password": {"a": 1}}):
        red(st.login, value)  # whatever the model sent, redaction itself must not raise


# ---- secret_at / contains_secret ----


def test_secret_at_resolves_locations() -> None:
    schema = redaction_schema(st.nested)
    assert secret_at(schema, ["creds", "password"])
    assert not secret_at(schema, ["creds", "user"])
    assert secret_at(schema, ["items", 0, "token"])
    assert not secret_at(schema, ["items", 3, "name"])
    assert not secret_at(schema, ["missing"]) or contains_secret(schema)


def test_secret_at_is_conservative_when_a_location_cannot_be_resolved() -> None:
    schema = redaction_schema(st.union_tool)
    # Pydantic adds union-branch tags to locations; we cannot follow them, so hide the value.
    assert secret_at(schema, ["target", "Item", "token"])
    assert secret_at(schema, ["target", "SomeTag", "name"])  # a secret exists below `target`
    assert not secret_at(redaction_schema(st.freeform), ["meta", "x"])


def test_contains_secret() -> None:
    assert contains_secret(redaction_schema(st.login))
    assert contains_secret(redaction_schema(st.secret_map))
    assert not contains_secret(redaction_schema(st.freeform))


# ---- validation errors never echo a secret ----


def fail(t: Any, args: Any) -> Any:
    with pytest.raises(ToolCallError) as ei:
        validate_call(t, args)
    return ei.value.info


def test_wrong_typed_secret_is_never_echoed() -> None:
    info = fail(st.login, {"user": "u", "password": 123456789})
    assert info.issues[0].got == REDACTED
    assert "123456789" not in info.message
    assert "(got [REDACTED])" in info.message


def test_secret_issues_in_nested_positions_are_redacted() -> None:
    info = fail(
        st.nested,
        {
            "creds": {"user": "u", "password": 111222333},
            "items": [{"name": "a", "token": 444555666}],
        },
    )
    assert {tuple(i.loc): i.got for i in info.issues} == {
        ("creds", "password"): REDACTED,
        ("items", 0, "token"): REDACTED,
    }
    assert "111222333" not in info.message
    assert "444555666" not in info.message


def test_non_secret_fields_still_show_what_was_received() -> None:
    info = fail(st.login, {"user": 42, "password": "pw"})
    assert info.issues[0].got == "42"


def test_secret_in_a_wrong_union_branch_is_hidden() -> None:
    info = fail(st.union_tool, {"target": {"name": "n", "token": 987654321}})
    assert "987654321" not in info.message
    assert all("987654321" not in (i.got or "") for i in info.issues)


def test_missing_secret_is_still_reported_as_missing() -> None:
    info = fail(st.login, {"user": "u"})
    assert info.issues[0].type == "missing"
    assert info.issues[0].got is None
    assert "password: Field required" in info.message


# ---- events, end to end ----


async def event_args(t: Any, arguments: Any) -> Any:
    rec = RecordingHook()
    await ToolRouter(ToolRegistry([t]), hooks=[rec]).dispatch(
        ToolCall(id="c", name=t.name, arguments=arguments)
    )
    return rec.before_events[0].arguments


async def test_events_never_contain_secret_values() -> None:
    assert await event_args(st.login, {"user": "u", "password": SENTINEL}) == {
        "user": "u",
        "password": REDACTED,
    }
    assert await event_args(st.call_api, json.dumps({"url": "u", "api_key": SENTINEL})) == {
        "url": "u",
        "api_key": REDACTED,
    }


async def test_events_redact_even_when_validation_fails() -> None:
    args = await event_args(st.login, {"user": 7, "password": SENTINEL, "oops": SENTINEL})
    assert SENTINEL not in json.dumps(args)
    assert args["user"] == 7


async def test_event_arguments_are_bounded() -> None:
    args = await event_args(st.freeform, {"meta": {f"k{i}": "v" * 800 for i in range(60)}})
    assert len(json.dumps(args)) < 8000
    assert "more keys" in json.dumps(args)


def test_redacted_arguments_is_total() -> None:
    for tool in (None, st.login):
        for raw in (None, 5, [1], "{", "", {}, {"a": float("nan")}, b"x", object()):
            assert redacted_arguments(tool, raw) is not None  # never raises


# ---- bound ----


def test_bound_cuts_strings_collections_and_depth() -> None:
    assert bound("x" * 10, max_str=4) == "xxxx…[+6 chars]"
    assert bound(list(range(30)), max_items=3) == [0, 1, 2, "…+27 more items"]
    assert bound({str(i): i for i in range(5)}, max_items=2) == {
        "0": 0,
        "1": 1,
        "…": "+3 more keys",
    }
    deep: Any = 1
    for _ in range(20):
        deep = [deep]
    assert bound(deep, max_depth=3) == [[["[…]"]]]
    assert bound(5) == 5
    assert bound(None) is None


def test_bound_results_are_json_serializable() -> None:
    json.dumps(bound({"a": [1, {"b": "c" * 1000}], "d": None}))


def test_secret_error_code_is_unaffected() -> None:
    assert ErrorCode.ARGUMENTS_INVALID.value == "arguments_invalid"
