import copy
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tests.fakes.sample_tools import ALL, get_weather, now, search, ship, title_tool
from toolrouter import ToolDefinitionError
from toolrouter.schema import args_schema, normalize_schema, tool_schema


def walk_schemas(node: Any) -> Any:
    """Yield every schema node (not data) in a normalized schema."""
    if not isinstance(node, dict):
        return
    yield node
    for key in ("properties", "patternProperties"):
        for sub in (node.get(key) or {}).values():
            yield from walk_schemas(sub)
    for key in ("items", "additionalProperties"):
        yield from walk_schemas(node.get(key))
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(key) or []:
            yield from walk_schemas(sub)


@pytest.mark.parametrize("t", ALL, ids=lambda t: t.name)
def test_every_schema_is_valid_json_schema_and_clean(t: Any) -> None:
    schema = tool_schema(t)
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "$defs" not in str(schema)
    assert "$ref" not in str(schema)
    for node in walk_schemas(schema):
        assert "title" not in node
        assert "discriminator" not in node
        if "properties" in node:
            assert node.get("additionalProperties") is False


def test_basic_shape_defaults_and_required() -> None:
    s = tool_schema(get_weather)
    assert s["required"] == ["city"]
    assert s["properties"]["city"] == {"type": "string", "description": "Name of the city."}
    assert s["properties"]["units"]["default"] == "c"
    assert s["properties"]["units"]["enum"] == ["c", "f"]


def test_constraints_survive() -> None:
    p = tool_schema(search)["properties"]
    assert p["query"]["minLength"] == 1
    assert p["query"]["maxLength"] == 200
    assert p["limit"]["minimum"] == 1
    assert p["limit"]["maximum"] == 50


def test_optional_becomes_anyof_with_null() -> None:
    tags = tool_schema(search)["properties"]["tags"]
    assert {"type": "null"} in tags["anyOf"]
    assert {"type": "array", "items": {"type": "string"}} in tags["anyOf"]


def test_nested_models_and_enums_are_inlined() -> None:
    s = tool_schema(ship)
    address = s["properties"]["address"]
    assert address["type"] == "object"
    assert address["required"] == ["street", "city"]
    assert address["properties"]["city"]["description"] == "City name."
    assert s["properties"]["extras"]["items"] == address  # list items inlined too
    color = s["properties"]["color"]
    assert color["enum"] == ["red", "blue"]
    assert color["default"] == "red"


def test_no_argument_tool_has_empty_closed_object() -> None:
    assert tool_schema(now) == {"type": "object", "properties": {}, "additionalProperties": False}


def test_field_names_that_look_like_keywords_are_preserved() -> None:
    s = tool_schema(title_tool)
    assert set(s["properties"]) == {"title", "properties"}
    assert s["properties"]["title"]["type"] == "string"
    assert s["required"] == ["title"]


def test_root_description_dropped() -> None:
    raw = {"type": "object", "description": "Class docstring.", "properties": {}, "title": "X"}
    assert "description" not in normalize_schema(raw)
    assert "title" not in normalize_schema(raw)


def test_single_allof_wrapper_flattened_parent_wins() -> None:
    raw: dict[str, Any] = {
        "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}}},
        "type": "object",
        "properties": {
            "x": {"allOf": [{"$ref": "#/$defs/Inner"}], "description": "From the field."}
        },
    }
    x = normalize_schema(raw)["properties"]["x"]
    assert "allOf" not in x
    assert x["description"] == "From the field."
    assert x["properties"]["a"] == {"type": "integer"}


def test_ref_siblings_override_resolved_definition() -> None:
    raw: dict[str, Any] = {
        "$defs": {"D": {"type": "string", "description": "def", "title": "D"}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/D", "description": "field"}},
    }
    assert normalize_schema(raw)["properties"]["x"] == {"type": "string", "description": "field"}


def test_discriminator_dropped_but_branches_inlined() -> None:
    raw: dict[str, Any] = {
        "$defs": {
            "Cat": {"type": "object", "properties": {"kind": {"const": "cat"}}},
            "Dog": {"type": "object", "properties": {"kind": {"const": "dog"}}},
        },
        "type": "object",
        "properties": {
            "pet": {
                "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                "discriminator": {
                    "propertyName": "kind",
                    "mapping": {"cat": "#/$defs/Cat", "dog": "#/$defs/Dog"},
                },
            }
        },
    }
    pet = normalize_schema(raw)["properties"]["pet"]
    assert "discriminator" not in pet
    assert pet["oneOf"][0]["properties"]["kind"] == {"const": "cat"}


def test_unknown_or_external_refs_rejected() -> None:
    with pytest.raises(ToolDefinitionError, match="unsupported schema reference"):
        normalize_schema({"type": "object", "properties": {"x": {"$ref": "http://x/y"}}}, "t")
    with pytest.raises(ToolDefinitionError, match="unsupported schema reference"):
        normalize_schema({"type": "object", "properties": {"x": {"$ref": "#/$defs/Nope"}}}, "t")


def test_data_values_are_not_rewritten() -> None:
    raw = {
        "type": "object",
        "properties": {"x": {"default": {"title": "keep", "$ref": "literal"}, "type": "object"}},
    }
    assert normalize_schema(raw)["properties"]["x"]["default"] == {
        "title": "keep",
        "$ref": "literal",
    }


def test_normalize_does_not_mutate_input() -> None:
    raw = ship.args_model.model_json_schema()
    before = copy.deepcopy(raw)
    normalize_schema(raw)
    assert raw == before


def test_args_schema_is_a_fresh_copy_each_time() -> None:
    a = args_schema(get_weather.args_model, "get_weather")
    a["properties"].clear()
    assert args_schema(get_weather.args_model, "get_weather")["properties"]
