import json

import pytest

from tests.fakes.sample_tools import ALL, get_weather, now, search, ship
from tests.snapshots import assert_matches_snapshot
from toolrouter import ToolRegistry
from toolrouter.formats import anthropic, ollama, openai
from toolrouter.schema import tool_schema


def test_anthropic_shape() -> None:
    p = anthropic.tool(get_weather)
    assert set(p) == {"name", "description", "input_schema"}
    assert p["name"] == "get_weather"
    assert p["description"] == "Current weather for a city."
    assert p["input_schema"] == tool_schema(get_weather)


def test_openai_chat_shape() -> None:
    p = openai.tool(get_weather)
    assert p["type"] == "function"
    assert set(p["function"]) == {"name", "description", "parameters"}
    assert p["function"]["parameters"] == tool_schema(get_weather)
    assert "strict" not in p["function"]  # strict mode deliberately not emitted yet


def test_openai_responses_shape_is_flat() -> None:
    p = openai.tool(get_weather, api="responses")
    assert p["type"] == "function"
    assert p["name"] == "get_weather"
    assert p["parameters"] == tool_schema(get_weather)
    assert "function" not in p


def test_ollama_shape_matches_openai_chat_envelope() -> None:
    assert ollama.tool(search) == openai.tool(search)


def test_tools_accepts_registry_or_any_iterable_preserving_order() -> None:
    reg = ToolRegistry(ALL)
    for fmt in (anthropic, ollama, openai):
        assert fmt.tools(reg) == fmt.tools(ALL) == fmt.tools(iter(ALL))
        assert len(fmt.tools(reg)) == len(ALL)
    assert [p["name"] for p in anthropic.tools(reg)] == reg.names()


@pytest.mark.parametrize("fmt", [anthropic, ollama, openai], ids=lambda m: m.__name__)
def test_payloads_are_json_serializable(fmt: object) -> None:
    json.dumps(fmt.tools(ALL))  # type: ignore[attr-defined]


def test_payloads_do_not_share_state() -> None:
    a = anthropic.tool(get_weather)
    a["input_schema"]["properties"].clear()
    assert anthropic.tool(get_weather)["input_schema"]["properties"]


# ---- snapshots (review diffs in git; regenerate with UPDATE_SNAPSHOTS=1) -------------------


def test_snapshot_anthropic() -> None:
    assert_matches_snapshot("anthropic_tools", anthropic.tools(ALL))


def test_snapshot_openai_chat() -> None:
    assert_matches_snapshot("openai_chat_tools", openai.tools(ALL))


def test_snapshot_openai_responses() -> None:
    assert_matches_snapshot("openai_responses_tools", openai.tools(ALL, api="responses"))


def test_snapshot_ollama() -> None:
    assert_matches_snapshot("ollama_tools", ollama.tools(ALL))


def test_snapshot_nested_schema() -> None:
    assert_matches_snapshot("schema_ship", tool_schema(ship))
    assert_matches_snapshot("schema_now", tool_schema(now))
