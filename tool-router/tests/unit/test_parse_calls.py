import json
from types import SimpleNamespace

from toolrouter.formats import anthropic, ollama, openai

# ---- Anthropic -----------------------------------------------------------------------------


def test_anthropic_dict_blocks_ignore_text() -> None:
    content = [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
        {"type": "tool_use", "id": "toolu_2", "name": "now", "input": {}},
    ]
    calls = anthropic.parse_calls(content)
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("toolu_1", "get_weather", {"city": "Oslo"}),
        ("toolu_2", "now", {}),
    ]


def test_anthropic_sdk_style_objects() -> None:
    content = [
        SimpleNamespace(type="text", text="hi"),
        SimpleNamespace(type="tool_use", id="t1", name="search", input={"query": "x"}),
    ]
    (call,) = anthropic.parse_calls(content)
    assert (call.id, call.name, call.arguments) == ("t1", "search", {"query": "x"})


def test_anthropic_missing_pieces_never_raise() -> None:
    calls = anthropic.parse_calls(
        [{"type": "tool_use"}, {"type": "tool_use", "id": "", "name": 5, "input": [1, 2]}, {}, 7]
    )
    assert [(c.id, c.name) for c in calls] == [("call_0", ""), ("call_1", "")]
    assert calls[0].arguments == {}
    assert calls[1].arguments == "[1, 2]"  # non-object input kept as text for the validator


# ---- OpenAI --------------------------------------------------------------------------------


def chat_call(id_: str, name: str, args: str) -> dict[str, object]:
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}


def test_openai_chat_message_and_bare_list() -> None:
    tool_calls = [chat_call("c1", "get_weather", '{"city": "Oslo"}'), chat_call("c2", "now", "{}")]
    for source in ({"role": "assistant", "tool_calls": tool_calls}, tool_calls):
        calls = openai.parse_calls(source)
        assert [(c.id, c.name, c.arguments) for c in calls] == [
            ("c1", "get_weather", '{"city": "Oslo"}'),
            ("c2", "now", "{}"),
        ]


def test_openai_chat_sdk_objects_and_no_tool_calls() -> None:
    msg = SimpleNamespace(
        tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="now", arguments="{}"))]
    )
    assert openai.parse_calls(msg)[0].name == "now"
    assert openai.parse_calls(SimpleNamespace(tool_calls=None)) == []
    assert openai.parse_calls({"content": "just text"}) == []


def test_openai_arguments_stay_as_raw_text_even_if_broken() -> None:
    (call,) = openai.parse_calls([chat_call("c1", "now", '{"city": ')])
    assert call.arguments == '{"city": '  # the validator reports INVALID_JSON, not the parser


def test_openai_responses_api() -> None:
    output = [
        {"type": "message", "content": []},
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city": "Oslo"}',
        },
    ]
    for source in ({"output": output}, output):
        (call,) = openai.parse_calls(source, api="responses")
        assert (call.id, call.name, call.arguments) == (
            "call_abc",
            "get_weather",
            '{"city": "Oslo"}',
        )


def test_openai_missing_ids_generated() -> None:
    calls = openai.parse_calls([chat_call("", "a", "{}"), chat_call("x", "b", "{}")])
    assert [c.id for c in calls] == ["call_0", "x"]


# ---- Ollama --------------------------------------------------------------------------------


def test_ollama_dict_message_generates_ids() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}},
            {"function": {"name": "now", "arguments": {}}},
        ],
    }
    calls = ollama.parse_calls(message)
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_0", "get_weather", {"city": "Oslo"}),
        ("call_1", "now", {}),
    ]


def test_ollama_sdk_objects_and_empty() -> None:
    msg = SimpleNamespace(
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name="now", arguments={}))]
    )
    assert ollama.parse_calls(msg)[0].name == "now"
    assert ollama.parse_calls(SimpleNamespace(tool_calls=None)) == []
    assert ollama.parse_calls({"content": "hi"}) == []


def test_parsed_calls_are_json_serializable() -> None:
    (call,) = ollama.parse_calls(
        {"tool_calls": [{"function": {"name": "n", "arguments": {"a": 1}}}]}
    )
    assert json.loads(call.model_dump_json())["arguments"] == {"a": 1}
