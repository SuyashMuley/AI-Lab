import json
from typing import Any

import pytest

from tests.fakes import router_tools as rt
from toolrouter import ErrorCode, ToolCall, ToolRegistry, ToolResult, ToolRouter
from toolrouter.calls import make_error
from toolrouter.feedback import EMPTY_RESULT
from toolrouter.formats import anthropic, ollama, openai


def ok(id_: str, content: str, name: str = "echo") -> ToolResult:
    return ToolResult(call_id=id_, name=name, ok=True, content=content)


def bad(id_: str, name: str = "echo") -> ToolResult:
    return ToolResult(
        call_id=id_,
        name=name,
        ok=False,
        error=make_error(ErrorCode.ARGUMENTS_INVALID, "city: Field required"),
    )


RESULTS = [ok("a", "sunny"), bad("b", "get_weather"), ok("c", "", "nothing")]


# ---- Anthropic -------------------------------------------------------------------------------


def test_anthropic_all_results_go_in_one_user_message_in_order() -> None:
    (message,) = anthropic.result_messages(RESULTS)
    assert message["role"] == "user"
    assert [b["tool_use_id"] for b in message["content"]] == ["a", "b", "c"]
    assert all(b["type"] == "tool_result" for b in message["content"])


def test_anthropic_block_contents_and_error_flag() -> None:
    blocks = anthropic.result_messages(RESULTS)[0]["content"]
    assert blocks[0] == {"type": "tool_result", "tool_use_id": "a", "content": "sunny"}
    assert blocks[1]["is_error"] is True
    assert blocks[1]["content"] == "Error (arguments_invalid): city: Field required"
    assert "is_error" not in blocks[0]
    assert blocks[2]["content"] == EMPTY_RESULT


def test_anthropic_no_results_means_no_message() -> None:
    assert anthropic.result_messages([]) == []


# ---- OpenAI ----------------------------------------------------------------------------------


def test_openai_chat_one_tool_message_per_result() -> None:
    messages = openai.result_messages(RESULTS)
    assert [m["role"] for m in messages] == ["tool"] * 3
    assert [m["tool_call_id"] for m in messages] == ["a", "b", "c"]
    assert messages[0]["content"] == "sunny"
    assert messages[1]["content"] == "Error (arguments_invalid): city: Field required"
    assert messages[2]["content"] == EMPTY_RESULT
    assert openai.result_messages([]) == []


def test_openai_responses_function_call_output_items() -> None:
    items = openai.result_messages(RESULTS, api="responses")
    assert [i["type"] for i in items] == ["function_call_output"] * 3
    assert [i["call_id"] for i in items] == ["a", "b", "c"]
    assert items[1]["output"].startswith("Error (arguments_invalid)")
    assert "role" not in items[0]


# ---- Ollama ----------------------------------------------------------------------------------


def test_ollama_tool_messages_use_tool_name() -> None:
    messages = ollama.result_messages(RESULTS)
    assert [m["role"] for m in messages] == ["tool"] * 3
    assert [m["tool_name"] for m in messages] == ["echo", "get_weather", "nothing"]
    assert messages[0]["content"] == "sunny"
    assert messages[1]["content"].startswith("Error (arguments_invalid)")
    assert ollama.result_messages([]) == []


@pytest.mark.parametrize(
    "fn",
    [
        anthropic.result_messages,
        openai.result_messages,
        lambda r: openai.result_messages(r, api="responses"),
        ollama.result_messages,
    ],
)
def test_all_payloads_are_json_serializable_and_iterables_work(fn: Any) -> None:
    json.dumps(fn(RESULTS))
    json.dumps(fn(iter(RESULTS)))


def test_unicode_and_large_results_pass_through_untouched() -> None:
    text = "Zürich 東京 🌦 " * 1000
    assert anthropic.result_messages([ok("a", text)])[0]["content"][0]["content"] == text
    assert openai.result_messages([ok("a", text)])[0]["content"] == text


# ---- full turns: parse -> dispatch -> messages -------------------------------------------------


def registry() -> ToolRegistry:
    return ToolRegistry([rt.echo, rt.add, rt.crash])


async def test_anthropic_full_turn() -> None:
    content = [
        {"type": "text", "text": "Working on it."},
        {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}},
        {"type": "tool_use", "id": "toolu_2", "name": "ad", "input": {"a": 1}},
        {"type": "tool_use", "id": "toolu_3", "name": "echo", "input": {"text": "hi"}},
    ]
    results = await ToolRouter(registry()).dispatch_many(anthropic.parse_calls(content))
    (message,) = anthropic.result_messages(results)
    blocks = message["content"]
    assert [b["tool_use_id"] for b in blocks] == ["toolu_1", "toolu_2", "toolu_3"]
    assert [b["content"] for b in blocks][0] == "3"
    assert "Did you mean 'add'?" in blocks[1]["content"]
    assert blocks[1]["is_error"] is True
    assert blocks[2]["content"] == "hi"


async def test_openai_full_turn() -> None:
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 5}'}},
        {"id": "call_2", "type": "function", "function": {"name": "add", "arguments": '{"a": '}},
    ]
    results = await ToolRouter(registry()).dispatch_many(openai.parse_calls(tool_calls))
    messages = openai.result_messages(results)
    assert [m["tool_call_id"] for m in messages] == ["call_1", "call_2"]
    assert messages[0]["content"] == "5"
    assert messages[1]["content"].startswith("Error (invalid_json):")
    assert "Expected: add(a: integer, b?: integer = 0)" in messages[1]["content"]


async def test_ollama_full_turn() -> None:
    message = {
        "tool_calls": [
            {"function": {"name": "echo", "arguments": {"text": "hi"}}},
            {"function": {"name": "crash", "arguments": {"path": "/etc/passwd"}}},
        ]
    }
    results = await ToolRouter(registry()).dispatch_many(ollama.parse_calls(message))
    out = ollama.result_messages(results)
    assert [m["tool_name"] for m in out] == ["echo", "crash"]
    assert out[0]["content"] == "hi"
    assert out[1]["content"] == "Error (tool_failed): Tool 'crash' failed unexpectedly (KeyError)."


# ---- a scripted model repairing its own mistake -------------------------------------------------


async def test_a_model_can_repair_a_bad_call_from_the_feedback_alone() -> None:
    """The first call is wrong; a 'model' that only reads the error text fixes it."""
    router = ToolRouter(registry())
    first = await router.dispatch(ToolCall(id="1", name="add", arguments={"a": "two", "bx": 3}))
    assert not first.ok
    from toolrouter.feedback import model_text

    feedback = model_text(first)
    # Everything needed to fix the call is in the text: what was wrong and what is expected.
    assert "a: Input should be a valid integer" in feedback
    assert "bx: Extra inputs are not permitted" in feedback
    assert "Expected: add(a: integer, b?: integer = 0)" in feedback

    second = await router.dispatch(ToolCall(id="2", name="add", arguments={"a": 2, "b": 3}))
    assert second.ok
    assert second.content == "5"
