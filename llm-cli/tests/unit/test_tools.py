"""Tool calling (R0): message parts, stream events, service behaviour, and the Ollama mapping."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import ollama
import pytest
from pydantic import ValidationError

from llmcli.core.errors import (
    InvalidRequestError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from llmcli.core.events import Done, TextDelta, ToolCallEvent, UsageEvent, fold
from llmcli.core.models import (
    LLMRequest,
    LLMResponse,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)
from llmcli.providers.ollama import OllamaProvider, to_ollama_messages, to_ollama_tools
from llmcli.services.llm import LLMService, RetryPolicy
from tests.fakes.provider import FakeProvider

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def call(id_: str, name: str = "get_weather", **arguments: Any) -> ToolCallPart:
    return ToolCallPart(id=id_, name=name, arguments=arguments)


def result(call_id: str, content: str = "sunny", name: str = "get_weather") -> ToolResultPart:
    return ToolResultPart(call_id=call_id, name=name, content=content)


def request(*messages: Message, **kw: Any) -> LLMRequest:
    return LLMRequest(model="m", messages=list(messages), **kw)


# ---- messages and parts ----


def test_text_messages_are_unchanged() -> None:
    assert Message(role="user", content="hi").text == "hi"  # type: ignore[arg-type]
    m = Message.of("user", "hello")
    assert (m.text, m.tool_calls, m.results) == ("hello", [], [])


def test_assistant_message_with_tool_calls() -> None:
    m = Message.assistant("Let me check.", [call("a", city="Oslo"), call("b", city="Rome")])
    assert m.role == "assistant"
    assert m.text == "Let me check."  # tool calls are not text
    assert [c.id for c in m.tool_calls] == ["a", "b"]
    assert Message.assistant(tool_calls=[call("a")]).text == ""
    assert Message.assistant("only text").tool_calls == []


def test_tool_result_message() -> None:
    m = Message.tool_results([result("a", "sunny"), result("b", "rain")])
    assert m.role == "tool"
    assert [r.content for r in m.results] == ["sunny", "rain"]
    assert m.text == ""


@pytest.mark.parametrize(
    "build",
    [
        lambda: Message(role="user", content=[call("a")]),  # a call in a user message
        lambda: Message(role="system", content=[call("a")]),
        lambda: Message(role="assistant", content=[result("a")]),  # a result outside a tool msg
        lambda: Message(role="user", content=[result("a")]),
        lambda: Message(role="tool", content=[TextPart(text="hi")]),  # text in a tool message
        lambda: Message(role="tool", content=[call("a")]),
    ],
)
def test_parts_must_match_the_role(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


def test_json_round_trip_preserves_every_part_type() -> None:
    for original in (
        Message.of("user", "hi"),
        Message.assistant("calling", [call("a", city="Oslo"), call("b")]),
        Message.tool_results(
            [result("a"), ToolResultPart(call_id="b", name="t", content="x", is_error=True)]
        ),
    ):
        restored = Message.model_validate_json(original.model_dump_json())
        assert restored == original


def test_parts_are_discriminated_by_type() -> None:
    m = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "tool_call", "id": "a", "name": "t", "arguments": {"k": 1}},
            ],
        }
    )
    assert isinstance(m.content[0], TextPart)
    assert isinstance(m.content[1], ToolCallPart)
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": [{"text": "no type"}]})


def test_tool_call_part_validation_and_raw_arguments() -> None:
    assert call("a").arguments == {}
    assert (
        ToolCallPart(id="a", name="t", arguments='{"x": 1}').arguments == '{"x": 1}'
    )  # raw text ok
    for bad in ({"id": "", "name": "t"}, {"id": "a", "name": ""}):
        with pytest.raises(ValidationError):
            ToolCallPart.model_validate(bad)


@pytest.mark.parametrize("name", ["a", "get_weather", "get-weather", "A1_b-2", "a" * 64])
def test_tool_spec_accepts_valid_names(name: str) -> None:
    assert ToolSpec(name=name).parameters == {"type": "object", "properties": {}}


@pytest.mark.parametrize("name", ["", "1abc", "has space", "dot.name", "a" * 65])
def test_tool_spec_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name=name)


def test_request_and_response_defaults_are_backwards_compatible() -> None:
    assert request(Message.of("user", "x")).tools is None
    assert LLMResponse(text="x", model="m").tool_calls == []


# ---- the stream ----


async def folded(*events: Any) -> LLMResponse:
    async def gen() -> Any:
        for e in events:
            yield e

    out = [e async for e in fold("m", gen())]
    last = out[-1]
    assert isinstance(last, Done)
    assert last.response is not None
    return last.response


async def test_fold_collects_text_and_tool_calls_in_order() -> None:
    resp = await folded(
        TextDelta(text="Checking"),
        ToolCallEvent(call=call("a", city="Oslo")),
        ToolCallEvent(call=call("b", city="Rome")),
        UsageEvent(usage={"input_tokens": 5, "output_tokens": 2}),  # type: ignore[arg-type]
        Done(finish_reason="stop"),
    )
    assert resp.text == "Checking"
    assert [(c.id, c.arguments) for c in resp.tool_calls] == [
        ("a", {"city": "Oslo"}),
        ("b", {"city": "Rome"}),
    ]
    assert resp.usage.total_tokens == 7


async def test_fold_without_tool_calls_has_an_empty_list() -> None:
    assert (await folded(TextDelta(text="hi"), Done())).tool_calls == []


# ---- the service ----


async def test_generate_returns_tool_calls_and_passes_tools_through() -> None:
    fake = FakeProvider(chunks=[], tool_calls=[[call("a", city="Oslo"), call("b", city="Rome")]])
    resp = await LLMService(fake).generate(request(Message.of("user", "weather?"), tools=[WEATHER]))
    assert [c.id for c in resp.tool_calls] == ["a", "b"]
    assert resp.text == ""
    assert fake.requests[0].tools == [WEATHER]


async def test_a_conversation_with_tool_results_reaches_the_provider_intact() -> None:
    fake = FakeProvider(chunks=["It is sunny."])
    history = [
        Message.of("user", "weather?"),
        Message.assistant(tool_calls=[call("a", city="Oslo")]),
        Message.tool_results([result("a", "sunny")]),
    ]
    resp = await LLMService(fake).generate(request(*history, tools=[WEATHER]))
    assert resp.text == "It is sunny."
    assert [m.role for m in fake.requests[0].messages] == ["user", "assistant", "tool"]


async def test_a_tool_call_counts_as_output_so_it_is_never_replayed() -> None:
    fake = FakeProvider(
        chunks=[],
        tool_calls=[[call("a")]],
        error=ProviderUnavailableError("connection dropped"),
        emit_before_error=True,
    )
    service = LLMService(fake, retry=RetryPolicy(max_attempts=3), sleep=_no_sleep)
    with pytest.raises(ProviderUnavailableError):
        await service.generate(request(Message.of("user", "x"), tools=[WEATHER]))
    assert len(fake.requests) == 1  # a tool call may have side effects: never ask again


async def test_failures_before_any_output_are_still_retried() -> None:
    fake = FakeProvider(
        chunks=[],
        tool_calls=[[call("a")]],
        error=ProviderUnavailableError("down"),
        error_calls=1,
        emit_before_error=False,
    )
    service = LLMService(fake, retry=RetryPolicy(max_attempts=3), sleep=_no_sleep)
    resp = await service.generate(request(Message.of("user", "x"), tools=[WEATHER]))
    assert len(fake.requests) == 2
    assert [c.id for c in resp.tool_calls] == ["a"]


async def _no_sleep(seconds: float) -> None:
    return None


async def test_tools_and_a_response_schema_cannot_be_combined() -> None:
    fake = FakeProvider()
    service = LLMService(fake)
    both = request(Message.of("user", "x"), tools=[WEATHER], response_schema={"type": "object"})
    with pytest.raises(InvalidRequestError, match="cannot be combined"):
        await service.generate(both)
    with pytest.raises(InvalidRequestError, match="cannot be combined"):
        await service.generate_structured(
            request(Message.of("user", "x"), tools=[WEATHER]), {"type": "object"}
        )
    assert fake.requests == []


# ---- Ollama: message and tool mapping (pure) ----


def test_text_messages_map_as_before() -> None:
    mapped = to_ollama_messages(
        request(Message.of("user", "hi"), Message.of("assistant", "hello"), system="Be brief.")
    )
    assert mapped == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_assistant_tool_calls_and_tool_results_map_to_ollama_shapes() -> None:
    mapped = to_ollama_messages(
        request(
            Message.of("user", "weather in Oslo and Rome?"),
            Message.assistant("Checking.", [call("a", city="Oslo"), call("b", city="Rome")]),
            Message.tool_results([result("a", "snow"), result("b", "sun")]),
        )
    )
    assert mapped[1] == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}},
            {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}},
        ],
    }
    assert mapped[2:] == [  # one Ollama message per result, matched by tool name and order
        {"role": "tool", "tool_name": "get_weather", "content": "snow"},
        {"role": "tool", "tool_name": "get_weather", "content": "sun"},
    ]


def test_calls_with_no_text_keep_an_empty_content_string() -> None:
    mapped = to_ollama_messages(request(Message.assistant(tool_calls=[call("a", city="x")])))
    assert mapped[0]["content"] == ""


def test_json_text_arguments_are_parsed_for_ollama() -> None:
    raw = ToolCallPart(id="a", name="t", arguments='{"city": "Oslo"}')
    empty = ToolCallPart(id="b", name="t", arguments="  ")
    mapped = to_ollama_messages(request(Message.assistant(tool_calls=[raw, empty])))
    args = [c["function"]["arguments"] for c in mapped[0]["tool_calls"]]
    assert args == [{"city": "Oslo"}, {}]


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"text"', "42", "null"])
def test_unusable_text_arguments_are_an_invalid_request(bad: str) -> None:
    part = ToolCallPart(id="a", name="t", arguments=bad)
    with pytest.raises(InvalidRequestError, match="not a JSON object"):
        to_ollama_messages(request(Message.assistant(tool_calls=[part])))


def test_tools_map_to_the_ollama_payload() -> None:
    assert to_ollama_tools(None) is None
    assert to_ollama_tools([]) is None  # an empty list is "no tools", never sent
    (payload,) = to_ollama_tools([WEATHER]) or []
    assert payload == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": WEATHER.parameters,
        },
    }


# ---- Ollama: streaming (with a fake client, no server) ----


def chunk(
    content: str = "",
    calls: tuple[tuple[str, dict[str, Any]], ...] = (),
    *,
    done: bool = False,
    prompt: int | None = None,
    evaluated: int | None = None,
) -> Any:
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in calls
    ] or None
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        done=done,
        done_reason="stop" if done else None,
        prompt_eval_count=prompt,
        eval_count=evaluated,
    )


class FakeClient:
    def __init__(self, *chunks: Any, error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def chat(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error

        async def gen() -> Any:
            for c in self.chunks:
                yield c

        return gen()


def provider_with(client: FakeClient) -> OllamaProvider:
    provider = OllamaProvider()
    provider._client = client  # type: ignore[assignment]
    return provider


async def test_streamed_tool_calls_become_events_with_numbered_ids() -> None:
    client = FakeClient(
        chunk("Checking. "),
        chunk(calls=(("get_weather", {"city": "Oslo"}), ("get_weather", {"city": "Rome"}))),
        chunk(calls=(("get_weather", {"city": "Cairo"}),)),  # a later chunk keeps counting
        chunk(done=True, prompt=155, evaluated=14),
    )
    events = [
        e
        async for e in provider_with(client).stream(
            request(Message.of("user", "x"), tools=[WEATHER])
        )
    ]
    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert [(c.id, c.arguments) for c in calls] == [
        ("call_0", {"city": "Oslo"}),
        ("call_1", {"city": "Rome"}),
        ("call_2", {"city": "Cairo"}),
    ]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Checking. "]
    usage = [e for e in events if isinstance(e, UsageEvent)][0]
    assert (usage.usage.input_tokens, usage.usage.output_tokens) == (155, 14)
    assert isinstance(events[-1], Done)


async def test_the_request_reaches_ollama_with_tools_and_mapped_messages() -> None:
    client = FakeClient(chunk(done=True))
    req = request(
        Message.of("user", "x"),
        Message.assistant(tool_calls=[call("a", city="Oslo")]),
        Message.tool_results([result("a", "snow")]),
        tools=[WEATHER],
        system="Be brief.",
        max_tokens=50,
    )
    _ = [e async for e in provider_with(client).stream(req)]
    kwargs = client.kwargs
    assert kwargs["stream"] is True
    assert kwargs["tools"] == to_ollama_tools([WEATHER])
    assert kwargs["messages"] == to_ollama_messages(req)
    assert kwargs["options"] == {"num_predict": 50}
    assert kwargs["format"] is None


async def test_no_tools_means_tools_none_so_text_requests_are_unchanged() -> None:
    client = FakeClient(chunk("hi"), chunk(done=True))
    _ = [e async for e in provider_with(client).stream(request(Message.of("user", "x")))]
    assert client.kwargs["tools"] is None


async def test_ollama_errors_are_still_normalized_for_tool_requests() -> None:
    missing = FakeClient(error=ollama.ResponseError("model 'x' not found", 404))
    with pytest.raises(ModelNotFoundError):
        _ = [
            e
            async for e in provider_with(missing).stream(
                request(Message.of("user", "x"), tools=[WEATHER])
            )
        ]
    unreachable = FakeClient(error=ConnectionError("refused"))
    with pytest.raises(ProviderUnavailableError):
        _ = [e async for e in provider_with(unreachable).stream(request(Message.of("user", "x")))]
