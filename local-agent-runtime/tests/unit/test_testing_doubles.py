import asyncio

import pytest
from llmcli.core.models import Message, ToolCallPart, Usage

from localagent import ModelTurn, TurnRequest
from localagent.testing import FakeClock, ScriptedModel, ScriptExhausted, call, turn


def request(text: str = "hi") -> TurnRequest:
    return TurnRequest(model="m", messages=[Message.of("user", text)])


async def test_steps_play_back_in_order_and_requests_are_recorded() -> None:
    model = ScriptedModel("one", turn("two"), "three")
    assert [(await model.respond(request(str(i)))).text for i in range(3)] == [
        "one",
        "two",
        "three",
    ]
    assert [r.messages[0].text for r in model.requests] == ["0", "1", "2"]
    assert model.remaining == 0


async def test_a_string_is_a_text_turn_and_a_model_turn_passes_through() -> None:
    assert (await ScriptedModel("hello").respond(request())).text == "hello"
    original = ModelTurn(text="x", usage=Usage(input_tokens=1))
    assert await ScriptedModel(original).respond(request()) is original


async def test_an_exception_step_is_raised() -> None:
    with pytest.raises(ValueError, match="boom"):
        await ScriptedModel(ValueError("boom")).respond(request())
    with pytest.raises(asyncio.CancelledError):  # BaseExceptions work too
        await ScriptedModel(asyncio.CancelledError()).respond(request())


async def test_callables_get_the_request_sync_or_async_and_may_return_any_step() -> None:
    echo = ScriptedModel(lambda r: f"echo {r.messages[-1].text}")
    assert (await echo.respond(request("a"))).text == "echo a"

    async def slow(r: TurnRequest) -> str:
        await asyncio.sleep(0)
        return "late"

    assert (await ScriptedModel(slow).respond(request())).text == "late"
    assert (await ScriptedModel(lambda r: turn("x")).respond(request())).text == "x"

    def fails(r: TurnRequest) -> BaseException:
        return RuntimeError("from a callable")

    with pytest.raises(RuntimeError, match="from a callable"):
        await ScriptedModel(fails).respond(request())


async def test_asking_for_more_than_scripted_raises_a_clear_error() -> None:
    model = ScriptedModel("only one")
    await model.respond(request())
    with pytest.raises(ScriptExhausted, match="model call #2"):
        await model.respond(request())
    assert len(model.requests) == 2  # the over-ask is recorded too


def test_turn_and_call_helpers() -> None:
    t = turn("hi", call("search", q="x"), call("read", "id-7", path="a"))
    assert t.text == "hi"
    assert [(c.id, c.name, c.arguments) for c in t.tool_calls] == [
        ("call_search", "search", {"q": "x"}),
        ("id-7", "read", {"path": "a"}),
    ]
    assert t.usage == Usage(input_tokens=10, output_tokens=5)
    assert turn(usage=Usage(input_tokens=1)).usage.input_tokens == 1
    assert isinstance(call("x"), ToolCallPart)


def test_fake_clock() -> None:
    clock = FakeClock(start=10.0)
    assert (clock(), clock()) == (10.0, 10.0)
    clock.advance(2.5)
    assert clock() == 12.5
    ticking = FakeClock(start=0.0, tick=1.0)
    assert [ticking(), ticking(), ticking()] == [0.0, 1.0, 2.0]
