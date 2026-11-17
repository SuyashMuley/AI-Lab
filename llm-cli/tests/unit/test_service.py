from __future__ import annotations

import pytest

from llmcli.core.errors import AuthenticationError, LLMError
from llmcli.core.events import Done, TextDelta
from llmcli.core.models import LLMRequest, Message
from llmcli.services.llm import LLMService
from tests.fakes.provider import FakeProvider


def req() -> LLMRequest:
    return LLMRequest(model="m", messages=[Message.of("user", "hi")])


def test_message_str_shorthand() -> None:
    assert Message(role="user", content="hi").text == "hi"  # type: ignore[arg-type]


async def test_generate_folds_stream() -> None:
    resp = await LLMService(FakeProvider()).generate(req())
    assert resp.text == "Hello, world"
    assert resp.finish_reason == "stop"
    assert resp.usage.input_tokens == 3
    assert resp.usage.latency_ms is not None


async def test_stream_in_order_and_done_terminates() -> None:
    events = [e async for e in LLMService(FakeProvider()).stream(req())]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["Hello", ", ", "world"]
    assert isinstance(events[-1], Done)
    assert events[-1].response is not None


async def test_errors_propagate_normalized() -> None:
    with pytest.raises(AuthenticationError):
        await LLMService(FakeProvider(error=AuthenticationError("bad key"))).generate(req())


async def test_stream_without_done_is_an_error() -> None:
    class Broken:
        name = "broken"
        supports_native_schema = False

        async def stream(self, request: LLMRequest):  # type: ignore[no-untyped-def]
            yield TextDelta(text="x")

    with pytest.raises(LLMError):
        await LLMService(Broken()).generate(req())
