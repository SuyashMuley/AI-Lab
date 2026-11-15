from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from llmcli.core.errors import (
    InvalidRequestError,
    LLMTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from llmcli.core.events import Done, Event, TextDelta
from llmcli.core.models import LLMRequest, Message
from llmcli.services.llm import LLMService, RetryPolicy
from tests.fakes.provider import FakeProvider


def req() -> LLMRequest:
    return LLMRequest(model="m", messages=[Message.of("user", "hi")])


class Recorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def service(provider: object, rec: Recorder, attempts: int = 3, **kw: float) -> LLMService:
    return LLMService(
        provider,  # type: ignore[arg-type]
        retry=RetryPolicy(max_attempts=attempts),
        sleep=rec,
        rand=lambda: 1.0,
        **kw,  # type: ignore[arg-type]
    )


async def test_retries_transient_error_then_succeeds() -> None:
    fake = FakeProvider(error=RateLimitError("429"), error_calls=2, emit_before_error=False)
    rec = Recorder()
    resp = await service(fake, rec).generate(req())
    assert resp.text == "Hello, world"
    assert len(fake.requests) == 3
    assert rec.delays == [0.5, 1.0]  # exponential, rand=1.0 -> top of jitter range


async def test_retries_are_bounded() -> None:
    fake = FakeProvider(error=ProviderUnavailableError("503"), emit_before_error=False)
    rec = Recorder()
    with pytest.raises(ProviderUnavailableError):
        await service(fake, rec, attempts=3).generate(req())
    assert len(fake.requests) == 3
    assert len(rec.delays) == 2


async def test_invalid_request_is_not_retried() -> None:
    fake = FakeProvider(error=InvalidRequestError("bad"), emit_before_error=False)
    rec = Recorder()
    with pytest.raises(InvalidRequestError):
        await service(fake, rec).generate(req())
    assert len(fake.requests) == 1
    assert rec.delays == []


async def test_no_retry_after_partial_stream_output() -> None:
    fake = FakeProvider(error=ProviderUnavailableError("dropped"), emit_before_error=True)
    rec = Recorder()
    seen: list[str] = []
    with pytest.raises(ProviderUnavailableError):
        async for ev in service(fake, rec).stream(req()):
            if isinstance(ev, TextDelta):
                seen.append(ev.text)
    assert seen == ["Hello", ", ", "world"]
    assert len(fake.requests) == 1  # never replayed: the user already saw output


def test_backoff_is_capped_and_jittered() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0)
    assert policy.delay(1, lambda: 0.0) == 0.5
    assert policy.delay(1, lambda: 1.0) == 1.0
    assert policy.delay(10, lambda: 1.0) == 4.0  # capped


class Silent:
    name = "silent"
    supports_native_schema = False

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        await asyncio.sleep(5)
        yield Done()


async def test_timeout_raises_llm_timeout() -> None:
    with pytest.raises(LLMTimeoutError):
        await LLMService(Silent(), timeout=0.05).generate(req())


async def test_timeout_is_not_retried() -> None:
    rec = Recorder()
    with pytest.raises(LLMTimeoutError):
        await service(Silent(), rec, timeout=0.05).generate(req())
    assert rec.delays == []
