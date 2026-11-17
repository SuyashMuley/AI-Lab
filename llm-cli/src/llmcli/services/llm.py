"""Service layer: the only thing the CLI talks to.

Owns timeouts, bounded retries and structured-output validation.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from llmcli.core.errors import (
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    SchemaValidationError,
)
from llmcli.core.events import Done, Event, TextDelta, ToolCallEvent, fold
from llmcli.core.models import LLMRequest, LLMResponse, Message, Usage
from llmcli.core.protocols import LLMProvider
from llmcli.services import structured


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter. Only transient errors are retried."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_on: tuple[type[LLMError], ...] = (RateLimitError, ProviderUnavailableError)

    def delay(self, attempt: int, rand: Callable[[], float]) -> float:
        cap = min(self.max_delay, self.base_delay * float(2 ** (attempt - 1)))
        return cap * (0.5 + 0.5 * rand())  # jitter within [cap/2, cap]


async def _with_idle_timeout(
    events: AsyncGenerator[Event], seconds: float
) -> AsyncGenerator[Event]:
    """Fail if the provider goes silent for `seconds` between events."""
    async with aclosing(events):
        it = events.__aiter__()
        while True:
            try:
                ev = await asyncio.wait_for(it.__anext__(), seconds)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise LLMTimeoutError(f"No response from provider within {seconds:g}s") from exc
            yield ev


class LLMService:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        schema_retries: int = 2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._provider = provider
        self._retry = retry or RetryPolicy()
        self._timeout = timeout
        self._schema_retries = schema_retries
        self._sleep = sleep
        self._rand = rand

    async def _attempt(self, request: LLMRequest) -> AsyncGenerator[Event]:
        events = self._provider.stream(request)
        if self._timeout is not None:
            events = _with_idle_timeout(events, self._timeout)
        async with aclosing(fold(request.model, events)) as folded:
            async for ev in folded:
                yield ev

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        if request.tools and request.response_schema is not None:
            raise InvalidRequestError(
                "tools and a response schema cannot be combined in one request"
            )
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            emitted = False  # never retry once the caller has seen output
            try:
                async with aclosing(self._attempt(request)) as events:
                    async for ev in events:
                        if isinstance(ev, TextDelta | ToolCallEvent):
                            emitted = True  # a tool call is output too: never replay it
                        elif isinstance(ev, Done) and ev.response is not None:
                            ev.response.usage.latency_ms = (time.perf_counter() - start) * 1000
                        yield ev
                return
            except LLMError as exc:
                retryable = isinstance(exc, self._retry.retry_on)
                if emitted or not retryable or attempt >= self._retry.max_attempts:
                    raise
                await self._sleep(self._retry.delay(attempt, self._rand))

    async def generate(self, request: LLMRequest) -> LLMResponse:
        async for ev in self.stream(request):
            if isinstance(ev, Done) and ev.response is not None:
                return ev.response
        raise AssertionError("unreachable: fold always ends with Done or raises")

    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> tuple[Any, LLMResponse]:
        """Generate JSON conforming to `schema`; on mismatch, retry with the error fed back."""
        structured.check_schema(schema)
        req = request.model_copy(
            update={
                "system": structured.build_system(request.system, schema),
                # Native structured-output support where the provider has it.
                "response_schema": schema if self._provider.supports_native_schema else None,
            }
        )
        start = time.perf_counter()
        total = Usage()
        last_error, last_text = "", ""
        for _ in range(self._schema_retries + 1):
            resp = await self.generate(req)
            total = total + resp.usage
            try:
                data = structured.parse_and_validate(resp.text, schema)
            except structured.SchemaMismatch as exc:
                last_error, last_text = str(exc), resp.text
                req = req.model_copy(
                    update={
                        "messages": [
                            *req.messages,
                            Message.of("assistant", resp.text),
                            Message.of("user", structured.retry_prompt(last_error)),
                        ]
                    }
                )
                continue
            total.latency_ms = (time.perf_counter() - start) * 1000
            return data, resp.model_copy(update={"usage": total})
        raise SchemaValidationError(
            f"Model output did not match the schema after {self._schema_retries + 1} "
            f"attempts. Last error: {last_error}",
            last_output=last_text,
        )
