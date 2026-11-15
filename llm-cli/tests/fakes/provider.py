from __future__ import annotations

from collections.abc import AsyncGenerator

from llmcli.core.errors import LLMError
from llmcli.core.events import Done, Event, TextDelta, UsageEvent
from llmcli.core.models import LLMRequest, Usage


class FakeProvider:
    """Deterministic provider: scripted chunks, optional per-call failures.

    - `chunks`: output for every call (default "Hello, world").
    - `responses`: per-call outputs (each a list of chunks); the last repeats.
    - `error` / `error_calls`: raise on the first N calls (None = every call).
    - `emit_before_error`: stream the chunks before raising (partial output).
    """

    name = "fake"

    def __init__(
        self,
        chunks: list[str] | None = None,
        error: LLMError | None = None,
        usage: Usage | None = None,
        *,
        responses: list[list[str]] | None = None,
        error_calls: int | None = None,
        emit_before_error: bool = True,
        native_schema: bool = True,
    ) -> None:
        self.chunks = chunks if chunks is not None else ["Hello", ", ", "world"]
        self.error = error
        self.usage = usage or Usage(input_tokens=3, output_tokens=len(self.chunks))
        self.responses = responses
        self.error_calls = error_calls
        self.emit_before_error = emit_before_error
        self.supports_native_schema = native_schema
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        call = len(self.requests)
        self.requests.append(request)
        chunks = (
            self.responses[min(call, len(self.responses) - 1)] if self.responses else self.chunks
        )
        failing = self.error is not None and (self.error_calls is None or call < self.error_calls)
        if not failing or self.emit_before_error:
            for c in chunks:
                yield TextDelta(text=c)
        if failing and self.error is not None:
            raise self.error
        yield UsageEvent(usage=self.usage)
        yield Done(finish_reason="stop")
