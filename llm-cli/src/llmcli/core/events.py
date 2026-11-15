"""Typed streaming events and a helper that folds a stream into an LLMResponse."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Literal

from pydantic import BaseModel

from llmcli.core.errors import LLMError
from llmcli.core.models import LLMResponse, Usage


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class UsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage


class Done(BaseModel):
    type: Literal["done"] = "done"
    finish_reason: str | None = None
    response: LLMResponse | None = None  # attached by the service layer


class ErrorEvent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    type: Literal["error"] = "error"
    error: LLMError


Event = TextDelta | UsageEvent | Done | ErrorEvent


async def fold(model: str, events: AsyncGenerator[Event]) -> AsyncGenerator[Event]:
    """Pass events through, attaching the folded LLMResponse to the final Done.

    Raises the error carried by an ErrorEvent. A stream that ends without
    Done/ErrorEvent is a provider bug and raises LLMError.
    """
    parts: list[str] = []
    usage = Usage()
    async with aclosing(events):
        async for ev in events:
            if isinstance(ev, TextDelta):
                parts.append(ev.text)
            elif isinstance(ev, UsageEvent):
                usage = ev.usage
            elif isinstance(ev, ErrorEvent):
                raise ev.error
            elif isinstance(ev, Done):
                resp = LLMResponse(
                    text="".join(parts), model=model, finish_reason=ev.finish_reason, usage=usage
                )
                yield Done(finish_reason=ev.finish_reason, response=resp)
                return
            yield ev
    raise LLMError("provider stream ended without a Done event")
