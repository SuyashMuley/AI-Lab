"""Ollama adapter (official `ollama` SDK -> internal events)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import cast

import httpx
import ollama

from llmcli.core.errors import (
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from llmcli.core.events import Done, Event, TextDelta, UsageEvent
from llmcli.core.models import LLMRequest, Usage


def _normalize(exc: Exception) -> LLMError:
    if isinstance(exc, ollama.ResponseError):
        code = exc.status_code
        if code == 404:
            err: LLMError = ModelNotFoundError(str(exc))
        elif code in (401, 403):
            err = AuthenticationError(str(exc))
        elif code == 429:
            err = RateLimitError(str(exc))
        elif code >= 500:
            err = ProviderUnavailableError(str(exc))
        else:
            err = InvalidRequestError(str(exc))
    elif isinstance(exc, httpx.TimeoutException):
        err = LLMTimeoutError(str(exc))
    elif isinstance(exc, ConnectionError | httpx.TransportError):
        err = ProviderUnavailableError(f"Cannot reach Ollama: {exc}")
    else:
        err = LLMError(str(exc))
    err.__cause__ = exc
    return err


class OllamaProvider:
    name = "ollama"
    supports_native_schema = True

    def __init__(self, host: str | None = None) -> None:
        self._client = ollama.AsyncClient(host=host)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        messages = [{"role": m.role, "content": m.text} for m in request.messages]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        options: dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        try:
            chunks = await self._client.chat(
                model=request.model,
                messages=messages,
                stream=True,
                options=options or None,
                format=request.response_schema,
            )
            async with aclosing(cast("AsyncGenerator[ollama.ChatResponse]", chunks)):
                async for chunk in chunks:
                    if chunk.message.content:
                        yield TextDelta(text=chunk.message.content)
                    if chunk.done:
                        yield UsageEvent(
                            usage=Usage(
                                input_tokens=chunk.prompt_eval_count or 0,
                                output_tokens=chunk.eval_count or 0,
                            )
                        )
                        yield Done(finish_reason=chunk.done_reason)
        except LLMError:
            raise
        except Exception as exc:
            raise _normalize(exc) from exc
