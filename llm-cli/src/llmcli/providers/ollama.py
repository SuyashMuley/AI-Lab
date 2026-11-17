"""Ollama adapter (official `ollama` SDK -> internal events)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, cast

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
from llmcli.core.events import Done, Event, TextDelta, ToolCallEvent, UsageEvent
from llmcli.core.models import LLMRequest, ToolCallPart, ToolSpec, Usage


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


def _arguments(call: ToolCallPart) -> dict[str, Any]:
    """Ollama wants tool-call arguments as an object; accept JSON text from other sources."""
    if isinstance(call.arguments, dict):
        return call.arguments
    if not call.arguments.strip():
        return {}
    try:
        parsed = json.loads(call.arguments)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        raise InvalidRequestError(f"arguments of tool call {call.name!r} are not a JSON object")
    return parsed


def to_ollama_messages(request: LLMRequest) -> list[dict[str, Any]]:
    """Internal messages -> Ollama chat messages (a tool message becomes one per result)."""
    out: list[dict[str, Any]] = []
    if request.system:
        out.append({"role": "system", "content": request.system})
    for m in request.messages:
        if m.role == "tool":
            out.extend(
                {"role": "tool", "tool_name": r.name, "content": r.content} for r in m.results
            )
        elif m.role == "assistant" and m.tool_calls:
            calls = [
                {"function": {"name": c.name, "arguments": _arguments(c)}} for c in m.tool_calls
            ]
            out.append({"role": "assistant", "content": m.text, "tool_calls": calls})
        else:
            out.append({"role": m.role, "content": m.text})
    return out


def to_ollama_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OllamaProvider:
    name = "ollama"
    supports_native_schema = True

    def __init__(
        self, host: str | None = None, *, options: Mapping[str, Any] | None = None
    ) -> None:
        """`options` are Ollama model options applied to every request (e.g. `{"num_ctx": 8192}`).

        Set `num_ctx` for anything with long conversations: Ollama's default context is small
        (4096 tokens) and it silently drops what does not fit instead of raising an error.
        Per-request settings (`temperature`, `max_tokens`) override these.
        """
        self._client = ollama.AsyncClient(host=host)
        self.options: dict[str, Any] = dict(options or {})

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        messages = to_ollama_messages(request)
        options: dict[str, Any] = dict(self.options)
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
                tools=to_ollama_tools(request.tools),
            )
            n_calls = 0
            async with aclosing(cast("AsyncGenerator[ollama.ChatResponse]", chunks)):
                async for chunk in chunks:
                    if chunk.message.content:
                        yield TextDelta(text=chunk.message.content)
                    # Ollama sends each tool call whole, with no id: number them per response.
                    for call in chunk.message.tool_calls or []:
                        yield ToolCallEvent(
                            call=ToolCallPart(
                                id=f"call_{n_calls}",
                                name=call.function.name,
                                arguments=dict(call.function.arguments),
                            )
                        )
                        n_calls += 1
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
