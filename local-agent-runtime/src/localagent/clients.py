"""Model clients: `ModelClient` implementations backed by `llmcli`.

`LlmcliModel` adapts any `llmcli` provider; `ollama_client` builds one for a local Ollama.

No hidden retries: the adapter makes exactly one attempt per call. Retrying belongs to the
runtime, where each retry is a visible, bounded event in the run log (R4); a retry buried here
would multiply with that one and leave no trace.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import aclosing
from typing import Any

from llmcli.core.events import Done, TextDelta
from llmcli.core.models import LLMRequest
from llmcli.core.protocols import LLMProvider
from llmcli.services.llm import LLMService, RetryPolicy

from localagent.models import ModelTurn, TurnRequest

_log = logging.getLogger("localagent")

#: Ollama's own default context is 4096 tokens and it silently drops what does not fit, which in
#: an agent loop quietly deletes the system prompt and early observations. We always set one.
DEFAULT_NUM_CTX = 8192


class LlmcliModel:
    """A `ModelClient` over an `llmcli` provider (one attempt per call, no hidden retries).

    Failures surface as `llmcli` errors (`ModelNotFoundError`, `ProviderUnavailableError`,
    `LLMTimeoutError`, ...); the runtime contains them as failed model calls.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        idle_timeout_s: float | None = None,
    ) -> None:
        """`idle_timeout_s` aborts a stream that goes silent that long between events; the
        runtime's per-reply `model_timeout_s` is the overall bound and is usually enough."""
        self.provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._service = LLMService(
            provider, retry=RetryPolicy(max_attempts=1), timeout=idle_timeout_s
        )

    async def respond(
        self, request: TurnRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelTurn:
        """One assistant turn. `on_text` receives text deltas as they stream in (for live display;
        deltas are not part of the run log). An exception in `on_text` never fails the call."""
        llm_request = LLMRequest(
            model=request.model,
            messages=request.messages,
            system=request.system,
            tools=request.tools or None,  # an empty list means "no tools", never sent
            temperature=request.temperature
            if request.temperature is not None
            else self._temperature,
            max_tokens=request.max_tokens if request.max_tokens is not None else self._max_tokens,
        )
        response = None
        async with aclosing(self._service.stream(llm_request)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    if on_text is not None:
                        try:
                            on_text(event.text)
                        except Exception:  # an observer's bug must not fail the model call
                            _log.debug("on_text callback raised", exc_info=True)
                elif isinstance(event, Done):
                    response = event.response
        assert response is not None  # llmcli guarantees a Done carrying the folded response
        return ModelTurn(
            text=response.text,
            tool_calls=response.tool_calls,
            usage=response.usage,
            finish_reason=response.finish_reason,
        )


def ollama_client(
    *,
    host: str | None = None,
    num_ctx: int | None = DEFAULT_NUM_CTX,
    options: Mapping[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    idle_timeout_s: float | None = None,
) -> LlmcliModel:
    """A model client for a local Ollama server.

    `num_ctx` sets the context window in tokens (default 8192). Pass `None` only if you want
    Ollama's small default, which silently truncates long conversations. `options` are further
    Ollama model options (e.g. `{"seed": 7}`); `temperature`/`max_tokens` apply to every call
    unless a request sets its own.
    """
    from llmcli.providers.ollama import OllamaProvider

    merged: dict[str, Any] = dict(options or {})
    if num_ctx is not None:
        merged["num_ctx"] = num_ctx
    return LlmcliModel(
        OllamaProvider(host, options=merged),
        temperature=temperature,
        max_tokens=max_tokens,
        idle_timeout_s=idle_timeout_s,
    )
