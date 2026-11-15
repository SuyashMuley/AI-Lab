from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from llmcli.core.events import Event
from llmcli.core.models import LLMRequest


class LLMProvider(Protocol):
    """Adapter between one provider SDK and the internal types.

    Only `stream` is required; non-streaming generation is a fold over it.
    Implementations raise normalized LLMError subclasses, never SDK errors.
    """

    name: str
    supports_native_schema: bool

    def stream(self, request: LLMRequest) -> AsyncGenerator[Event]: ...
