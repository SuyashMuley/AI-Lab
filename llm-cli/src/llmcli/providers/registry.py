from __future__ import annotations

from llmcli.config.settings import Settings
from llmcli.core.errors import InvalidRequestError
from llmcli.core.protocols import LLMProvider


def get_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "ollama":
        from llmcli.providers.ollama import OllamaProvider

        return OllamaProvider(host=settings.ollama_host)
    raise InvalidRequestError(f"Unknown provider: {name!r}. Available: ollama")
