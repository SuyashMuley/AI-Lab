"""Provider-level Ollama options (e.g. num_ctx) and how per-request settings override them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llmcli.core.models import LLMRequest, Message
from llmcli.providers.ollama import OllamaProvider


class FakeClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def chat(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs

        async def gen() -> Any:
            yield SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=None),
                done=True,
                done_reason="stop",
                prompt_eval_count=1,
                eval_count=1,
            )

        return gen()


async def sent_options(provider: OllamaProvider, **request: Any) -> Any:
    client = FakeClient()
    provider._client = client  # type: ignore[assignment]
    req = LLMRequest(model="m", messages=[Message.of("user", "hi")], **request)
    _ = [e async for e in provider.stream(req)]
    return client.kwargs["options"]


async def test_no_options_sends_none_exactly_as_before() -> None:
    assert await sent_options(OllamaProvider()) is None


async def test_provider_options_are_sent_with_every_request() -> None:
    options = await sent_options(OllamaProvider(options={"num_ctx": 8192, "seed": 7}))
    assert options == {"num_ctx": 8192, "seed": 7}


async def test_per_request_settings_override_and_merge_with_provider_options() -> None:
    provider = OllamaProvider(options={"num_ctx": 8192, "temperature": 0.9})
    options = await sent_options(provider, temperature=0.1, max_tokens=50)
    assert options == {"num_ctx": 8192, "temperature": 0.1, "num_predict": 50}


async def test_the_providers_options_are_copied_not_shared() -> None:
    shared = {"num_ctx": 4096}
    provider = OllamaProvider(options=shared)
    shared["num_ctx"] = 1
    await sent_options(provider, max_tokens=5)
    assert provider.options == {"num_ctx": 4096}  # neither the caller's dict nor a request edits it
