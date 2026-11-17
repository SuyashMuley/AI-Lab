"""Real-model tool calling through llmcli's own layers. Auto-skipped without Ollama + the model.

uv run pytest -m integration tests/integration/test_smoke_tools.py
LLMCLI_SMOKE_TOOL_MODEL=qwen2.5 uv run pytest -m integration      # pick a model
"""

from __future__ import annotations

import os

import ollama
import pytest

from llmcli.core.models import LLMRequest, Message, ToolResultPart, ToolSpec
from llmcli.providers.ollama import OllamaProvider
from llmcli.services.llm import LLMService

MODEL = os.environ.get("LLMCLI_SMOKE_TOOL_MODEL", "llama3.2")


def _skip_reason() -> str | None:
    try:
        names = {m.model for m in ollama.Client(timeout=2).list().models}
    except Exception as exc:
        return f"Ollama not reachable: {type(exc).__name__}"
    if any(n and n.split(":")[0] == MODEL.split(":")[0] for n in names):
        return None
    return f"model {MODEL!r} not pulled (ollama pull {MODEL})"


_reason = _skip_reason()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_reason is not None, reason=_reason or ""),
]

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def service() -> LLMService:
    return LLMService(OllamaProvider(), timeout=120)


async def test_tool_calling_round_trip() -> None:
    first = LLMRequest(
        model=MODEL,
        messages=[Message.of("user", "What is the weather in Oslo right now?")],
        tools=[WEATHER],
        temperature=0,
    )
    asked = await service().generate(first)
    assert asked.tool_calls, "a tool-capable model should call the tool"
    call = asked.tool_calls[0]
    assert call.name == "get_weather"
    assert isinstance(call.arguments, dict)
    assert "oslo" in str(call.arguments.get("city", "")).lower()
    assert asked.usage.input_tokens > 0

    result = ToolResultPart(
        call_id=call.id, name=call.name, content="4 degrees Celsius, light snow"
    )
    followup = first.model_copy(
        update={
            "messages": [
                *first.messages,
                Message.assistant(asked.text, asked.tool_calls),
                Message.tool_results([result]),
            ]
        }
    )
    answered = await service().generate(followup)
    assert not answered.tool_calls  # it used the result instead of asking again
    assert answered.text.strip()


async def test_text_requests_are_unaffected_by_tool_support() -> None:
    plain = LLMRequest(model=MODEL, messages=[Message.of("user", "Say hello.")], max_tokens=60)
    resp = await service().generate(plain)
    assert resp.text.strip()
    assert resp.tool_calls == []


async def test_a_model_that_does_not_need_a_tool_just_answers() -> None:
    req = LLMRequest(
        model=MODEL,
        messages=[Message.of("user", "What is 2 + 2? Answer with just the number.")],
        tools=[WEATHER],
        temperature=0,
    )
    resp = await service().generate(req)
    assert "4" in resp.text or resp.tool_calls  # either answers directly or asks; never crashes
