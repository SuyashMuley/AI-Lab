"""Smoke tests against a real local Ollama. Auto-skipped when it isn't available.

uv run pytest -m integration            # only these
uv run pytest -m "not integration"      # everything else
LLMCLI_SMOKE_MODEL=llama3.2 uv run pytest -m integration   # pick the model
"""

from __future__ import annotations

import os

import ollama
import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.core.errors import ModelNotFoundError
from llmcli.core.events import Done, TextDelta
from llmcli.core.models import LLMRequest, Message
from llmcli.providers.ollama import OllamaProvider
from llmcli.services.llm import LLMService

MODEL = os.environ.get("LLMCLI_SMOKE_MODEL", "llama3.2:1b")


def _skip_reason() -> str | None:
    try:
        names = {m.model for m in ollama.Client(timeout=2).list().models}
    except Exception as exc:
        return f"Ollama not reachable: {type(exc).__name__}"
    return None if MODEL in names else f"model {MODEL!r} not pulled (ollama pull {MODEL})"


_reason = _skip_reason()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_reason is not None, reason=_reason or ""),
]


def req(prompt: str, model: str = MODEL) -> LLMRequest:
    return LLMRequest(model=model, messages=[Message.of("user", prompt)], max_tokens=200)


def service() -> LLMService:
    return LLMService(OllamaProvider(), timeout=120)


async def test_generate_returns_text_and_usage() -> None:
    resp = await service().generate(req("Say hello."))
    assert resp.text.strip()
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    assert resp.usage.latency_ms is not None and resp.usage.latency_ms > 0


async def test_stream_yields_deltas_then_done() -> None:
    events = [e async for e in service().stream(req("Count to five."))]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas
    assert isinstance(events[-1], Done)
    assert events[-1].response is not None
    assert events[-1].response.text == "".join(deltas)


async def test_structured_output_matches_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    data, _ = await service().generate_structured(req("What is 2 + 2?"), schema)
    assert isinstance(data["answer"], int)


async def test_unknown_model_is_normalized() -> None:
    with pytest.raises(ModelNotFoundError):
        await service().generate(req("hi", model="definitely-not-a-model"))


def test_cli_end_to_end() -> None:
    r = CliRunner().invoke(cli.app, ["ask", "Say hello.", "-m", MODEL, "--usage"])
    assert r.exit_code == 0
    assert r.stdout.strip()
    assert "tokens:" in r.stderr
