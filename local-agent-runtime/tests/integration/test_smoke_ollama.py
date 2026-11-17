"""Smoke tests against a real local Ollama model. Auto-skipped when it isn't available.

    uv run pytest -m integration                    # only these
    uv run pytest -m "not integration"              # everything else
    LOCALAGENT_SMOKE_MODEL=qwen2.5 uv run pytest -m integration

A real model is not deterministic, so these assert what must ALWAYS hold (safety, clean failures,
the context window really being applied), plus the one thing a tool-capable model should reliably
do (call the add tool for arithmetic), using temperature 0 and a fixed seed.
"""

import os
import time

import ollama
import pytest
from llmcli.core.models import Message
from toolrouter import ToolRegistry

from localagent import (
    Agent,
    AgentRuntime,
    Budgets,
    ToolResulted,
    TurnRequest,
    ollama_client,
)
from tests.fakes import tools as fake_tools

MODEL = os.environ.get("LOCALAGENT_SMOKE_MODEL", "llama3.2")


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


def client(**kw: object):  # type: ignore[no-untyped-def]
    return ollama_client(temperature=0, options={"seed": 7}, max_tokens=300, **kw)  # type: ignore[arg-type]


def agent(*tools: object, **kw: object) -> Agent:
    registry = ToolRegistry(list(tools))  # type: ignore[arg-type]
    return Agent(model=MODEL, tools=registry, **kw)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def reset_spy() -> None:
    fake_tools.CALLS.clear()


async def test_a_plain_question_gets_an_answer_and_usage() -> None:
    result = await AgentRuntime(client()).run(
        Agent(model=MODEL), "Reply with exactly one word: pong."
    )
    assert result.ok, result.error
    assert result.answer and result.answer.strip()
    assert result.steps == 1
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


async def test_the_model_uses_a_tool_and_the_loop_finishes() -> None:
    # Lenient validation: llama3.2 often sends integers as strings ("17"), and measured runs show
    # it then fails to repair the call properly. With coerce the tool task is reliable.
    a = agent(fake_tools.add, fake_tools.echo, coerce=True)
    result = await AgentRuntime(client()).run(a, "What is 17 + 25? Use the add tool.")
    assert result.tool_calls >= 1, "a tool-capable model should call the tool"
    assert "add" in fake_tools.CALLS
    observed = [e.outcome.text for e in result.events if isinstance(e, ToolResulted)]
    assert "42" in observed or "42" in (result.answer or "")
    assert result.ok, result.error
    assert result.steps <= 6


async def test_strict_validation_reports_a_malformed_call_honestly() -> None:
    # Strict is the default. Whatever the model does, a rejected call is reported with the repair
    # hint, the handler runs only for valid calls, and the run ends inside its budget.
    a = agent(fake_tools.add, fake_tools.echo)
    result = await AgentRuntime(client()).run(a, "What is 17 + 25? Use the add tool.")
    outcomes = [e.outcome for e in result.events if isinstance(e, ToolResulted)]
    for outcome in outcomes:
        if outcome.error_code == "arguments_invalid":
            assert "Expected: add(a: integer" in outcome.text
    assert ("add" in fake_tools.CALLS) == any(o.ok for o in outcomes)
    assert result.steps <= 12
    assert result.status in {"completed", "failed"}


async def test_text_streams_in_pieces_that_add_up_to_the_final_text() -> None:
    seen: list[str] = []
    turn = await client().respond(
        TurnRequest(model=MODEL, messages=[Message.of("user", "Count from one to five.")]),
        on_text=seen.append,
    )
    assert len(seen) > 1  # really streamed, not delivered in one lump
    assert "".join(seen) == turn.text


async def test_the_context_window_setting_really_reaches_ollama() -> None:
    """Characterizes Ollama's silent truncation: the same prompt, two window sizes."""
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 600 + "\nSay OK."
    request = TurnRequest(model=MODEL, messages=[Message.of("user", long_prompt)], max_tokens=5)

    big = await ollama_client(num_ctx=8192, temperature=0, options={"seed": 7}).respond(request)
    small = await ollama_client(num_ctx=2048, temperature=0, options={"seed": 7}).respond(request)

    assert big.usage.input_tokens > 4500  # the whole prompt was read (it is ~6000 tokens)
    assert small.usage.input_tokens <= 2048  # ...but a small window silently drops the rest
    assert small.usage.input_tokens < big.usage.input_tokens


async def test_read_only_holds_with_a_real_model_whatever_it_tries() -> None:
    a = agent(fake_tools.save, fake_tools.echo)
    result = await AgentRuntime(client()).run(a, "Save a file named report using the save tool.")
    assert "save" not in fake_tools.CALLS  # the WRITE tool's handler never ran
    assert result.status in {"completed", "failed"}  # and the run ended cleanly either way


async def test_an_unknown_model_fails_the_run_cleanly() -> None:
    result = await AgentRuntime(client()).run(Agent(model="definitely-not-a-model-xyz"), "hi")
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.startswith("ModelNotFoundError")


async def test_an_unreachable_server_fails_fast_not_forever() -> None:
    dead = ollama_client(host="http://localhost:1")
    start = time.perf_counter()
    result = await AgentRuntime(dead).run(Agent(model=MODEL), "hi")
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.startswith("ProviderUnavailableError")
    assert time.perf_counter() - start < 10


async def test_a_tiny_reply_timeout_fails_the_run_not_the_process() -> None:
    a = Agent(model=MODEL, budgets=Budgets(model_timeout_s=0.001))
    result = await AgentRuntime(client()).run(a, "Write a long story about the sea.")
    assert result.status == "failed"
    assert result.error == "the model did not answer within 0.001s"
