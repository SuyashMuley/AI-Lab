"""LlmcliModel and ollama_client, against a fake llmcli provider (no network)."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from llmcli.core.errors import (
    LLMTimeoutError,
    ModelNotFoundError,
    ProviderUnavailableError,
)
from llmcli.core.events import Done, Event, TextDelta, ToolCallEvent, UsageEvent
from llmcli.core.models import LLMRequest, Message, ToolCallPart, ToolResultPart, ToolSpec, Usage
from llmcli.providers.ollama import OllamaProvider
from toolrouter import ToolRegistry

from localagent import (
    DEFAULT_NUM_CTX,
    Agent,
    AgentRuntime,
    Budgets,
    LlmcliModel,
    ModelClient,
    TurnRequest,
    ollama_client,
)
from tests.fakes import tools as fake_tools


class FakeProvider:
    """A scripted llmcli provider: each script is a list of events, or an exception to raise."""

    name = "fake"
    supports_native_schema = False

    def __init__(self, *scripts: list[Event] | BaseException) -> None:
        self.scripts = list(scripts)
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
        self.requests.append(request)
        script = self.scripts.pop(0)
        if isinstance(script, BaseException):
            raise script
        for event in script:
            yield event


def reply(
    *texts: str, calls: tuple[ToolCallPart, ...] = (), tokens: tuple[int, int] = (10, 5)
) -> list[Event]:
    events: list[Event] = [TextDelta(text=t) for t in texts]
    events += [ToolCallEvent(call=c) for c in calls]
    events += [UsageEvent(usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1]))]
    return [*events, Done(finish_reason="stop")]


def asked(text: str = "hi", **kw: Any) -> TurnRequest:
    return TurnRequest(model="m", messages=[Message.of("user", text)], **kw)


# ---- the adapter ----


async def test_a_text_reply_becomes_a_model_turn() -> None:
    client = LlmcliModel(FakeProvider(reply("Hel", "lo", tokens=(7, 3))))
    turn = await client.respond(asked())
    assert (turn.text, turn.tool_calls, turn.finish_reason) == ("Hello", [], "stop")
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (7, 3)
    assert turn.usage.latency_ms is not None  # llmcli measures it


async def test_tool_calls_come_through_whole() -> None:
    call = ToolCallPart(id="call_0", name="add", arguments={"a": 1, "b": 2})
    turn = await LlmcliModel(FakeProvider(reply(calls=(call,)))).respond(asked())
    assert turn.text == ""
    assert turn.tool_calls == [call]


async def test_the_request_is_mapped_faithfully() -> None:
    provider = FakeProvider(reply("ok"))
    spec = ToolSpec(name="add", description="Add.", parameters={"type": "object", "properties": {}})
    history = [
        Message.of("user", "add 1 and 2"),
        Message.assistant("", [ToolCallPart(id="call_0", name="add", arguments={"a": 1})]),
        Message.tool_results([ToolResultPart(call_id="call_0", name="add", content="3")]),
    ]
    request = TurnRequest(model="llama3.2", messages=history, system="Be brief.", tools=[spec])
    await LlmcliModel(provider).respond(request)
    (sent,) = provider.requests
    assert (sent.model, sent.system) == ("llama3.2", "Be brief.")
    assert sent.messages == history
    assert sent.tools == [spec]


async def test_no_tools_means_none_not_an_empty_list() -> None:
    provider = FakeProvider(reply("ok"))
    await LlmcliModel(provider).respond(asked())
    assert provider.requests[0].tools is None


async def test_defaults_apply_and_a_requests_own_settings_win() -> None:
    provider = FakeProvider(reply("a"), reply("b"))
    client = LlmcliModel(provider, temperature=0.2, max_tokens=100)
    await client.respond(asked())
    await client.respond(asked(temperature=0.9, max_tokens=7))
    assert [(r.temperature, r.max_tokens) for r in provider.requests] == [(0.2, 100), (0.9, 7)]


def test_it_satisfies_the_model_client_protocol() -> None:
    client: ModelClient = LlmcliModel(FakeProvider())
    assert client is not None


# ---- streaming ----


async def test_text_deltas_arrive_in_order_and_match_the_final_text() -> None:
    seen: list[str] = []
    turn = await LlmcliModel(FakeProvider(reply("The ", "answer ", "is 42"))).respond(
        asked(), on_text=seen.append
    )
    assert seen == ["The ", "answer ", "is 42"]
    assert "".join(seen) == turn.text


async def test_a_tool_only_turn_streams_no_text() -> None:
    seen: list[str] = []
    call = ToolCallPart(id="c", name="add")
    await LlmcliModel(FakeProvider(reply(calls=(call,)))).respond(asked(), on_text=seen.append)
    assert seen == []


async def test_a_failing_observer_never_fails_the_call() -> None:
    def broken(text: str) -> None:
        raise RuntimeError("the UI crashed")

    turn = await LlmcliModel(FakeProvider(reply("a", "b"))).respond(asked(), on_text=broken)
    assert turn.text == "ab"


# ---- failures: one attempt, errors surface ----


async def test_there_are_no_hidden_retries() -> None:
    provider = FakeProvider(ProviderUnavailableError("connection refused"), reply("never reached"))
    with pytest.raises(ProviderUnavailableError):
        await LlmcliModel(provider).respond(asked())
    assert len(provider.requests) == 1  # R4 retries visibly, in the run log; the adapter never does


@pytest.mark.parametrize("error", [ModelNotFoundError("no such model"), LLMTimeoutError("slow")])
async def test_llmcli_errors_surface_unchanged(error: Exception) -> None:
    with pytest.raises(type(error), match=str(error)):
        await LlmcliModel(FakeProvider(error)).respond(asked())


async def test_a_stalled_stream_hits_the_idle_timeout() -> None:
    class Stalls:
        name = "stalls"
        supports_native_schema = False

        async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
            yield TextDelta(text="start")
            await asyncio.sleep(30)

    with pytest.raises(LLMTimeoutError):
        await LlmcliModel(Stalls(), idle_timeout_s=0.05).respond(asked())


# ---- inside the runtime ----


def agent(**kw: Any) -> Agent:
    return Agent(model="m", tools=ToolRegistry([fake_tools.add]), **kw)


async def test_a_full_react_run_through_the_real_adapter() -> None:
    fake_tools.CALLS.clear()
    provider = FakeProvider(
        reply(
            "Let me add. ",
            calls=(ToolCallPart(id="call_0", name="add", arguments={"a": 2, "b": 3}),),
        ),
        reply("The sum is 5."),
    )
    result = await AgentRuntime(LlmcliModel(provider)).run(agent(), "what is 2 + 3?")
    assert result.ok
    assert result.answer == "The sum is 5."
    assert (result.steps, result.tool_calls) == (2, 1)
    assert fake_tools.CALLS == ["add"]
    assert result.usage.input_tokens == 20
    second = provider.requests[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "tool"]
    assert second.messages[2].results[0].content == "5"
    assert [t.name for t in second.tools or []] == ["add"]


async def test_provider_failures_become_failed_runs_not_exceptions() -> None:
    for error, expected in (
        (ModelNotFoundError("model 'x' not found"), "ModelNotFoundError: model 'x' not found"),
        (
            ProviderUnavailableError("Cannot reach Ollama"),
            "ProviderUnavailableError: Cannot reach Ollama",
        ),
    ):
        result = await AgentRuntime(LlmcliModel(FakeProvider(error))).run(agent(), "g")
        assert (result.status, result.error) == ("failed", expected)


async def test_a_hung_provider_is_cut_off_and_its_stream_closed() -> None:
    closed: list[str] = []

    class Hangs:
        name = "hangs"
        supports_native_schema = False

        async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
            try:
                yield TextDelta(text="partial")
                await asyncio.sleep(30)
            finally:
                closed.append("stream closed")

    a = agent(budgets=Budgets(model_timeout_s=0.1))
    result = await AgentRuntime(LlmcliModel(Hangs())).run(a, "g")
    assert result.status == "failed"
    assert result.error == "the model did not answer within 0.1s"
    assert closed == ["stream closed"]  # the SDK stream is closed, not leaked


async def test_cancelling_a_run_closes_the_provider_stream() -> None:
    started = asyncio.Event()
    closed: list[str] = []

    class Slow:
        name = "slow"
        supports_native_schema = False

        async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
            try:
                started.set()
                await asyncio.sleep(30)
                yield Done()
            finally:
                closed.append("closed")

    task = asyncio.create_task(AgentRuntime(LlmcliModel(Slow())).run(agent(), "g"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == ["closed"]


# ---- ollama_client ----


def options_of(client: LlmcliModel) -> dict[str, Any]:
    assert isinstance(client.provider, OllamaProvider)
    return client.provider.options


def test_ollama_client_always_sets_a_context_window_by_default() -> None:
    assert options_of(ollama_client()) == {"num_ctx": DEFAULT_NUM_CTX}
    assert DEFAULT_NUM_CTX > 4096  # larger than Ollama's silently-truncating default


def test_ollama_client_options_can_be_tuned() -> None:
    assert options_of(ollama_client(num_ctx=16384)) == {"num_ctx": 16384}
    assert options_of(ollama_client(num_ctx=None)) == {}  # explicitly opting into Ollama's default
    merged = options_of(ollama_client(options={"seed": 7, "num_ctx": 1}, num_ctx=2048))
    assert merged == {"seed": 7, "num_ctx": 2048}  # the explicit num_ctx argument wins


def test_ollama_client_does_not_share_the_callers_options_dict() -> None:
    mine = {"seed": 1}
    client = ollama_client(options=mine)
    mine["seed"] = 99
    assert options_of(client)["seed"] == 1
