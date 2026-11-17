import asyncio
import dataclasses
import itertools
from typing import Any

import pytest
from llmcli.core.models import Usage
from toolrouter import ReadOnlyPolicy

from localagent import (
    Agent,
    AgentRuntime,
    Budgets,
    ModelFailed,
    ModelResponded,
    RunFinished,
    RunResult,
    RunState,
    ToolResulted,
    TurnRequest,
    replay,
)
from localagent.testing import FakeClock, ScriptedModel, call, turn

AGENT = Agent(model="m")


def runtime(model: ScriptedModel, **kw: Any) -> AgentRuntime:
    ids = itertools.count(1)
    return AgentRuntime(model, new_run_id=lambda: f"run-{next(ids)}", **kw)


async def run(model: ScriptedModel, goal: str = "do the thing", agent: Agent = AGENT) -> RunResult:
    return await runtime(model).run(agent, goal)


def types(result: RunResult) -> list[str]:
    return [e.type for e in result.events]


# ---- the happy path ----


async def test_a_simple_task_completes_with_the_models_answer() -> None:
    result = await run(ScriptedModel("42"))
    assert result.ok
    assert (result.status, result.answer, result.error) == ("completed", "42", None)
    assert (result.run_id, result.steps, result.recoveries) == ("run-1", 1, [])
    assert result.usage == Usage(input_tokens=10, output_tokens=5)


async def test_the_event_log_tells_the_whole_story_in_order() -> None:
    result = await run(ScriptedModel("42"))
    assert types(result) == ["run_started", "model_requested", "model_responded", "run_finished"]
    assert [e.seq for e in result.events] == [0, 1, 2, 3]
    times = [e.at for e in result.events]
    assert times == sorted(times)
    last = result.events[-1]
    assert isinstance(last, RunFinished)
    assert (last.status, last.answer) == ("completed", "42")


async def test_replaying_the_log_reproduces_the_final_state() -> None:
    result = await run(ScriptedModel(turn("42", usage=Usage(input_tokens=7, output_tokens=3))))
    state = replay(result.events)
    assert isinstance(state, RunState)
    assert (state.status, state.answer, state.error) == (result.status, result.answer, result.error)
    assert (state.steps, state.usage, state.elapsed_s) == (
        result.steps,
        result.usage,
        result.elapsed_s,
    )
    assert state.usage == Usage(input_tokens=7, output_tokens=3)


async def test_the_model_receives_the_system_prompt_goal_and_no_tools() -> None:
    model = ScriptedModel("ok")
    await run(model, "summarise this", Agent(model="llama3.2", system="Be terse."))
    (request,) = model.requests
    assert isinstance(request, TurnRequest)
    assert (request.model, request.system) == ("llama3.2", "Be terse.")
    assert [(m.role, m.text) for m in request.messages] == [("user", "summarise this")]
    assert request.tools == []


async def test_the_recorded_response_carries_usage_and_finish_reason() -> None:
    result = await run(ScriptedModel(turn("x", finish_reason="length")))
    response = next(e for e in result.events if isinstance(e, ModelResponded))
    assert (response.step, response.text, response.finish_reason) == (1, "x", "length")


async def test_elapsed_time_comes_from_the_injected_clock() -> None:
    clock = FakeClock()

    def slow(request: TurnRequest) -> str:
        clock.advance(2.5)
        return "done"

    result = await runtime(ScriptedModel(slow), clock=clock).run(AGENT, "g")
    assert result.elapsed_s == 2.5


# ---- failures stay inside the run ----


async def test_a_model_error_fails_the_run_instead_of_raising() -> None:
    result = await run(ScriptedModel(RuntimeError("the model is down")))
    assert not result.ok
    assert (result.status, result.answer) == ("failed", None)
    assert result.error == "RuntimeError: the model is down"
    assert types(result) == ["run_started", "model_requested", "model_failed", "run_finished"]
    failure = next(e for e in result.events if isinstance(e, ModelFailed))
    assert failure.kind == "error"
    assert result.steps == 1


async def test_a_hung_model_times_out_and_is_cancelled() -> None:
    cleaned: list[str] = []

    async def hang(request: TurnRequest) -> str:
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append("model call cancelled")
        return "never"

    agent = Agent(model="m", budgets=Budgets(model_timeout_s=0.05))
    result = await run(ScriptedModel(hang), agent=agent)
    assert result.status == "failed"
    assert result.error == "the model did not answer within 0.05s"
    failure = next(e for e in result.events if isinstance(e, ModelFailed))
    assert failure.kind == "timeout"
    assert cleaned == ["model call cancelled"]
    assert result.elapsed_s < 2


async def test_a_model_clients_own_timeout_is_reported_as_its_error_not_ours() -> None:
    result = await run(ScriptedModel(TimeoutError("socket timed out")))
    failure = next(e for e in result.events if isinstance(e, ModelFailed))
    assert failure.kind == "error"
    assert result.error == "TimeoutError: socket timed out"


async def test_error_messages_are_bounded() -> None:
    result = await run(ScriptedModel(RuntimeError("x" * 5000)))
    assert result.error is not None
    assert len(result.error) <= 300


async def test_an_exhausted_script_is_contained_like_any_model_failure() -> None:
    result = await run(ScriptedModel())
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.startswith("ScriptExhausted")


async def test_an_empty_reply_is_a_failure_not_an_empty_answer() -> None:
    for text in ("", "   "):
        result = await run(ScriptedModel(text))
        assert (result.status, result.answer) == ("failed", None)
        assert result.error == "the model returned an empty reply"


async def test_tool_calls_with_no_tools_registered_get_an_honest_unknown_tool_answer() -> None:
    model = ScriptedModel(turn("", call("search", q="x")), "I could not search, sorry.")
    result = await run(model)
    assert result.ok
    assert (result.steps, result.tool_calls) == (2, 1)
    outcomes = [e for e in result.events if isinstance(e, ToolResulted)]
    assert outcomes[0].outcome.error_code == "unknown_tool"
    assert "No tools are available" in outcomes[0].outcome.text
    assert model.requests[0].tools == []  # nothing was offered, so nothing is claimed to exist


# ---- BaseException is not ours to swallow ----


async def test_cancellation_propagates_and_cancels_the_model_call() -> None:
    started = asyncio.Event()
    cleaned: list[str] = []

    async def hang(request: TurnRequest) -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append("cleanup")
        return "never"

    task = asyncio.create_task(runtime(ScriptedModel(hang)).run(AGENT, "g"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["cleanup"]


async def test_a_cancelled_error_raised_by_the_model_propagates() -> None:
    with pytest.raises(asyncio.CancelledError):
        await runtime(ScriptedModel(asyncio.CancelledError())).run(AGENT, "g")


def test_keyboard_interrupt_propagates() -> None:
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(runtime(ScriptedModel(KeyboardInterrupt())).run(AGENT, "g"))


# ---- many runs, one runtime ----


async def test_concurrent_runs_share_no_state() -> None:
    def echo(request: TurnRequest) -> str:
        return f"answer to: {request.messages[-1].text}"

    model = ScriptedModel(*[echo] * 25)
    rt = AgentRuntime(model)
    results = await asyncio.gather(*(rt.run(AGENT, f"goal-{i}") for i in range(25)))
    assert [r.answer for r in results] == [f"answer to: goal-{i}" for i in range(25)]
    assert len({r.run_id for r in results}) == 25
    assert all(types(r) == types(results[0]) for r in results)


async def test_default_run_ids_are_short_hex_and_unique() -> None:
    rt = AgentRuntime(ScriptedModel(*["x"] * 50))
    ids = [(await rt.run(AGENT, "g")).run_id for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(len(i) == 12 and int(i, 16) >= 0 for i in ids)


# ---- config and result types ----


def test_agent_defaults_and_immutability() -> None:
    agent = Agent(model="m")
    assert agent.system is None  # None means "the default prompt"
    assert agent.budgets == Budgets()
    assert len(agent.tools) == 0
    assert isinstance(agent.policy, ReadOnlyPolicy)  # safe by default
    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.model = "other"  # type: ignore[misc]


async def test_a_result_round_trips_through_json_with_its_log() -> None:
    result = await run(ScriptedModel(turn("hi", call("t"))))
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert [type(e).__name__ for e in restored.events] == [type(e).__name__ for e in result.events]


async def test_the_log_is_kept_out_of_the_repr() -> None:
    result = await run(ScriptedModel("hi"))
    assert "events" not in repr(result)
    assert "run_started" in result.model_dump_json()


def test_a_result_cannot_be_built_from_an_unfinished_run() -> None:
    from localagent import RunStarted, reduce

    state = reduce(None, RunStarted(run_id="r", model="m", system="S", goal="g", budgets=Budgets()))
    with pytest.raises(ValueError, match="unfinished"):
        RunResult.from_state(state, [])
