"""The ReAct loop end to end: a scripted model driving real toolrouter tools."""

import asyncio
import dataclasses
import itertools
from typing import Any

import pytest
from llmcli.core.models import Usage
from toolrouter import AllowAll, ToolRegistry

from localagent import (
    DEFAULT_SYSTEM,
    TOOL_GUIDANCE,
    Agent,
    AgentRuntime,
    Budgets,
    ModelRequested,
    ModelResponded,
    RunResult,
    RunStarted,
    ToolResulted,
    ToolsDispatching,
    TurnRequest,
    replay,
    tool_specs,
)
from localagent.testing import ScriptedModel, call, turn
from tests.fakes import tools as fake_tools
from tests.fakes.tools import registry


@pytest.fixture(autouse=True)
def reset_spy() -> None:
    fake_tools.CALLS.clear()


def agent(*tool_list: object, **kw: Any) -> Agent:
    return Agent(model="m", tools=registry(*tool_list), **kw)


async def run(model: ScriptedModel, a: Agent | None = None, goal: str = "goal") -> RunResult:
    ids = itertools.count(1)
    return await AgentRuntime(model, new_run_id=lambda: f"run-{next(ids)}").run(a or agent(), goal)


def types(result: RunResult) -> list[str]:
    return [e.type for e in result.events]


def tool_message_contents(request: TurnRequest) -> list[str]:
    return [r.content for m in request.messages if m.role == "tool" for r in m.results]


# ---- think, act, observe ----


async def test_a_tool_call_then_an_answer() -> None:
    model = ScriptedModel(turn("Let me add.", call("add", a=2, b=3)), "The sum is 5.")
    result = await run(model)
    assert result.ok
    assert result.answer == "The sum is 5."  # the thought was not mistaken for the answer
    assert (result.steps, result.tool_calls) == (2, 1)
    assert result.usage == Usage(input_tokens=20, output_tokens=10)
    assert types(result) == [
        "run_started",
        "model_requested",
        "model_responded",
        "tools_dispatching",
        "tool_resulted",
        "model_requested",
        "model_responded",
        "run_finished",
    ]
    assert fake_tools.CALLS == ["add"]


async def test_the_model_sees_its_own_call_and_the_result_on_the_next_turn() -> None:
    model = ScriptedModel(turn("Let me add.", call("add", a=2, b=3)), "5")
    await run(model)
    second = model.requests[1].messages
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[1].text == "Let me add."
    assert [(c.name, c.arguments) for c in second[1].tool_calls] == [("add", {"a": 2, "b": 3})]
    (part,) = second[2].results
    assert (part.name, part.content, part.is_error, part.call_id) == (
        "add",
        "5",
        False,
        "call_add",
    )


async def test_a_chain_of_tool_rounds_grows_the_transcript_each_time() -> None:
    model = ScriptedModel(
        turn("", call("add", a=1, b=1)),
        turn("", call("add", a=2, b=2)),
        turn("", call("echo", text="done")),
        "all done",
    )
    result = await run(model)
    assert (result.steps, result.tool_calls) == (4, 3)
    roles = [m.role for m in model.requests[3].messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]
    assert tool_message_contents(model.requests[3]) == ["2", "4", "done"]


async def test_the_tools_on_offer_are_the_registrys_specs_and_listed_in_the_log() -> None:
    a = agent()
    model = ScriptedModel("ok")
    result = await run(model, a)
    assert model.requests[0].tools == tool_specs(a.tools)
    assert [t.name for t in model.requests[0].tools] == [t.name for t in a.tools]
    started = result.events[0]
    assert isinstance(started, RunStarted)
    assert started.tools == [t.name for t in a.tools]
    add_spec = next(t for t in model.requests[0].tools if t.name == "add")
    assert add_spec.parameters["properties"]["a"] == {"type": "integer"}
    assert add_spec.description == "Add two numbers."


async def test_the_default_system_prompt_mentions_tools_only_when_there_are_some() -> None:
    with_tools = ScriptedModel("ok")
    await run(with_tools, agent())
    assert with_tools.requests[0].system == f"{DEFAULT_SYSTEM} {TOOL_GUIDANCE}"
    without = ScriptedModel("ok")
    await run(without, Agent(model="m"))
    assert without.requests[0].system == DEFAULT_SYSTEM
    custom = ScriptedModel("ok")
    await run(custom, agent(system="Only say yes."))
    assert custom.requests[0].system == "Only say yes."  # a caller's prompt is used verbatim


# ---- observations ----


async def test_parallel_results_reach_the_model_in_call_order_not_completion_order() -> None:
    model = ScriptedModel(
        turn(
            "",
            call("slow", "c0", ms=60, label="a"),
            call("slow", "c1", ms=10, label="b"),
            call("slow", "c2", ms=30, label="c"),
        ),
        "done",
    )
    result = await run(model)
    assert tool_message_contents(model.requests[1]) == ["slow-a", "slow-b", "slow-c"]
    indexes = [e.index for e in result.events if isinstance(e, ToolResulted)]
    assert indexes == [0, 1, 2]
    assert result.tool_calls == 3


async def test_a_tool_error_is_an_observation_and_the_run_carries_on() -> None:
    model = ScriptedModel(turn("", call("boom")), "The backend is down, so I cannot continue.")
    result = await run(model)
    assert result.ok
    (part,) = model.requests[1].messages[-1].results
    assert part.content == "Error (tool_failed): backend down"
    assert part.is_error
    outcome = next(e for e in result.events if isinstance(e, ToolResulted)).outcome
    assert (outcome.ok, outcome.error_code) == (False, "tool_failed")


async def test_a_bad_call_is_repaired_from_the_error_text() -> None:
    model = ScriptedModel(
        turn("", call("add", a="two", b=3)),
        turn("", call("add", a=2, b=3)),
        "5",
    )
    result = await run(model)
    assert result.ok
    first_feedback = tool_message_contents(model.requests[1])[0]
    assert first_feedback.startswith("Error (arguments_invalid):")
    assert "Expected: add(a: integer, b?: integer = 0)" in first_feedback
    assert tool_message_contents(model.requests[2]) == [first_feedback, "5"]
    assert (result.steps, result.tool_calls) == (3, 2)


async def test_a_misspelled_tool_gets_a_did_you_mean() -> None:
    model = ScriptedModel(turn("", call("ad", a=1)), turn("", call("add", a=1)), "1")
    await run(model)
    assert "Did you mean 'add'?" in tool_message_contents(model.requests[1])[0]


async def test_an_unexpected_tool_exception_never_leaks_into_the_transcript() -> None:
    model = ScriptedModel(turn("", call("crash", path="x")), "it failed")
    await run(model)
    feedback = tool_message_contents(model.requests[1])[0]
    assert feedback == "Error (tool_failed): Tool 'crash' failed unexpectedly (KeyError)."
    assert "/etc/secret" not in feedback
    assert "abc123" not in feedback


async def test_a_slow_tool_times_out_and_the_run_continues() -> None:
    model = ScriptedModel(turn("", call("hang")), "the tool timed out")
    result = await run(model)
    assert result.ok
    assert tool_message_contents(model.requests[1])[0].startswith("Error (tool_timeout):")
    assert result.elapsed_s < 5


async def test_long_tool_output_arrives_truncated_with_its_marker() -> None:
    model = ScriptedModel(turn("", call("big")), "ok")
    result = await run(model)
    content = tool_message_contents(model.requests[1])[0]
    assert len(content) <= 60
    assert "[truncated" in content
    outcome = next(e for e in result.events if isinstance(e, ToolResulted)).outcome
    assert outcome.truncated


async def test_duplicate_call_ids_in_one_turn_do_not_confuse_the_loop() -> None:
    model = ScriptedModel(
        turn("", call("echo", "dup", text="first"), call("echo", "dup", text="second")), "ok"
    )
    await run(model)
    assert tool_message_contents(model.requests[1]) == ["first", "second"]


# ---- safety ----


async def test_write_tools_are_denied_by_default_and_never_run() -> None:
    model = ScriptedModel(turn("", call("save", name="x")), "I could not save it.")
    result = await run(model)
    assert result.ok
    assert fake_tools.CALLS == []  # the handler never ran
    feedback = tool_message_contents(model.requests[1])[0]
    assert feedback.startswith("Error (policy_denied):")
    assert "read-only" in feedback


async def test_writes_run_only_when_the_agent_is_given_a_permissive_policy() -> None:
    model = ScriptedModel(turn("", call("save", name="x")), "saved")
    result = await run(model, agent(policy=AllowAll()))
    assert result.ok
    assert fake_tools.CALLS == ["save"]
    assert tool_message_contents(model.requests[1]) == ["saved x"]


# ---- lenient validation ----


async def test_numbers_sent_as_strings_are_rejected_by_default_and_accepted_with_coerce() -> None:
    def script() -> ScriptedModel:
        return ScriptedModel(turn("", call("add", a="17", b="25")), "done")

    strict = script()
    await run(strict)
    feedback = tool_message_contents(strict.requests[1])[0]
    assert feedback.startswith("Error (arguments_invalid):")
    assert fake_tools.CALLS == []

    lenient = script()
    await run(lenient, agent(coerce=True))
    assert tool_message_contents(lenient.requests[1]) == ["42"]
    assert fake_tools.CALLS == ["add"]


async def test_coerce_never_loosens_a_write_tool() -> None:
    from toolrouter import SideEffect, tool

    ran: list[int] = []

    @tool(side_effect=SideEffect.WRITE)
    def write_number(n: int) -> str:
        """A WRITE tool with an integer argument."""
        ran.append(n)
        return "written"

    model = ScriptedModel(turn("", call("write_number", n="5")), "done")
    a = Agent(model="m", tools=ToolRegistry([write_number]), policy=AllowAll(), coerce=True)
    await run(model, a)
    assert ran == []  # "5" was not coerced: WRITE tools must opt in themselves
    assert tool_message_contents(model.requests[1])[0].startswith("Error (arguments_invalid):")


# ---- budgets ----


async def test_too_many_tool_calls_in_one_turn_fail_the_run_before_any_run() -> None:
    model = ScriptedModel(turn("", call("add", a=1), call("add", a=2), call("add", a=3)), "never")
    result = await run(model, agent(budgets=Budgets(max_tool_calls=2)))
    assert result.status == "failed"
    assert result.error == "tool call budget exhausted (2)"
    assert fake_tools.CALLS == []  # nothing ran: the whole turn is refused
    assert "tools_dispatching" not in types(result)
    assert result.tool_calls == 0


async def test_the_tool_call_budget_spans_the_whole_run() -> None:
    model = ScriptedModel(turn("", call("add", a=1)), turn("", call("add", a=2)), "never")
    result = await run(model, agent(budgets=Budgets(max_tool_calls=1)))
    assert result.status == "failed"
    assert result.error == "tool call budget exhausted (1)"
    assert (result.tool_calls, fake_tools.CALLS) == (1, ["add"])  # the first batch did run


async def test_a_model_that_never_stops_calling_tools_is_stopped_by_the_step_budget() -> None:
    forever = [turn("", call("add", a=1)) for _ in range(50)]
    model = ScriptedModel(*forever)
    result = await run(model, agent(budgets=Budgets(max_steps=3)))
    assert result.status == "failed"
    assert result.error == "step budget exhausted (3)"
    assert (result.steps, len(model.requests)) == (3, 3)
    assert result.tool_calls == 3


# ---- the log ----


async def test_dispatching_is_logged_before_any_result_and_results_cover_the_batch() -> None:
    model = ScriptedModel(turn("", call("add", a=1), call("echo", text="x")), "done")
    result = await run(model)
    kinds = types(result)
    dispatching = kinds.index("tools_dispatching")
    resulted = [i for i, k in enumerate(kinds) if k == "tool_resulted"]
    assert resulted == [dispatching + 1, dispatching + 2]
    batch = result.events[dispatching]
    assert isinstance(batch, ToolsDispatching)
    assert [c.name for c in batch.calls] == ["add", "echo"]
    outcomes = [e.outcome for e in result.events if isinstance(e, ToolResulted)]
    assert [(o.name, o.ok, o.text) for o in outcomes] == [("add", True, "1"), ("echo", True, "x")]
    assert all(o.duration_ms >= 0 for o in outcomes)


async def test_replaying_a_multi_step_log_reproduces_the_run_exactly() -> None:
    model = ScriptedModel(
        turn("", call("add", a=1, b=2), call("boom")), turn("", call("echo", text="hi")), "fin"
    )
    result = await run(model)
    state = replay(result.events)
    assert (state.status, state.answer, state.steps, state.tool_calls_made) == (
        result.status,
        result.answer,
        result.steps,
        result.tool_calls,
    )
    assert state.usage == result.usage
    # the replayed transcript is what the model last saw, plus its final answer
    assert state.messages[:-1] == model.requests[-1].messages
    assert state.messages[-1].text == "fin"
    assert state.pending is None


async def test_the_result_with_tool_events_round_trips_through_json() -> None:
    model = ScriptedModel(turn("", call("add", a=1)), "done")
    result = await run(model)
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert replay(restored.events) == replay(result.events)


# ---- many runs, cancellation ----


async def test_concurrent_runs_with_tools_are_isolated() -> None:
    class AddThenAnswer:
        async def respond(self, request: TurnRequest) -> Any:
            last = request.messages[-1]
            if last.role == "tool":
                return turn(f"result: {last.results[0].content}")
            return turn("", call("add", a=int(last.text)))

    rt = AgentRuntime(AddThenAnswer())
    a = agent()
    results = await asyncio.gather(*(rt.run(a, str(i)) for i in range(20)))
    assert [r.answer for r in results] == [f"result: {i}" for i in range(20)]
    assert len({r.run_id for r in results}) == 20
    assert all(r.steps == 2 and r.tool_calls == 1 for r in results)


async def test_cancelling_a_run_cancels_the_tool_it_is_waiting_on() -> None:
    model = ScriptedModel(turn("", call("wait_forever")), "never")
    task = asyncio.create_task(AgentRuntime(model).run(agent(), "g"))
    for _ in range(200):
        if fake_tools.CALLS == ["wait_forever"]:
            break
        await asyncio.sleep(0.01)
    assert fake_tools.CALLS == ["wait_forever"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_agent_is_still_a_plain_frozen_value_with_tools() -> None:
    a = agent()
    assert isinstance(a.tools, ToolRegistry)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.tools = ToolRegistry()  # type: ignore[misc]
    assert ModelRequested(step=1).type == "model_requested"
    assert ModelResponded(step=1, text="").type == "model_responded"
