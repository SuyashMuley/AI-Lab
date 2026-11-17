"""Chaos for the ReAct loop: arbitrary model behaviour against real tools stays safe and sane."""

import asyncio
from typing import Any

from hypothesis import given
from hypothesis import strategies as st
from llmcli.core.models import ToolCallPart, Usage
from toolrouter import ErrorCode, ToolRegistry

from localagent import (
    Agent,
    AgentRuntime,
    Budgets,
    ModelRequested,
    ModelTurn,
    RunFinished,
    ToolResulted,
    ToolsDispatching,
    replay,
)
from localagent.testing import ScriptedModel
from tests.fakes import tools as fake_tools
from tests.fuzz import fuzz

MAX_STEPS, MAX_TOOL_CALLS = 5, 6
REGISTRY = ToolRegistry(
    [fake_tools.add, fake_tools.echo, fake_tools.slow, fake_tools.boom, fake_tools.crash,
     fake_tools.save, fake_tools.big]
)  # fmt: skip
KNOWN_CODES = {code.value for code in ErrorCode}

names = st.sampled_from(["add", "echo", "slow", "boom", "crash", "save", "big", "nope", "ad", ""])
arg_keys = st.sampled_from(["a", "b", "text", "ms", "label", "path", "name", "x"])
arg_values = st.one_of(st.integers(-5, 5), st.text(max_size=5), st.none(), st.booleans())
calls = st.lists(
    st.builds(
        lambda n, i, a: ToolCallPart(id=i, name=n or "blank", arguments=a),
        names,
        st.sampled_from(["c0", "c1", "dup"]),
        st.dictionaries(arg_keys, arg_values, max_size=3),
    ),
    max_size=3,
)
usages = st.builds(Usage, input_tokens=st.integers(0, 1000), output_tokens=st.integers(0, 1000))
model_turns = st.builds(ModelTurn, text=st.text(max_size=20), tool_calls=calls, usage=usages)
errors = st.builds(lambda m: RuntimeError(m), st.text(max_size=10))
scripts = st.lists(st.one_of(model_turns, errors, st.text(max_size=10)), max_size=8)


async def run_script(script: list[Any]) -> Any:
    agent = Agent(
        model="m",
        tools=REGISTRY,
        budgets=Budgets(max_steps=MAX_STEPS, max_tool_calls=MAX_TOOL_CALLS, model_timeout_s=5),
    )
    return await asyncio.wait_for(AgentRuntime(ScriptedModel(*script)).run(agent, "goal"), 30)


@fuzz(150)
@given(scripts)
def test_arbitrary_model_behaviour_stays_inside_budgets_and_the_log_stays_consistent(
    script: list[Any],
) -> None:
    fake_tools.CALLS.clear()
    result = asyncio.run(run_script(script))
    log = result.events

    # terminates, bounded on every axis, honest status
    assert result.status in {"completed", "failed"}
    assert result.steps <= MAX_STEPS
    assert result.tool_calls <= MAX_TOOL_CALLS
    assert sum(isinstance(e, ModelRequested) for e in log) == result.steps
    assert (result.error is None) == (result.status == "completed")

    # the log is dense, ordered, and ends exactly once
    assert [e.seq for e in log] == list(range(len(log)))
    assert [e.at for e in log] == sorted(e.at for e in log)
    assert sum(isinstance(e, RunFinished) for e in log) == 1
    assert isinstance(log[-1], RunFinished)

    # every dispatched batch is resolved, completely and in order, before anything else happens
    dispatched = 0
    for k, event in enumerate(log):
        if isinstance(event, ToolsDispatching):
            n = len(event.calls)
            dispatched += n
            following = log[k + 1 : k + 1 + n]
            assert all(isinstance(e, ToolResulted) and e.step == event.step for e in following)
            assert [e.index for e in following if isinstance(e, ToolResulted)] == list(range(n))
            after = log[k + 1 + n]
            assert isinstance(after, ModelRequested | RunFinished)
    assert dispatched == result.tool_calls

    # replaying the log reproduces the run
    state = replay(log)
    assert (state.status, state.answer, state.error) == (result.status, result.answer, result.error)
    assert (state.steps, state.tool_calls_made, state.usage) == (
        result.steps,
        result.tool_calls,
        result.usage,
    )
    assert state.pending is None

    # the default policy is read-only: the WRITE tool's handler never runs, whatever was asked
    assert "save" not in fake_tools.CALLS

    # outcomes are either successes or errors with a real code
    for e in log:
        if isinstance(e, ToolResulted):
            assert e.outcome.ok == (e.outcome.error_code is None)
            assert e.outcome.error_code is None or e.outcome.error_code in KNOWN_CODES
            assert e.outcome.text
            assert "abc123" not in e.outcome.text  # crash() details never reach the transcript

    # the transcript is well formed: each assistant turn with calls is answered by exactly one
    # tool message with a result per call (the run may end right after a refused batch)
    msgs = state.messages
    assert msgs[0].role == "user"
    i = 1
    while i < len(msgs):
        m = msgs[i]
        assert m.role == "assistant"
        if m.tool_calls and i + 1 < len(msgs):
            reply = msgs[i + 1]
            assert reply.role == "tool"
            assert [r.name for r in reply.results] == [c.name for c in m.tool_calls]
            i += 2
        else:
            i += 1
