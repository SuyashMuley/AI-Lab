"""Fuzz the loop: whatever the model does, a run ends with an honest, replayable result."""

import asyncio
from typing import Any

from hypothesis import given
from hypothesis import strategies as st
from llmcli.core.models import ToolCallPart, Usage

from localagent import (
    Agent,
    AgentRuntime,
    Budgets,
    CallModel,
    DispatchTools,
    Finish,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    ModelTurn,
    RunEvent,
    RunFinished,
    RunStarted,
    Stop,
    decide,
    reduce,
    replay,
)
from localagent.testing import ScriptedModel
from tests.fuzz import fuzz

texts = st.text(max_size=60)
usages = st.builds(
    Usage,
    input_tokens=st.integers(0, 10**6),
    output_tokens=st.integers(0, 10**6),
    cached_tokens=st.integers(0, 1000),
)
tool_calls = st.lists(
    st.builds(
        ToolCallPart,
        id=st.text(min_size=1, max_size=8),
        name=st.text(min_size=1, max_size=8),
        arguments=st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
    ),
    max_size=3,
)
turns = st.builds(
    ModelTurn,
    text=texts,
    tool_calls=tool_calls,
    usage=usages,
    finish_reason=st.none() | st.text(max_size=6),
)
errors = st.builds(
    lambda cls, message: cls(message),
    st.sampled_from([RuntimeError, ValueError, TimeoutError, OSError, KeyError]),
    texts,
)
steps = st.one_of(turns, errors, texts)


def expected_status(step: Any) -> str:
    if isinstance(step, BaseException):
        return "failed"
    if isinstance(step, ModelTurn):
        return "failed" if step.tool_calls or not step.text.strip() else "completed"
    return "completed" if step.strip() else "failed"


@fuzz(400)
@given(steps)
def test_any_model_behaviour_ends_in_a_consistent_replayable_result(step: Any) -> None:
    result = asyncio.run(AgentRuntime(ScriptedModel(step)).run(Agent(model="m"), "goal"))

    assert result.status == expected_status(step)
    assert result.ok == (result.status == "completed")
    assert (result.answer is None) == (result.status == "failed")
    assert (result.error is None) == (result.status == "completed")
    asked_for_tools = isinstance(step, ModelTurn) and bool(step.tool_calls)
    assert result.steps == (2 if asked_for_tools else 1)  # tool calls trigger a second model call
    assert result.tool_calls == (len(step.tool_calls) if asked_for_tools else 0)

    log = result.events
    assert [e.seq for e in log] == list(range(len(log)))  # dense and monotonic
    assert [e.at for e in log] == sorted(e.at for e in log)
    assert sum(isinstance(e, RunFinished) for e in log) == 1
    assert isinstance(log[-1], RunFinished)
    assert isinstance(log[0], RunStarted)

    state = replay(log)  # the log alone reproduces the result
    assert (state.status, state.answer, state.error) == (result.status, result.answer, result.error)
    assert (state.steps, state.usage) == (result.steps, result.usage)

    if isinstance(step, ModelTurn):
        assert result.usage == step.usage  # what the model cost is always recorded
    elif isinstance(step, BaseException):
        assert result.usage == Usage()


# A random but VALID event sequence: started, then (requested, responded|failed)*, maybe finished.
outcomes = st.lists(
    st.one_of(
        st.tuples(st.just("ok"), texts, tool_calls, usages),
        st.tuples(st.just("fail"), texts, st.just([]), st.just(Usage())),
    ),
    max_size=8,
)


def build_events(outcome_list: list[Any], finish: bool) -> list[RunEvent]:
    events: list[RunEvent] = [
        RunStarted(run_id="r", model="m", system="S", goal="g", budgets=Budgets(max_steps=100))
    ]
    for step, (kind, text, calls, usage) in enumerate(outcome_list, start=1):
        events.append(ModelRequested(step=step))
        if kind == "ok":
            events.append(ModelResponded(step=step, text=text, tool_calls=calls, usage=usage))
        else:
            events.append(ModelFailed(step=step, kind="error", message=text))
    if finish:
        events.append(RunFinished(status="failed", error="e"))
    return events


@fuzz(300)
@given(outcomes, st.booleans())
def test_the_fold_agrees_with_the_events_it_folds(outcome_list: list[Any], finish: bool) -> None:
    events = build_events(outcome_list, finish)
    state = replay(events)
    responded = [e for e in events if isinstance(e, ModelResponded)]
    assert state.steps == sum(isinstance(e, ModelRequested) for e in events)
    assert len(state.messages) == 1 + len(responded)
    total = Usage()
    for e in responded:
        total = total + e.usage
    assert state.usage == total
    assert state.finished == finish
    assert replay(events) == state  # deterministic

    # reducing one event at a time passes through exactly the states replaying each prefix does
    running = None
    for i, event in enumerate(events):
        running = reduce(running, event)
        assert running == replay(events[: i + 1])


@fuzz(300)
@given(outcomes, st.booleans())
def test_decide_always_answers_and_stops_exactly_when_finished(
    outcome_list: list[Any], finish: bool
) -> None:
    state = None
    for event in build_events(outcome_list, finish):
        state = reduce(state, event)
        action = decide(state)
        assert isinstance(action, CallModel | DispatchTools | Finish | Stop)
        assert isinstance(action, Stop) == state.finished
        assert decide(state) == action  # pure
