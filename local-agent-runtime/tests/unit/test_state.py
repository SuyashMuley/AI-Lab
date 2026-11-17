import pytest
from llmcli.core.models import ToolCallPart, Usage
from pydantic import TypeAdapter, ValidationError

from localagent import (
    Budgets,
    EventLog,
    InvalidEventError,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    RunEvent,
    RunFinished,
    RunStarted,
    RunState,
    ToolOutcome,
    ToolResulted,
    ToolsDispatching,
    reduce,
    replay,
)
from localagent.testing import FakeClock, call


def started(at: float = 0.0) -> RunStarted:
    return RunStarted(run_id="r1", model="m", system="SYS", goal="do it", budgets=Budgets(), at=at)


def responded(
    text: str = "ok",
    *,
    step: int = 1,
    calls: list[ToolCallPart] | None = None,
    tokens: int = 5,
    at: float = 0.0,
) -> ModelResponded:
    return ModelResponded(
        step=step,
        text=text,
        tool_calls=calls or [],
        usage=Usage(input_tokens=tokens, output_tokens=1),
        finish_reason="stop",
        at=at,
    )


# ---- reduce ----


def test_run_started_builds_the_initial_state() -> None:
    s = reduce(None, started())
    assert (s.run_id, s.model, s.system, s.goal) == ("r1", "m", "SYS", "do it")
    assert [(m.role, m.text) for m in s.messages] == [("user", "do it")]
    assert (s.steps, s.status, s.answer, s.error, s.last_turn) == (0, None, None, None, None)
    assert not s.finished


def test_model_requested_counts_a_step_and_clears_the_last_turn_and_error() -> None:
    s = reduce(None, started())
    s = reduce(s, ModelRequested(step=1))
    s = reduce(s, responded())
    assert s.last_turn is not None
    s = reduce(s, ModelFailed(step=1, kind="error", message="boom"))
    assert (s.model_error, s.last_turn) == ("boom", None)
    s = reduce(s, ModelRequested(step=2))
    assert (s.steps, s.last_turn, s.model_error) == (2, None, None)


def test_model_responded_accumulates_usage_and_extends_the_transcript() -> None:
    s = reduce(None, started())
    s = reduce(s, ModelRequested(step=1))
    s = reduce(s, responded("first", calls=[call("t", a=1)], tokens=5))
    s = reduce(s, ModelRequested(step=2))
    s = reduce(s, responded("second", step=2, tokens=7))
    assert s.usage == Usage(input_tokens=12, output_tokens=2)
    assert [m.role for m in s.messages] == ["user", "assistant", "assistant"]
    assert s.messages[1].tool_calls[0].name == "t"
    assert s.messages[2].text == "second"
    assert s.last_turn is not None
    assert s.last_turn.text == "second"
    assert s.last_turn.tool_calls == []
    assert s.last_turn.finish_reason == "stop"


def test_run_finished_sets_the_outcome() -> None:
    s = reduce(None, started())
    done = reduce(s, RunFinished(status="completed", answer="42"))
    assert (done.status, done.answer, done.error, done.finished) == ("completed", "42", None, True)
    failed = reduce(s, RunFinished(status="failed", error="why"))
    assert (failed.status, failed.answer, failed.error) == ("failed", None, "why")


def test_elapsed_tracks_the_latest_event_time() -> None:
    s = reduce(None, started(at=0.0))
    s = reduce(s, ModelRequested(step=1, at=1.5))
    s = reduce(s, responded(at=4.25))
    assert s.elapsed_s == 4.25


def test_an_event_before_run_started_is_rejected() -> None:
    for event in (ModelRequested(step=1), responded(), RunFinished(status="failed")):
        with pytest.raises(InvalidEventError, match="before run_started"):
            reduce(None, event)


def test_a_second_run_started_is_rejected() -> None:
    with pytest.raises(InvalidEventError, match="must be the first event"):
        reduce(reduce(None, started()), started())


def test_nothing_may_follow_run_finished() -> None:
    done = reduce(reduce(None, started()), RunFinished(status="completed", answer="x"))
    for event in (ModelRequested(step=1), responded(), RunFinished(status="failed")):
        with pytest.raises(InvalidEventError, match="after the run finished"):
            reduce(done, event)


def test_reduce_is_pure_and_never_shares_mutable_state() -> None:
    s0 = reduce(None, started())
    before = s0.model_dump()
    s1 = reduce(s0, responded("x"))
    assert s0.model_dump() == before  # the old state is untouched
    assert s0.messages is not s1.messages
    assert len(s0.messages) == 1
    assert len(s1.messages) == 2


def test_state_events_and_budgets_are_immutable() -> None:
    s = reduce(None, started())
    with pytest.raises(ValidationError):
        s.steps = 5
    with pytest.raises(ValidationError):
        started().goal = "other"
    with pytest.raises(ValidationError):
        Budgets().max_steps = 1


def test_replay_folds_a_log_and_rejects_an_empty_one() -> None:
    events: list[RunEvent] = [
        started(),
        ModelRequested(step=1, seq=1),
        responded().model_copy(update={"seq": 2}),
        RunFinished(status="completed", answer="ok", seq=3),
    ]
    s = replay(events)
    assert isinstance(s, RunState)
    assert (s.status, s.steps, s.answer) == ("completed", 1, "ok")
    with pytest.raises(InvalidEventError, match="empty log"):
        replay([])


# ---- the event log ----


def test_the_log_stamps_sequence_numbers_and_relative_times() -> None:
    clock = FakeClock(start=100.0)
    log = EventLog(clock)
    clock.advance(1.5)
    first = log.append(started())
    clock.advance(0.25)
    second = log.append(ModelRequested(step=1))
    assert (first.seq, first.at) == (0, 1.5)
    assert (second.seq, second.at) == (1, 1.75)
    assert len(log) == 2


def test_the_log_returns_copies_and_never_edits_what_it_was_given() -> None:
    log = EventLog(FakeClock())
    original = started()
    stamped = log.append(original)
    assert original.seq == 0  # the caller's event is untouched
    assert stamped is not original
    snapshot = log.events
    snapshot.clear()
    assert len(log.events) == 1  # mutating the returned list does not touch the log


def test_events_round_trip_through_json_by_type() -> None:
    adapter = TypeAdapter(list[RunEvent])
    events: list[RunEvent] = [
        started(),
        ModelRequested(step=1, seq=1, at=0.5),
        responded(calls=[call("t", x=1)]),
        ModelFailed(step=1, kind="timeout", message="slow"),
        RunFinished(status="failed", error="e"),
    ]
    restored = adapter.validate_json(adapter.dump_json(events))
    assert restored == events
    assert [type(e).__name__ for e in restored] == [type(e).__name__ for e in events]


@pytest.mark.parametrize(
    "bad",
    [
        {"max_steps": 0},
        {"max_steps": -1},
        {"model_timeout_s": 0},
        {"model_timeout_s": -1.0},
        {"model_timeout_s": float("inf")},
        {"model_timeout_s": float("nan")},
    ],
)
def test_budgets_reject_unusable_values(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Budgets(**bad)  # type: ignore[arg-type]


def test_budgets_defaults_are_sane() -> None:
    b = Budgets()
    assert b.max_steps >= 1
    assert b.model_timeout_s > 0


# ---- tool batches ----


def outcome(name: str = "t", *, ok: bool = True, text: str = "") -> ToolOutcome:
    return ToolOutcome(
        call_id=f"call_{name}",
        name=name,
        ok=ok,
        text=text or f"{name} result",
        error_code=None if ok else "tool_failed",
    )


def after_reply(calls: list[ToolCallPart]) -> RunState:
    s = reduce(None, started())
    s = reduce(s, ModelRequested(step=1))
    return reduce(s, responded("thinking", calls=calls))


def test_dispatching_opens_a_batch_and_counts_the_calls() -> None:
    calls = [call("a"), call("b")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    assert s.pending is not None
    assert (s.pending.step, s.pending.calls, s.pending.outcomes) == (1, calls, {})
    assert s.tool_calls_made == 2


def test_the_tool_message_is_assembled_in_call_order_whatever_order_results_arrive() -> None:
    calls = [call("a"), call("b"), call("c")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    s = reduce(s, ToolResulted(step=1, index=2, outcome=outcome("c")))  # finished first
    assert s.pending is not None
    assert len(s.messages) == 2  # still waiting: no tool message yet
    s = reduce(s, ToolResulted(step=1, index=0, outcome=outcome("a")))
    s = reduce(
        s, ToolResulted(step=1, index=1, outcome=outcome("b", ok=False, text="Error (x): no"))
    )
    assert s.pending is None
    assert s.last_turn is None  # the run goes back to the model
    tool_message = s.messages[-1]
    assert tool_message.role == "tool"
    assert [(r.name, r.content, r.is_error) for r in tool_message.results] == [
        ("a", "a result", False),
        ("b", "Error (x): no", True),
        ("c", "c result", False),
    ]
    assert [r.call_id for r in tool_message.results] == ["call_a", "call_b", "call_c"]


def test_duplicate_call_ids_are_told_apart_by_position() -> None:
    calls = [call("t", "dup", n=1), call("t", "dup", n=2)]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    s = reduce(s, ToolResulted(step=1, index=1, outcome=outcome(text="second")))
    s = reduce(s, ToolResulted(step=1, index=0, outcome=outcome(text="first")))
    assert [r.content for r in s.messages[-1].results] == ["first", "second"]


def test_the_transcript_after_a_tool_round_is_user_assistant_tool() -> None:
    calls = [call("a")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    s = reduce(s, ToolResulted(step=1, index=0, outcome=outcome("a")))
    assert [m.role for m in s.messages] == ["user", "assistant", "tool"]
    assert s.messages[1].text == "thinking"
    assert [c.name for c in s.messages[1].tool_calls] == ["a"]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: reduce(reduce(None, started()), ToolsDispatching(step=1, calls=[call("a")])),
            "without tool calls",
        ),
        (
            lambda: reduce(
                after_reply([call("a")]), ToolsDispatching(step=1, calls=[call("other")])
            ),
            "does not match",
        ),
        (
            lambda: reduce(after_reply([call("a")]), ToolsDispatching(step=9, calls=[call("a")])),
            "does not match",
        ),
        (lambda: reduce(after_reply([]), ToolsDispatching(step=1, calls=[])), "without tool calls"),
        (
            lambda: reduce(
                after_reply([call("a")]), ToolResulted(step=1, index=0, outcome=outcome())
            ),
            "no tool batch in flight",
        ),
    ],
)
def test_impossible_tool_sequences_are_rejected(build: object, message: str) -> None:
    with pytest.raises(InvalidEventError, match=message):
        build()  # type: ignore[operator]


def test_a_batch_cannot_be_opened_twice_or_overtaken_by_a_model_request() -> None:
    calls = [call("a"), call("b")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    with pytest.raises(InvalidEventError, match="unresolved"):
        reduce(s, ToolsDispatching(step=1, calls=calls))
    with pytest.raises(InvalidEventError, match="unresolved"):
        reduce(s, ModelRequested(step=2))


def test_results_must_match_the_batch_and_arrive_once() -> None:
    calls = [call("a"), call("b")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    for index in (-1, 2, 99):
        with pytest.raises(InvalidEventError, match="does not match"):
            reduce(s, ToolResulted(step=1, index=index, outcome=outcome()))
    with pytest.raises(InvalidEventError, match="does not match"):
        reduce(s, ToolResulted(step=2, index=0, outcome=outcome()))
    once = reduce(s, ToolResulted(step=1, index=0, outcome=outcome("a")))
    with pytest.raises(InvalidEventError, match="twice"):
        reduce(once, ToolResulted(step=1, index=0, outcome=outcome("a")))


def test_a_run_can_finish_with_a_batch_still_in_flight() -> None:
    calls = [call("a")]
    s = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    done = reduce(s, RunFinished(status="failed", error="cancelled mid-batch"))
    assert done.finished
    assert done.pending is not None  # the log keeps the evidence of what was unresolved


def test_tool_events_do_not_mutate_the_previous_state() -> None:
    calls = [call("a"), call("b")]
    s0 = reduce(after_reply(calls), ToolsDispatching(step=1, calls=calls))
    before = s0.model_dump()
    s1 = reduce(s0, ToolResulted(step=1, index=0, outcome=outcome("a")))
    assert s0.model_dump() == before
    assert s1.pending is not None
    assert s0.pending is not None
    assert s0.pending.outcomes == {}
    assert len(s1.pending.outcomes) == 1


def test_tool_events_round_trip_through_json() -> None:
    adapter = TypeAdapter(list[RunEvent])
    events: list[RunEvent] = [
        ToolsDispatching(step=1, calls=[call("a", x=1)], seq=3, at=0.5),
        ToolResulted(step=1, index=0, outcome=outcome("a", ok=False), seq=4, at=0.75),
    ]
    assert adapter.validate_json(adapter.dump_json(events)) == events
