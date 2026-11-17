from llmcli.core.models import Usage

from localagent import (
    Budgets,
    CallModel,
    DispatchTools,
    Finish,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    RunFinished,
    RunStarted,
    RunState,
    Stop,
    ToolOutcome,
    ToolResulted,
    ToolsDispatching,
    decide,
    reduce,
)
from localagent.testing import call


def outcome(name: str) -> ToolOutcome:
    return ToolOutcome(call_id=f"call_{name}", name=name, ok=True, text=f"{name} ok")


def state_after(*events: object, budgets: Budgets | None = None) -> RunState:
    s = reduce(
        None,
        RunStarted(run_id="r", model="m", system="S", goal="g", budgets=budgets or Budgets()),
    )
    for e in events:
        s = reduce(s, e)  # type: ignore[arg-type]
    return s


def reply(text: str = "", calls: list | None = None) -> ModelResponded:  # type: ignore[type-arg]
    return ModelResponded(step=1, text=text, tool_calls=calls or [], usage=Usage())


def test_a_fresh_run_calls_the_model() -> None:
    assert decide(state_after()) == CallModel()


def test_a_text_reply_completes_the_run_with_that_answer() -> None:
    s = state_after(ModelRequested(step=1), reply("  The answer is 42.  "))
    assert decide(s) == Finish("completed", answer="  The answer is 42.  ")


def test_an_empty_reply_is_an_honest_failure_not_a_completed_run() -> None:
    for text in ("", "   ", "\n\t"):
        s = state_after(ModelRequested(step=1), reply(text))
        assert decide(s) == Finish("failed", error="the model returned an empty reply")


def test_tool_calls_are_dispatched_and_any_accompanying_text_is_not_the_answer() -> None:
    s = state_after(ModelRequested(step=1), reply("let me look", [call("a"), call("b")]))
    action = decide(s)
    assert isinstance(action, DispatchTools)
    assert [c.name for c in action.calls] == ["a", "b"]
    assert decide(s) != Finish("completed", answer="let me look")


def test_the_tool_call_budget_is_exact_and_checked_before_anything_runs() -> None:
    two = [call("a"), call("b")]
    ok = state_after(ModelRequested(step=1), reply("", two), budgets=Budgets(max_tool_calls=2))
    assert isinstance(decide(ok), DispatchTools)  # exactly at the limit is allowed
    over = state_after(ModelRequested(step=1), reply("", two), budgets=Budgets(max_tool_calls=1))
    assert decide(over) == Finish("failed", error="tool call budget exhausted (1)")
    none = state_after(
        ModelRequested(step=1), reply("", [call("a")]), budgets=Budgets(max_tool_calls=0)
    )
    assert decide(none) == Finish("failed", error="tool call budget exhausted (0)")


def test_the_tool_budget_counts_calls_across_batches() -> None:
    s = state_after(
        ModelRequested(step=1),
        reply("", [call("a"), call("b")]),
        ToolsDispatching(step=1, calls=[call("a"), call("b")]),
        ToolResulted(step=1, index=0, outcome=outcome("a")),
        ToolResulted(step=1, index=1, outcome=outcome("b")),
        ModelRequested(step=2),
        reply("", [call("c")]).model_copy(update={"step": 2}),
        budgets=Budgets(max_tool_calls=2),
    )
    assert s.tool_calls_made == 2
    assert decide(s) == Finish("failed", error="tool call budget exhausted (2)")


def test_a_finished_batch_sends_the_run_back_to_the_model() -> None:
    s = state_after(
        ModelRequested(step=1),
        reply("", [call("a")]),
        ToolsDispatching(step=1, calls=[call("a")]),
    )
    assert decide(s) == Finish("failed", error="a tool batch was left unresolved")  # R8 resumes it
    s = reduce(s, ToolResulted(step=1, index=0, outcome=outcome("a")))
    assert decide(s) == CallModel()


def test_the_step_budget_applies_after_tool_results_too() -> None:
    s = state_after(
        ModelRequested(step=1),
        reply("", [call("a")]),
        ToolsDispatching(step=1, calls=[call("a")]),
        ToolResulted(step=1, index=0, outcome=outcome("a")),
        budgets=Budgets(max_steps=1),
    )
    assert decide(s) == Finish("failed", error="step budget exhausted (1)")


def test_a_model_error_ends_the_run_failed_for_now() -> None:
    s = state_after(ModelRequested(step=1), ModelFailed(step=1, kind="error", message="boom"))
    assert decide(s) == Finish("failed", error="boom")


def test_a_model_error_takes_priority_over_an_earlier_reply() -> None:
    s = state_after(
        ModelRequested(step=1),
        reply("x"),
        ModelFailed(step=1, kind="error", message="late failure"),
    )
    assert decide(s) == Finish("failed", error="late failure")


def test_the_step_budget_stops_further_model_calls() -> None:
    s = state_after(ModelRequested(step=1), budgets=Budgets(max_steps=1))
    s = reduce(s, ModelFailed(step=1, kind="error", message="x"))
    s = s.model_copy(update={"model_error": None})  # as if a retry had cleared it
    assert decide(s) == Finish("failed", error="step budget exhausted (1)")
    below = state_after(budgets=Budgets(max_steps=2))
    assert decide(below) == CallModel()


def test_a_finished_run_stops() -> None:
    s = state_after(RunFinished(status="completed", answer="x"))
    assert decide(s) == Stop()
    assert decide(state_after(RunFinished(status="failed", error="e"))) == Stop()


def test_decide_is_pure_and_deterministic() -> None:
    s = state_after(ModelRequested(step=1), reply("ok"))
    before = s.model_dump()
    assert decide(s) == decide(s)
    assert s.model_dump() == before
