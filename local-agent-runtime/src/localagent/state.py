"""Run state is a fold over events: `reduce` is pure, so replaying a log rebuilds the state."""

from __future__ import annotations

from collections.abc import Iterable

from llmcli.core.models import Message, ToolCallPart, ToolResultPart, Usage
from pydantic import BaseModel, ConfigDict, Field

from localagent.budgets import Budgets
from localagent.events import (
    ModelFailed,
    ModelRequested,
    ModelResponded,
    RunEvent,
    RunStarted,
    RunStatus,
    ToolOutcome,
    ToolResulted,
    ToolsDispatching,
)


class InvalidEventError(ValueError):
    """An event that cannot follow the current state (a corrupt or out-of-order log)."""


class LastTurn(BaseModel):
    """The model's most recent reply, until the next request supersedes it."""

    model_config = ConfigDict(frozen=True)

    text: str
    tool_calls: list[ToolCallPart] = Field(default_factory=list)
    finish_reason: str | None = None


class PendingBatch(BaseModel):
    """A tool batch in flight: the calls, and the outcomes collected so far (by position)."""

    model_config = ConfigDict(frozen=True)

    step: int
    calls: list[ToolCallPart]
    outcomes: dict[int, ToolOutcome] = Field(default_factory=dict)


class RunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    model: str
    system: str
    goal: str
    budgets: Budgets
    messages: list[Message]  # the conversation the model sees (the system prompt is separate)
    steps: int = 0  # model calls requested so far
    tool_calls_made: int = 0  # tool calls dispatched so far
    usage: Usage = Field(default_factory=Usage)
    last_turn: LastTurn | None = None
    pending: PendingBatch | None = None  # a tool batch that has not fully resolved
    model_error: str | None = None  # set when a model call failed and nothing recovered yet
    status: RunStatus | None = None  # None while the run is still going
    answer: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0  # `at` of the latest event

    @property
    def finished(self) -> bool:
        return self.status is not None


def reduce(state: RunState | None, event: RunEvent) -> RunState:
    """The state after `event`. Pure: never mutates `state`; raises on an impossible sequence."""
    if isinstance(event, RunStarted):
        if state is not None:
            raise InvalidEventError("run_started must be the first event")
        return RunState(
            run_id=event.run_id,
            model=event.model,
            system=event.system,
            goal=event.goal,
            budgets=event.budgets,
            messages=[Message.of("user", event.goal)],
            elapsed_s=event.at,
        )
    if state is None:
        raise InvalidEventError(f"{event.type} before run_started")
    if state.finished:
        raise InvalidEventError(f"{event.type} after the run finished")

    base = {"elapsed_s": event.at}
    if isinstance(event, ModelRequested):
        if state.pending is not None:
            raise InvalidEventError("model_requested while a tool batch is unresolved")
        return state.model_copy(
            update={**base, "steps": state.steps + 1, "last_turn": None, "model_error": None}
        )
    if isinstance(event, ModelResponded):
        return state.model_copy(
            update={
                **base,
                "usage": state.usage + event.usage,
                "messages": [*state.messages, Message.assistant(event.text, event.tool_calls)],
                "last_turn": LastTurn(
                    text=event.text,
                    tool_calls=event.tool_calls,
                    finish_reason=event.finish_reason,
                ),
            }
        )
    if isinstance(event, ToolsDispatching):
        turn = state.last_turn
        if state.pending is not None:
            raise InvalidEventError("tools_dispatching while a tool batch is unresolved")
        if turn is None or not turn.tool_calls:
            raise InvalidEventError("tools_dispatching without tool calls from the model")
        if event.calls != turn.tool_calls or event.step != state.steps:
            raise InvalidEventError("tools_dispatching does not match the model's tool calls")
        return state.model_copy(
            update={
                **base,
                "pending": PendingBatch(step=event.step, calls=event.calls),
                "tool_calls_made": state.tool_calls_made + len(event.calls),
            }
        )
    if isinstance(event, ToolResulted):
        pending = state.pending
        if pending is None:
            raise InvalidEventError("tool_resulted with no tool batch in flight")
        if event.step != pending.step or not 0 <= event.index < len(pending.calls):
            raise InvalidEventError("tool_resulted does not match the tool batch in flight")
        if event.index in pending.outcomes:
            raise InvalidEventError(f"tool_resulted twice for call #{event.index}")
        outcomes = {**pending.outcomes, event.index: event.outcome}
        if len(outcomes) < len(pending.calls):
            partial = pending.model_copy(update={"outcomes": outcomes})
            return state.model_copy(update={**base, "pending": partial})
        # The batch is complete: the model sees one tool message, results in CALL order,
        # however they happened to finish.
        parts = [
            ToolResultPart(
                call_id=call.id,
                name=call.name,
                content=outcomes[i].text,
                is_error=not outcomes[i].ok,
            )
            for i, call in enumerate(pending.calls)
        ]
        return state.model_copy(
            update={
                **base,
                "pending": None,
                "last_turn": None,
                "messages": [*state.messages, Message.tool_results(parts)],
            }
        )
    if isinstance(event, ModelFailed):
        return state.model_copy(update={**base, "model_error": event.message, "last_turn": None})
    return state.model_copy(
        update={**base, "status": event.status, "answer": event.answer, "error": event.error}
    )


def replay(events: Iterable[RunEvent]) -> RunState:
    """Rebuild the state from a log. The same function the live runtime uses, so they agree."""
    state: RunState | None = None
    for event in events:
        state = reduce(state, event)
    if state is None:
        raise InvalidEventError("an empty log has no state")
    return state
