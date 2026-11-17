"""The run log: the single source of truth about what happened.

Every effect the runtime performs appends an event; state is a fold over the events
(see `state.reduce`). Events are immutable. `seq` and `at` (seconds since the run started) are
stamped by the `EventLog` as the event is appended.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from llmcli.core.models import ToolCallPart, Usage
from pydantic import BaseModel, ConfigDict, Field

from localagent.budgets import Budgets

RunStatus = Literal["completed", "failed"]  # partial / cancelled arrive with R6 / R9


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int = 0
    at: float = 0.0


class RunStarted(_Event):
    type: Literal["run_started"] = "run_started"
    run_id: str
    model: str
    system: str
    goal: str
    budgets: Budgets  # the log alone is enough to rebuild the state (replay, resume)
    tools: list[str] = Field(default_factory=list)  # names on offer, for readers of the log


class ModelRequested(_Event):
    type: Literal["model_requested"] = "model_requested"
    step: int


class ModelResponded(_Event):
    type: Literal["model_responded"] = "model_responded"
    step: int
    text: str
    tool_calls: list[ToolCallPart] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None


class ToolOutcome(BaseModel):
    """What one tool call came to, as the model saw it."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    ok: bool
    text: str  # exactly the text sent back to the model (errors read `Error (<code>): ...`)
    error_code: str | None = None
    duration_ms: float = 0.0
    truncated: bool = False


class ToolsDispatching(_Event):
    """A batch of tool calls is about to run (logged first, so a crash mid-batch is detectable)."""

    type: Literal["tools_dispatching"] = "tools_dispatching"
    step: int
    calls: list[ToolCallPart]


class ToolResulted(_Event):
    """One call of the batch finished. `index` is its position in the batch (ids can repeat)."""

    type: Literal["tool_resulted"] = "tool_resulted"
    step: int
    index: int
    outcome: ToolOutcome


class ModelFailed(_Event):
    type: Literal["model_failed"] = "model_failed"
    step: int
    kind: Literal["timeout", "error"]
    message: str


class RunFinished(_Event):
    type: Literal["run_finished"] = "run_finished"
    status: RunStatus
    answer: str | None = None
    error: str | None = None


RunEvent = Annotated[
    RunStarted
    | ModelRequested
    | ModelResponded
    | ToolsDispatching
    | ToolResulted
    | ModelFailed
    | RunFinished,
    Field(discriminator="type"),
]


class EventLog:
    """An append-only, in-memory log that stamps each event with its position and time."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._start = clock()
        self._events: list[RunEvent] = []

    def append(self, event: RunEvent) -> RunEvent:
        stamped = event.model_copy(
            update={"seq": len(self._events), "at": round(self._clock() - self._start, 6)}
        )
        self._events.append(stamped)
        return stamped

    @property
    def events(self) -> list[RunEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
