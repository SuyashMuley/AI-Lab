"""What a run returns."""

from __future__ import annotations

from llmcli.core.models import Usage
from pydantic import BaseModel, ConfigDict, Field

from localagent.events import RunEvent, RunStatus
from localagent.state import RunState


class Recovery(BaseModel):
    """One self-healing move the runtime made (filled in from R4 on)."""

    model_config = ConfigDict(frozen=True)

    kind: str
    detail: str = ""


class RunResult(BaseModel):
    """The honest outcome of a run: how it ended, what it cost, and everything that happened."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    answer: str | None = None
    error: str | None = None  # why it failed; never contains internals beyond the error's message
    steps: int
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    elapsed_s: float
    recoveries: list[Recovery] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list, repr=False)  # the full log, replayable

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @classmethod
    def from_state(cls, state: RunState, events: list[RunEvent]) -> RunResult:
        if state.status is None:
            raise ValueError("cannot build a result from an unfinished run")
        return cls(
            run_id=state.run_id,
            status=state.status,
            answer=state.answer,
            error=state.error,
            steps=state.steps,
            tool_calls=state.tool_calls_made,
            usage=state.usage,
            elapsed_s=state.elapsed_s,
            events=events,
        )
