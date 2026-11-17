"""`decide`: the next action, as a pure function of the state.

Every policy decision of the runtime lives here (and, from R4 on, every self-healing move), so
each one is a unit test with no model, no clock and no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmcli.core.models import ToolCallPart

from localagent.events import RunStatus
from localagent.state import RunState


@dataclass(frozen=True)
class CallModel:
    """Ask the model for its next turn."""


@dataclass(frozen=True)
class DispatchTools:
    """Run the model's tool calls and feed the results back."""

    calls: list[ToolCallPart] = field(default_factory=list)


@dataclass(frozen=True)
class Finish:
    """End the run with this outcome."""

    status: RunStatus
    answer: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Stop:
    """The run is already over; nothing more to do."""


Action = CallModel | DispatchTools | Finish | Stop


def decide(state: RunState) -> Action:
    if state.finished:
        return Stop()
    if state.model_error is not None:  # R4 adds bounded retries and recoveries before this
        return Finish("failed", error=state.model_error)

    if state.pending is not None:  # live runs finish a batch atomically; R8 resumes one properly
        return Finish("failed", error="a tool batch was left unresolved")

    turn = state.last_turn
    if turn is None:
        if state.steps >= state.budgets.max_steps:  # R6 turns this into a graceful wrap-up
            return Finish("failed", error=f"step budget exhausted ({state.budgets.max_steps})")
        return CallModel()

    if turn.tool_calls:  # whatever text came with the calls is the model's "thought"
        wanted = state.tool_calls_made + len(turn.tool_calls)
        if wanted > state.budgets.max_tool_calls:  # R6 turns this into a graceful wrap-up
            return Finish(
                "failed", error=f"tool call budget exhausted ({state.budgets.max_tool_calls})"
            )
        return DispatchTools(calls=list(turn.tool_calls))
    if not turn.text.strip():  # R4 nudges instead
        return Finish("failed", error="the model returned an empty reply")
    return Finish("completed", answer=turn.text)
