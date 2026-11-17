"""localagent: a bounded, observable, self-healing agent loop over local models and strict tools."""

from localagent.agent import DEFAULT_SYSTEM, Agent, AgentRuntime
from localagent.budgets import Budgets
from localagent.decide import Action, CallModel, DispatchTools, Finish, Stop, decide
from localagent.events import (
    EventLog,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    RunEvent,
    RunFinished,
    RunStarted,
    RunStatus,
    ToolOutcome,
    ToolResulted,
    ToolsDispatching,
)
from localagent.models import ModelClient, ModelTurn, TurnRequest
from localagent.result import Recovery, RunResult
from localagent.state import InvalidEventError, LastTurn, PendingBatch, RunState, reduce, replay
from localagent.tooling import TOOL_GUIDANCE, effective_system, tool_specs

__all__ = [
    "DEFAULT_SYSTEM",
    "Action",
    "Agent",
    "AgentRuntime",
    "Budgets",
    "CallModel",
    "DispatchTools",
    "EventLog",
    "Finish",
    "InvalidEventError",
    "LastTurn",
    "ModelClient",
    "ModelFailed",
    "ModelRequested",
    "ModelResponded",
    "ModelTurn",
    "PendingBatch",
    "Recovery",
    "RunEvent",
    "RunFinished",
    "RunResult",
    "RunStarted",
    "RunState",
    "RunStatus",
    "Stop",
    "TOOL_GUIDANCE",
    "ToolOutcome",
    "ToolResulted",
    "ToolsDispatching",
    "TurnRequest",
    "decide",
    "effective_system",
    "reduce",
    "replay",
    "tool_specs",
]
__version__ = "0.1.0"
