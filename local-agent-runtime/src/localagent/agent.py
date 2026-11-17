"""The agent definition and the runtime that runs it.

`AgentRuntime.run` is a thin shell: it asks the pure `decide` what to do, performs that one
effect, appends the resulting events, and repeats. All logic is in `state` and `decide`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from llmcli.core.models import ToolSpec
from toolrouter import Policy, ReadOnlyPolicy, ToolCall, ToolRegistry, ToolRouter

from localagent.budgets import Budgets
from localagent.decide import CallModel, DispatchTools, Finish, decide
from localagent.events import (
    EventLog,
    ModelFailed,
    ModelRequested,
    ModelResponded,
    RunEvent,
    RunFinished,
    RunStarted,
    ToolResulted,
    ToolsDispatching,
)
from localagent.models import ModelClient, TurnRequest
from localagent.result import RunResult
from localagent.state import RunState, reduce
from localagent.tooling import DEFAULT_SYSTEM, effective_system, outcome_from_result, tool_specs

__all__ = ["DEFAULT_SYSTEM", "Agent", "AgentRuntime"]

_MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class Agent:
    """What to run: a model, its instructions, its tools, and the limits it operates under.

    Safe by default: the policy is read-only, so a model can use read tools but WRITE tools are
    denied until you pass a more permissive policy (e.g. `toolrouter.AllowAll()`).
    """

    model: str
    system: str | None = None  # None: the default prompt (with tool guidance when tools exist)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    policy: Policy = field(default_factory=ReadOnlyPolicy)
    budgets: Budgets = field(default_factory=Budgets)
    # Lenient argument validation for READ/NONE tools ("17" accepted for an int). Small models
    # often send numbers as strings; WRITE tools never inherit this (toolrouter's rule).
    coerce: bool = False


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class AgentRuntime:
    """Runs agents against a `ModelClient`. Stateless between runs, so it is safe to share."""

    def __init__(
        self,
        client: ModelClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        new_run_id: Callable[[], str] = _new_run_id,
    ) -> None:
        self._client = client
        self._clock = clock
        self._new_run_id = new_run_id

    async def run(self, agent: Agent, goal: str) -> RunResult:
        """Pursue `goal`. Always returns a RunResult for anything the model or tools can cause.

        Only `Exception` from the model is contained (tool failures are results, never raised);
        `CancelledError`, `KeyboardInterrupt` and `SystemExit` propagate.
        """
        log = EventLog(self._clock)
        state: RunState | None = None
        router = ToolRouter(agent.tools, policy=agent.policy, coerce=agent.coerce)
        specs = tool_specs(agent.tools)

        def apply(event: RunEvent) -> RunState:
            nonlocal state
            state = reduce(state, log.append(event))
            return state

        current = apply(
            RunStarted(
                run_id=self._new_run_id(),
                model=agent.model,
                system=effective_system(agent.system, bool(specs)),
                goal=goal,
                budgets=agent.budgets,
                tools=[s.name for s in specs],
            )
        )
        while True:
            action = decide(current)
            if isinstance(action, CallModel):
                current = await self._call_model(current, apply, specs)
            elif isinstance(action, DispatchTools):
                current = await self._dispatch_tools(current, apply, router)
            elif isinstance(action, Finish):
                current = apply(
                    RunFinished(status=action.status, answer=action.answer, error=action.error)
                )
            else:  # Stop: the run is over
                return RunResult.from_state(current, log.events)

    async def _call_model(
        self, state: RunState, apply: Callable[[RunEvent], RunState], specs: list[ToolSpec]
    ) -> RunState:
        state = apply(ModelRequested(step=state.steps + 1))
        request = TurnRequest(
            model=state.model, system=state.system, messages=state.messages, tools=specs
        )
        deadline = asyncio.timeout(state.budgets.model_timeout_s)
        try:
            async with deadline:
                turn = await self._client.respond(request)
        except Exception as exc:
            if isinstance(exc, TimeoutError) and deadline.expired():  # ours, not the client's own
                seconds = state.budgets.model_timeout_s
                return apply(
                    ModelFailed(
                        step=state.steps,
                        kind="timeout",
                        message=f"the model did not answer within {seconds:g}s",
                    )
                )
            message = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
            return apply(ModelFailed(step=state.steps, kind="error", message=message))
        return apply(
            ModelResponded(
                step=state.steps,
                text=turn.text,
                tool_calls=turn.tool_calls,
                usage=turn.usage,
                finish_reason=turn.finish_reason,
            )
        )

    async def _dispatch_tools(
        self, state: RunState, apply: Callable[[RunEvent], RunState], router: ToolRouter
    ) -> RunState:
        assert state.last_turn is not None  # decide only asks for this after a reply with calls
        calls = state.last_turn.tool_calls
        step = state.steps
        state = apply(ToolsDispatching(step=step, calls=calls))
        results = await router.dispatch_many(
            [ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls]
        )
        for index, result in enumerate(results):
            state = apply(ToolResulted(step=step, index=index, outcome=outcome_from_result(result)))
        return state
