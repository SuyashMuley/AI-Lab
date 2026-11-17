"""Test doubles for code built on localagent (and for localagent's own tests).

`ScriptedModel` plays back a fixed script; `FakeClock` makes time deterministic. Both are public
so later projects (evals, orchestration) can reuse them.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from llmcli.core.models import ToolCallPart, Usage

from localagent.models import ModelTurn, TurnRequest

Step = ModelTurn | str | BaseException | Callable[[TurnRequest], Any]


class ScriptExhausted(AssertionError):
    """The runtime asked the scripted model for more turns than the script contains."""


def turn(
    text: str = "",
    *calls: ToolCallPart,
    usage: Usage | None = None,
    finish_reason: str | None = "stop",
) -> ModelTurn:
    """A model turn: text and/or tool calls."""
    return ModelTurn(
        text=text,
        tool_calls=list(calls),
        usage=usage if usage is not None else Usage(input_tokens=10, output_tokens=5),
        finish_reason=finish_reason,
    )


def call(name: str, call_id: str | None = None, /, **arguments: Any) -> ToolCallPart:
    """A tool call, with a deterministic id derived from the name unless given.

    `name` and `call_id` are positional-only, so a tool may have arguments with those names."""
    return ToolCallPart(id=call_id or f"call_{name}", name=name, arguments=arguments)


class ScriptedModel:
    """A `ModelClient` that plays back a script, one step per `respond` call.

    A step is a `ModelTurn`, a plain string (a text-only turn), an exception to raise, or a
    callable (sync or async) that receives the request and returns any of those. Every request
    is recorded in `requests`. Asking for more steps than scripted raises `ScriptExhausted`.
    """

    def __init__(self, *steps: Step) -> None:
        self._steps = list(steps)
        self.requests: list[TurnRequest] = []

    async def respond(self, request: TurnRequest) -> ModelTurn:
        self.requests.append(request)
        if not self._steps:
            raise ScriptExhausted(f"the script has no step for model call #{len(self.requests)}")
        step = self._steps.pop(0)
        while callable(step) and not isinstance(step, ModelTurn | BaseException):
            outcome = step(request)
            step = await outcome if inspect.isawaitable(outcome) else outcome
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, str):
            return turn(step)
        assert isinstance(step, ModelTurn)
        return step

    @property
    def remaining(self) -> int:
        return len(self._steps)


class FakeClock:
    """A controllable clock: call it for the time, `advance` to move it."""

    def __init__(self, start: float = 1000.0, tick: float = 0.0) -> None:
        self.now = start
        self.tick = tick  # added automatically on every read, to simulate work taking time

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


__all__ = ["Awaitable", "FakeClock", "ScriptExhausted", "ScriptedModel", "call", "turn"]
