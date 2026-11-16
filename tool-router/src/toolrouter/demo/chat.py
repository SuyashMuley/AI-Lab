"""A minimal model <-> router loop with a bounded repair budget.

This is deliberately tiny (the real agent loop is a separate project). It exists to prove the
router against a real model and to measure how often models need to repair their calls.
It imports no provider SDK: the caller supplies `chat_fn`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from toolrouter.errors import ErrorCode
from toolrouter.formats import ollama as fmt
from toolrouter.formats._util import get
from toolrouter.router import ToolRouter

# (messages, tools payload) -> the assistant's message (a dict or an SDK object)
ChatFn = Callable[[list[Any], list[dict[str, Any]]], Awaitable[Any]]

DEFAULT_SYSTEM = (
    "You can call tools when they help. Call a tool only with arguments that match its "
    "signature. If a tool returns an error, read it, fix your call, and try again."
)

# Failures that are the model's to fix by changing its call (the repair budget counts these).
REPAIRABLE = frozenset(
    {
        ErrorCode.UNKNOWN_TOOL,
        ErrorCode.INVALID_JSON,
        ErrorCode.ARGUMENTS_INVALID,
        ErrorCode.ARGUMENTS_TOO_LARGE,
    }
)


class ModelTimeoutError(Exception):
    """The model did not answer within `model_timeout_s`."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"The model did not answer within {seconds:g}s")
        self.seconds = seconds


@dataclass
class ChatReport:
    answer: str | None = None  # the model's final text, if it finished
    gave_up: str | None = None  # why the loop stopped without an answer
    turns: int = 0  # model responses
    tool_calls: int = 0
    failures: dict[str, int] = field(default_factory=dict)  # error code -> count
    repair_turns: int = 0  # turns in which a call had to be repaired
    messages: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.gave_up is None and self.answer is not None


async def run_chat(
    prompt: str,
    *,
    router: ToolRouter,
    chat_fn: ChatFn,
    system: str | None = DEFAULT_SYSTEM,
    max_turns: int = 6,
    max_repairs: int = 2,
    model_timeout_s: float | None = 120.0,
) -> ChatReport:
    """Let a model use the router's tools until it answers.

    Stops when the model replies without tool calls (success), when it fails to fix its calls
    more than `max_repairs` turns in a row, or after `max_turns` model responses.
    Each model call is bounded by `model_timeout_s` (None disables it): a stalled or runaway
    generation raises ModelTimeoutError instead of hanging. Other errors raised by `chat_fn`
    propagate to the caller.
    """
    report = ChatReport()
    messages: list[Any] = report.messages
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    tools = fmt.tools(router.registry)

    in_a_row = 0
    for _ in range(max_turns):
        deadline = asyncio.timeout(model_timeout_s)
        try:
            async with deadline:
                message = await chat_fn(messages, tools)
        except TimeoutError:
            if deadline.expired() and model_timeout_s is not None:
                raise ModelTimeoutError(model_timeout_s) from None
            raise
        report.turns += 1
        messages.append(message)
        calls = fmt.parse_calls(message)
        if not calls:
            report.answer = str(get(message, "content") or "")
            return report

        results = await router.dispatch_many(calls)
        report.tool_calls += len(calls)
        for result in results:
            if result.error is not None:
                code = result.error.code.value
                report.failures[code] = report.failures.get(code, 0) + 1
        messages.extend(fmt.result_messages(results))

        if any(r.error is not None and r.error.code in REPAIRABLE for r in results):
            report.repair_turns += 1
            in_a_row += 1
            if in_a_row > max_repairs:
                report.gave_up = f"could not fix its tool calls after {max_repairs} repair attempts"
                return report
        else:
            in_a_row = 0

    report.gave_up = f"no final answer within {max_turns} turns"
    return report
