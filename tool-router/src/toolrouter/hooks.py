"""Hooks: observe every dispatch (audit, logging, metrics) without being able to break one.

Every `dispatch` fires exactly one `before` and one `after`, whatever the outcome: success,
unknown tool, invalid arguments, policy denial, timeout, tool failure, or the caller being
cancelled. Hooks run in registration order, may be sync or async, and are contained: a hook that
raises or exceeds the router's `hook_timeout_s` is logged and skipped; it never changes the result.
Hooks may be called concurrently for different calls (e.g. in `dispatch_many`).

Events carry **redacted** arguments: secret fields are replaced, unknown tools and unparseable
arguments are omitted entirely (we cannot know what is secret), and everything is size-bounded.
`raw_call` is the unredacted call; do not log it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from toolrouter.calls import ToolCall, ToolCallError, ToolResult
from toolrouter.redact import bound, redact_value
from toolrouter.registry import ToolRegistry
from toolrouter.schema import redaction_schema
from toolrouter.tool import Tool
from toolrouter.validate import DEFAULT_LIMITS, Limits, arguments_text


@dataclass(frozen=True)
class CallEvent:
    raw_call: ToolCall  # UNREDACTED: may contain secrets
    tool: Tool | None  # None when the name is unknown
    arguments: Any  # redacted, bounded, JSON-compatible (or an "<arguments omitted: ...>" note)

    @property
    def call_id(self) -> str:
        return self.raw_call.id

    @property
    def name(self) -> str:
        return self.raw_call.name


@dataclass(frozen=True)
class ResultEvent:
    call: CallEvent
    result: ToolResult


class RouterHook:
    """Base class: override `before` and/or `after`, sync or async."""

    def before(self, event: CallEvent) -> Awaitable[None] | None:
        return None

    def after(self, event: ResultEvent) -> Awaitable[None] | None:
        return None


def redacted_arguments(tool: Tool | None, raw: object, limits: Limits = DEFAULT_LIMITS) -> Any:
    """Arguments safe to show in logs, or a note explaining why they were omitted. Never raises."""
    try:
        if tool is None:
            return "<arguments omitted: unknown tool>"
        try:
            text = arguments_text(raw, limits)
        except ToolCallError as exc:
            return f"<arguments omitted: {exc.info.code.value}>"
        return bound(redact_value(json.loads(text), redaction_schema(tool)))
    except Exception:
        return "<arguments omitted: could not be redacted>"


def build_call_event(
    registry: ToolRegistry, call: ToolCall, limits: Limits = DEFAULT_LIMITS
) -> CallEvent:
    """Never raises: a broken registry still yields an event (with the tool unknown)."""
    try:
        tool = registry.get(call.name)
    except Exception:
        tool = None
    return CallEvent(call, tool, redacted_arguments(tool, call.arguments, limits))


class RecordingHook(RouterHook):
    """Keeps every event in order; handy in tests and for in-process audit trails."""

    def __init__(self) -> None:
        self.before_events: list[CallEvent] = []
        self.after_events: list[ResultEvent] = []

    def before(self, event: CallEvent) -> None:
        self.before_events.append(event)

    def after(self, event: ResultEvent) -> None:
        self.after_events.append(event)


class LoggingHook(RouterHook):
    """One log line per call with redacted, bounded arguments.

    Successes log at `level`, failures at `failure_level`. Exception details stay out unless
    `include_tracebacks=True`: library errors often embed connection strings or paths.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
        failure_level: int = logging.WARNING,
        include_tracebacks: bool = False,
    ) -> None:
        self.logger = logger or logging.getLogger("toolrouter")
        self.level = level
        self.failure_level = failure_level
        self.include_tracebacks = include_tracebacks

    @staticmethod
    def _args(event: CallEvent) -> str:
        return json.dumps(event.arguments, ensure_ascii=False, default=repr, sort_keys=True)

    def before(self, event: CallEvent) -> None:
        self.logger.debug(
            "tool call started name=%s id=%s args=%s", event.name, event.call_id, self._args(event)
        )

    def after(self, event: ResultEvent) -> None:
        r = event.result
        parts = [f"name={r.name}", f"id={r.call_id}", f"ok={str(r.ok).lower()}"]
        parts.append(f"duration_ms={r.duration_ms:.1f}")
        if r.truncated:
            parts.append("truncated=true")
        if r.error is not None:
            parts.append(f"error={r.error.code.value}")
            parts.append(f"message={json.dumps(r.error.message[:200], ensure_ascii=False)}")
        if r.cause is not None:
            parts.append(f"cause={type(r.cause).__name__}")
        exc_info = None
        if self.include_tracebacks and r.cause is not None:
            exc_info = (type(r.cause), r.cause, r.cause.__traceback__)
        self.logger.log(
            self.level if r.ok else self.failure_level,
            "tool call %s args=%s",
            " ".join(parts),
            self._args(event.call),
            exc_info=exc_info,
        )
