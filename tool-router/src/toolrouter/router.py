"""ToolRouter: one call in, one ToolResult out. Never raises for anything a model can cause."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from typing import cast

from toolrouter.calls import (
    ErrorInfo,
    ToolCall,
    ToolCallError,
    ToolResult,
    make_error,
)
from toolrouter.errors import ErrorCode, ToolError
from toolrouter.executor import ToolBusyError, ToolExecutor, ToolTimeoutError, WorkerExit
from toolrouter.registry import ToolRegistry
from toolrouter.results import serialize_result, truncate
from toolrouter.tool import SideEffect
from toolrouter.validate import DEFAULT_LIMITS, Limits, validate_call

_MAX_LISTED_TOOLS = 20
_MAX_ECHOED_NAME = 64
_MAX_TOOL_ERROR_CHARS = 500


def _check_concurrency(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_concurrency must be a positive int")
    return value


def _tool_error_message(exc: ToolError) -> str:
    """The model-facing text of a ToolError: bounded, never raising, never empty."""
    try:
        text = str(exc).strip()
    except Exception:  # a hostile __str__ must not break the router
        text = ""
    return text[:_MAX_TOOL_ERROR_CHARS] or "The tool reported a failure."


class ToolRouter:
    """Routes tool calls to handlers.

    Pipeline per call: look up the tool -> parse and strictly validate the arguments -> run the
    handler -> validate and serialize the result. A failure at any stage becomes a
    `ToolResult(ok=False)`; the handler only ever sees validated arguments.

    Execution is bounded: each tool has a timeout, sync handlers run in worker threads (never on
    the event loop), and output is capped at the tool's `max_output_chars`. See `ToolExecutor`
    for what a timeout can and cannot stop.

    Cancellation: if the *calling task* is cancelled, `CancelledError` propagates (asyncio
    requires it). A `CancelledError` raised from inside a handler while the caller was not
    cancelled is a failure of that tool, reported as `CANCELLED`. `KeyboardInterrupt` and
    `SystemExit` raised on the event loop propagate; raised in a sync handler's thread they are
    just a failed tool call.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        coerce: bool = False,
        limits: Limits = DEFAULT_LIMITS,
        clock: Callable[[], float] = time.perf_counter,
        max_threads: int = 32,
        executor: ToolExecutor | None = None,
        max_concurrency: int = 4,
        serialize_writes: bool = True,
    ) -> None:
        self.registry = registry
        self.max_concurrency = _check_concurrency(max_concurrency)
        self.serialize_writes = serialize_writes
        self.executor = executor if executor is not None else ToolExecutor(max_threads)
        self.coerce = coerce  # router default; see resolve_coerce for the per-tool rules
        self.limits = limits
        self._clock = clock

    async def dispatch(self, call: ToolCall) -> ToolResult:
        start = self._clock()

        def fail(info: ErrorInfo, cause: BaseException | None = None) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=info,
                duration_ms=(self._clock() - start) * 1000,
                cause=cause,
            )

        try:
            tool = self.registry.get(call.name)
            if tool is None:
                return fail(self._unknown_tool(call.name))

            try:
                args = validate_call(
                    tool, call.arguments, router_coerce=self.coerce, limits=self.limits
                )
            except ToolCallError as exc:
                return fail(exc.info, exc.__cause__)

            try:
                value = await self.executor.run(tool, args)
            except ToolTimeoutError as exc:
                return fail(
                    make_error(
                        ErrorCode.TOOL_TIMEOUT,
                        f"Tool {tool.name!r} did not finish within {tool.timeout_s:g}s.",
                    ),
                    exc,
                )
            except ToolBusyError as exc:
                return fail(
                    make_error(
                        ErrorCode.TOOL_FAILED,
                        "Too many tool calls are already running; try again in a moment.",
                    ),
                    exc,
                )
            except WorkerExit as exc:
                return fail(
                    make_error(
                        ErrorCode.TOOL_FAILED,
                        f"Tool {tool.name!r} tried to terminate the process "
                        f"({type(exc.original).__name__}).",
                    ),
                    exc.original,
                )
            except ToolError as exc:
                return fail(make_error(ErrorCode.TOOL_FAILED, _tool_error_message(exc)), exc)
            except asyncio.CancelledError as exc:
                task = asyncio.current_task()
                if task is None or task.cancelling() > 0:
                    raise  # the caller really is being cancelled: let it unwind
                return fail(
                    make_error(
                        ErrorCode.CANCELLED, f"Tool {tool.name!r} was cancelled before it finished."
                    ),
                    exc,
                )
            except Exception as exc:
                return fail(
                    make_error(
                        ErrorCode.TOOL_FAILED,
                        f"Tool {tool.name!r} failed unexpectedly ({type(exc).__name__}).",
                    ),
                    exc,
                )

            try:
                content = serialize_result(tool, value)
            except ToolCallError as exc:
                return fail(exc.info, exc.__cause__)

            content, truncated = truncate(content, tool.max_output_chars)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=True,
                content=content,
                truncated=truncated,
                duration_ms=(self._clock() - start) * 1000,
            )
        except Exception as exc:  # a bug in the router itself; still never raise to the caller
            return fail(
                make_error(ErrorCode.INTERNAL_ERROR, "The tool router hit an internal error."),
                exc,
            )

    async def dispatch_many(
        self,
        calls: Iterable[ToolCall],
        *,
        max_concurrency: int | None = None,
        serialize_writes: bool | None = None,
    ) -> list[ToolResult]:
        """Dispatch several calls concurrently; returns one result per call, in call order.

        - At most `max_concurrency` calls are in flight (default: the router's setting). Work is
          done by that many workers pulling from the batch, so a huge batch costs few tasks.
        - Calls are independent: a failing or timed-out call never affects the others.
        - `serialize_writes` (default: the router's setting, on): calls to WRITE tools run one at
          a time and in call order, while READ/NONE calls still run in parallel. A waiting WRITE
          call holds its concurrency slot.
        - If the caller is cancelled, running handlers are cancelled (sync threads become
          abandoned, see ToolExecutor) and `CancelledError` propagates; nothing keeps running.
        - `KeyboardInterrupt`/`SystemExit` raised on the event loop by a handler propagate as-is.
        """
        batch = list(calls)
        if not batch:
            return []
        limit = _check_concurrency(
            self.max_concurrency if max_concurrency is None else max_concurrency
        )
        serialize = self.serialize_writes if serialize_writes is None else serialize_writes

        results: list[ToolResult | None] = [None] * len(batch)
        pending = iter(range(len(batch)))  # shared: each worker takes the next index
        write_lock = asyncio.Lock() if serialize else None

        async def worker() -> None:
            for i in pending:
                results[i] = await self._dispatch_in_batch(batch[i], write_lock)

        # TaskGroup cancels the other workers if one is cancelled or raises, and re-raises
        # KeyboardInterrupt/SystemExit directly (not wrapped in an exception group).
        async with asyncio.TaskGroup() as group:
            for _ in range(min(limit, len(batch))):
                group.create_task(worker())
        return cast(list[ToolResult], results)

    async def _dispatch_in_batch(
        self, call: ToolCall, write_lock: asyncio.Lock | None
    ) -> ToolResult:
        if write_lock is not None and self._is_write(call.name):
            async with write_lock:
                return await self.dispatch(call)
        return await self.dispatch(call)

    def _is_write(self, name: str) -> bool:
        try:
            tool = self.registry.get(name)
        except Exception:  # let dispatch() report the broken registry as INTERNAL_ERROR
            return False
        return tool is not None and tool.side_effect is SideEffect.WRITE

    def _unknown_tool(self, name: str) -> ErrorInfo:
        shown = name[:_MAX_ECHOED_NAME] + ("…" if len(name) > _MAX_ECHOED_NAME else "")
        names = self.registry.names()
        listed = ", ".join(names[:_MAX_LISTED_TOOLS])
        more = (
            f" (+{len(names) - _MAX_LISTED_TOOLS} more)" if len(names) > _MAX_LISTED_TOOLS else ""
        )
        available = f"Available tools: {listed}{more}." if names else "No tools are available."
        return make_error(ErrorCode.UNKNOWN_TOOL, f"Unknown tool {shown!r}. {available}")
