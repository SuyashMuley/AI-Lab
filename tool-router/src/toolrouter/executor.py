"""Run tool handlers with timeouts, thread offload for sync code, and honest cancellation.

Async handlers run on the event loop under `asyncio.timeout`. Sync handlers run in their own
daemon thread (never on the loop) with the caller's contextvars:

- Threads are per call, not pooled, so a stuck handler can't wedge a shared pool, can't block
  interpreter exit (daemon), and a call that already timed out can never start later.
- Concurrency is bounded (`max_threads`). When saturated (e.g. by stuck handlers) new sync calls
  fail fast with ToolBusyError instead of queueing silently.
- A thread cannot be killed. On timeout or cancellation the caller stops waiting and the thread
  is counted as *abandoned* until it finishes; its eventual result is discarded. Real isolation
  for runaway code needs the sandboxed-execution project.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from typing import Any

from pydantic import BaseModel

from toolrouter.tool import Tool


class ToolTimeoutError(Exception):
    def __init__(self, name: str, timeout_s: float) -> None:
        super().__init__(f"Tool {name!r} did not finish within {timeout_s:g}s")
        self.timeout_s = timeout_s


class ToolBusyError(Exception):
    """Too many sync handlers are already running."""


class WorkerExit(Exception):
    """A sync handler raised a BaseException (SystemExit, KeyboardInterrupt, ...) in its thread."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(type(original).__name__)
        self.original = original


class _Job:
    def __init__(self) -> None:
        self.abandoned = threading.Event()  # set when the caller stopped waiting
        self.delivered = False  # loop thread only: outcome reached the awaiting future
        self.finished = False  # guarded by the executor lock: thread has exited
        self.counted = False  # guarded by the executor lock: included in abandoned_threads


class ToolExecutor:
    def __init__(self, max_threads: int = 32) -> None:
        if isinstance(max_threads, bool) or not isinstance(max_threads, int) or max_threads < 1:
            raise ValueError("max_threads must be a positive int")
        self.max_threads = max_threads
        self._lock = threading.Lock()
        self._active = 0
        self._abandoned = 0

    @property
    def active_threads(self) -> int:
        """Sync handler threads currently running (including abandoned ones)."""
        with self._lock:
            return self._active

    @property
    def abandoned_threads(self) -> int:
        """Threads still running although their caller timed out or was cancelled."""
        with self._lock:
            return self._abandoned

    async def run(self, tool: Tool, args: BaseModel) -> Any:
        """Run the handler; raises ToolTimeoutError, ToolBusyError, WorkerExit, or whatever the
        handler raised. A handler's own TimeoutError is *not* mistaken for our timeout."""
        deadline = asyncio.timeout(tool.timeout_s)
        try:
            async with deadline:
                if tool.is_async:
                    return await tool.invoke(args)
                value = await self._in_thread(tool, args)
                if inspect.isawaitable(value):  # a sync function that returns a coroutine
                    value = await value
                return value
        except TimeoutError:
            if deadline.expired():
                raise ToolTimeoutError(tool.name, tool.timeout_s) from None
            raise

    async def _in_thread(self, tool: Tool, args: BaseModel) -> Any:
        with self._lock:
            if self._active >= self.max_threads:
                raise ToolBusyError(f"{self._active} sync tool calls are already running")
            self._active += 1

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        job = _Job()
        call = tool.bind(args)
        ctx = contextvars.copy_context()

        def deliver(ok: bool, payload: Any) -> None:
            def settle() -> None:
                job.delivered = True
                if not fut.done():  # already cancelled/timed out: discard
                    fut.set_result(payload) if ok else fut.set_exception(payload)

            try:
                loop.call_soon_threadsafe(settle)
            except RuntimeError:  # loop already closed
                pass

        def target() -> None:
            try:
                if job.abandoned.is_set():  # caller left before we started: don't run at all
                    return
                try:
                    value = ctx.run(call)
                except BaseException as exc:
                    deliver(False, exc if isinstance(exc, Exception) else WorkerExit(exc))
                else:
                    deliver(True, value)
            finally:
                with self._lock:
                    job.finished = True
                    self._active -= 1
                    if job.counted:
                        self._abandoned -= 1

        thread = threading.Thread(target=target, name=f"toolrouter-{tool.name}", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            with self._lock:
                self._active -= 1
            raise ToolBusyError("could not start a worker thread") from None

        try:
            return await fut
        finally:
            if not job.delivered:  # timed out or cancelled: the thread outlives its caller
                job.abandoned.set()
                with self._lock:
                    if not job.finished:
                        job.counted = True
                        self._abandoned += 1
