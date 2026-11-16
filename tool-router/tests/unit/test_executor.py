import asyncio
import contextvars
import threading
import time
from typing import Any

import pytest

from toolrouter import ErrorCode, Tool, ToolCall, ToolRegistry, ToolResult, ToolRouter, tool
from toolrouter.executor import ToolExecutor


def router_for(*tools: Tool, **kw: Any) -> ToolRouter:
    return ToolRouter(ToolRegistry(tools), **kw)


async def dispatch(router: ToolRouter, name: str, args: Any = "{}") -> ToolResult:
    return await router.dispatch(ToolCall(id="c", name=name, arguments=args))


async def wait_until(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout
    while not predicate():
        assert time.perf_counter() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


# ---- sync handlers run off the loop ----------------------------------------------------------


async def test_sync_handler_runs_in_a_worker_thread() -> None:
    seen: list[int] = []

    @tool
    def work() -> str:
        """Record the thread."""
        seen.append(threading.get_ident())
        return "x"

    r = await dispatch(router_for(work), "work")
    assert r.ok
    assert seen[0] != threading.get_ident()


async def test_event_loop_stays_responsive_during_a_blocking_sync_handler() -> None:
    @tool
    def block() -> str:
        """Block the thread."""
        time.sleep(0.3)
        return "done"

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    r = await dispatch(router_for(block), "block")
    task.cancel()
    assert r.content == "done"
    assert ticks >= 10  # the loop kept turning while the handler blocked


async def test_sync_handlers_run_in_parallel() -> None:
    @tool
    def nap(i: int) -> int:
        """Sleep a bit."""
        time.sleep(0.2)
        return i

    router = router_for(nap)
    start = time.perf_counter()
    results = await asyncio.gather(*(dispatch(router, "nap", {"i": i}) for i in range(10)))
    assert time.perf_counter() - start < 1.0  # serial would take 2s
    assert [r.content for r in results] == [str(i) for i in range(10)]


async def test_contextvars_reach_sync_handlers() -> None:
    request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="none")

    @tool
    def whoami() -> str:
        """Read the context."""
        return request_id.get()

    request_id.set("req-42")
    assert (await dispatch(router_for(whoami), "whoami")).content == "req-42"


async def test_sync_function_returning_a_coroutine_is_awaited() -> None:
    @tool
    def returns_coro() -> int:
        """Sync function that returns an awaitable."""

        async def inner() -> int:
            return 5

        return inner()  # type: ignore[return-value]

    assert (await dispatch(router_for(returns_coro), "returns_coro")).content == "5"


# ---- timeouts ---------------------------------------------------------------------------------


async def test_async_handler_timeout_cancels_the_handler() -> None:
    cleaned: list[str] = []

    @tool(timeout_s=0.05)
    async def hang() -> str:
        """Hang forever."""
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append("cleanup ran")
        return "never"

    start = time.perf_counter()
    r = await dispatch(router_for(hang), "hang")
    assert time.perf_counter() - start < 1.0
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_TIMEOUT
    assert r.error.retryable
    assert "0.05s" in r.error.message
    assert cleaned == ["cleanup ran"]  # cooperative cancellation reached the handler


async def test_sync_handler_timeout_returns_on_time_and_tracks_the_stuck_thread() -> None:
    started, release = threading.Event(), threading.Event()
    finished: list[str] = []

    @tool(timeout_s=0.05)
    def stuck() -> str:
        """Block until released."""
        started.set()
        release.wait(5)
        finished.append("done")
        return "late"

    router = router_for(stuck)
    start = time.perf_counter()
    r = await dispatch(router, "stuck")
    assert time.perf_counter() - start < 1.0  # returned on time although the thread is stuck
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_TIMEOUT
    assert started.is_set()
    assert router.executor.active_threads == 1
    assert router.executor.abandoned_threads == 1

    release.set()
    await wait_until(lambda: router.executor.active_threads == 0)
    assert router.executor.abandoned_threads == 0  # accounting settles when the thread ends
    assert finished == ["done"]  # it ran to completion; its late result was discarded


async def test_per_tool_timeouts_differ() -> None:
    @tool(timeout_s=0.05)
    async def quick_deadline() -> str:
        """Short deadline."""
        await asyncio.sleep(0.2)
        return "x"

    @tool(timeout_s=2)
    async def long_deadline() -> str:
        """Long deadline."""
        await asyncio.sleep(0.2)
        return "x"

    router = router_for(quick_deadline, long_deadline)
    assert (await dispatch(router, "quick_deadline")).error is not None
    assert (await dispatch(router, "long_deadline")).ok


@pytest.mark.parametrize("is_async", [False, True])
async def test_a_handlers_own_timeout_error_is_not_our_timeout(is_async: bool) -> None:
    if is_async:

        @tool
        async def flaky() -> str:
            """Raises TimeoutError itself (e.g. a socket)."""
            raise TimeoutError("connect timed out to 10.0.0.5")

    else:

        @tool
        def flaky() -> str:
            """Raises TimeoutError itself (e.g. a socket)."""
            raise TimeoutError("connect timed out to 10.0.0.5")

    r = await dispatch(router_for(flaky), "flaky")
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert "TimeoutError" in r.error.message
    assert "10.0.0.5" not in r.error.message


# ---- saturation -------------------------------------------------------------------------------


async def test_saturated_executor_fails_fast_and_never_runs_the_rejected_call() -> None:
    started, release = threading.Event(), threading.Event()
    ran: list[str] = []

    @tool(timeout_s=10)
    def hold() -> str:
        """Hold the only thread."""
        started.set()
        release.wait(5)
        return "held"

    @tool
    def quick() -> str:
        """Quick call."""
        ran.append("quick")
        return "q"

    router = router_for(hold, quick, max_threads=1)
    first = asyncio.create_task(dispatch(router, "hold"))
    await wait_until(started.is_set)

    start = time.perf_counter()
    r = await dispatch(router, "quick")
    assert time.perf_counter() - start < 0.5  # no queueing
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert r.error.retryable
    assert "Too many tool calls" in r.error.message
    assert ran == []

    release.set()
    assert (await first).content == "held"
    await wait_until(lambda: router.executor.active_threads == 0)
    assert (await dispatch(router, "quick")).content == "q"  # capacity is back


def test_executor_validates_max_threads() -> None:
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ToolExecutor(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ToolRouter(ToolRegistry(), max_threads=0)


# ---- cancellation of the caller ---------------------------------------------------------------


async def test_cancelling_the_caller_cancels_an_async_handler_and_propagates() -> None:
    cleaned: list[str] = []
    started = asyncio.Event()

    @tool(timeout_s=30)
    async def sleeper() -> str:
        """Sleep a long time."""
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append("cleanup ran")
        return "never"

    task = asyncio.create_task(dispatch(router_for(sleeper), "sleeper"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["cleanup ran"]


async def test_cancelling_the_caller_abandons_a_sync_handler_thread() -> None:
    started, release = threading.Event(), threading.Event()
    runs: list[str] = []

    @tool(timeout_s=30)
    def blocker() -> str:
        """Block until released."""
        runs.append("started")
        started.set()
        release.wait(5)
        return "late"

    router = router_for(blocker)
    task = asyncio.create_task(dispatch(router, "blocker"))
    await wait_until(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert router.executor.abandoned_threads == 1

    release.set()
    await wait_until(lambda: router.executor.active_threads == 0)
    assert router.executor.abandoned_threads == 0
    assert runs == ["started"]  # exactly one run; nothing re-ran it


async def test_a_timeout_does_not_leak_into_the_next_call() -> None:
    @tool(timeout_s=0.05)
    async def slow() -> str:
        """Too slow."""
        await asyncio.sleep(1)
        return "x"

    @tool
    async def fast() -> str:
        """Fast."""
        return "ok"

    router = router_for(slow, fast)
    assert (await dispatch(router, "slow")).error is not None
    assert (await dispatch(router, "fast")).content == "ok"
    task = asyncio.current_task()
    assert task is not None
    assert task.cancelling() == 0  # our own timeout was fully unwound


# ---- BaseExceptions in worker threads ------------------------------------------------------------


@pytest.mark.parametrize("exc", [SystemExit(1), KeyboardInterrupt()])
async def test_sync_handler_raising_baseexception_is_just_a_failed_call(
    exc: BaseException,
) -> None:
    @tool
    def rude() -> str:
        """Tries to exit."""
        raise exc

    r = await dispatch(router_for(rude), "rude")
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert "tried to terminate the process" in r.error.message
    assert type(exc).__name__ in r.error.message
    assert r.cause is exc


async def test_late_exception_from_an_abandoned_thread_is_swallowed() -> None:
    release = threading.Event()

    @tool(timeout_s=0.05)
    def explodes_late() -> str:
        """Fails after the caller gave up."""
        release.wait(5)
        raise RuntimeError("too late to matter")

    router = router_for(explodes_late)
    r = await dispatch(router, "explodes_late")
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_TIMEOUT
    release.set()
    await wait_until(lambda: router.executor.active_threads == 0)
    await asyncio.sleep(0.05)  # let any stray callbacks run; nothing may blow up the loop
