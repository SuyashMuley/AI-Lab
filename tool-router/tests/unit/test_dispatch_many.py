import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from toolrouter import (
    ErrorCode,
    SideEffect,
    Tool,
    ToolCall,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolRouter,
    tool,
)


class Gauge:
    """Counts how many handlers are inside `hold()` at once (thread-safe)."""

    def __init__(self) -> None:
        self.now = 0
        self.max = 0
        self._lock = threading.Lock()

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._lock:
            self.now += 1
            self.max = max(self.max, self.now)
        try:
            yield
        finally:
            with self._lock:
                self.now -= 1


def c(name: str, args: Any = "{}", id_: str | None = None) -> ToolCall:
    return ToolCall(id=id_ or f"id-{name}", name=name, arguments=args)


def router_for(*tools: Tool, **kw: Any) -> ToolRouter:
    return ToolRouter(ToolRegistry(tools), **kw)


async def wait_until(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout
    while not predicate():
        assert time.perf_counter() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


# ---- ordering and basics ---------------------------------------------------------------------


async def test_results_follow_call_order_even_when_completion_order_is_reversed() -> None:
    finished: list[int] = []

    @tool
    async def work(i: int) -> int:
        """Later calls finish first."""
        await asyncio.sleep((6 - i) * 0.03)
        finished.append(i)
        return i * 10

    calls = [c("work", {"i": i}, id_=f"c{i}") for i in range(6)]
    results = await router_for(work, max_concurrency=6).dispatch_many(calls)
    assert finished != sorted(finished)  # the work really did complete out of order
    assert [r.call_id for r in results] == [f"c{i}" for i in range(6)]
    assert [r.content for r in results] == [str(i * 10) for i in range(6)]


async def test_empty_batch_and_non_list_iterables() -> None:
    @tool
    def one() -> int:
        """One."""
        return 1

    router = router_for(one)
    assert await router.dispatch_many([]) == []
    assert [r.content for r in await router.dispatch_many((c("one"), c("one")))] == ["1", "1"]
    assert [r.content for r in await router.dispatch_many(c("one") for _ in range(3))] == ["1"] * 3


async def test_duplicate_call_ids_are_kept_positionally() -> None:
    @tool
    def echo(x: int) -> int:
        """Echo."""
        return x

    results = await router_for(echo).dispatch_many(
        [c("echo", {"x": 1}, id_="same"), c("echo", {"x": 2}, id_="same")]
    )
    assert [(r.call_id, r.content) for r in results] == [("same", "1"), ("same", "2")]


# ---- concurrency bound -------------------------------------------------------------------------


@pytest.mark.parametrize(("limit", "n"), [(1, 6), (2, 9), (4, 20), (10, 20), (50, 5)])
async def test_max_concurrency_is_honored_and_reached(limit: int, n: int) -> None:
    gauge = Gauge()

    @tool
    async def work(i: int) -> int:
        """Track in-flight calls."""
        with gauge.hold():
            await asyncio.sleep(0.03)
        return i

    results = await router_for(work).dispatch_many(
        [c("work", {"i": i}, id_=str(i)) for i in range(n)], max_concurrency=limit
    )
    assert all(r.ok for r in results)
    assert gauge.max == min(limit, n)  # never above the bound, and actually used


async def test_router_default_and_per_call_override() -> None:
    gauge = Gauge()

    @tool
    async def work() -> int:
        """Track in-flight calls."""
        with gauge.hold():
            await asyncio.sleep(0.03)
        return 1

    router = router_for(work, max_concurrency=2)
    await router.dispatch_many([c("work", id_=str(i)) for i in range(8)])
    assert gauge.max == 2
    gauge.max = 0
    await router.dispatch_many([c("work", id_=str(i)) for i in range(8)], max_concurrency=5)
    assert gauge.max == 5


async def test_concurrency_one_runs_in_call_order() -> None:
    order: list[int] = []

    @tool
    async def work(i: int) -> int:
        """Record execution order."""
        order.append(i)
        await asyncio.sleep(0.005)
        return i

    await router_for(work).dispatch_many(
        [c("work", {"i": i}, id_=str(i)) for i in range(8)], max_concurrency=1
    )
    assert order == list(range(8))


async def test_a_huge_batch_costs_a_few_tasks_not_one_per_call() -> None:
    peak_tasks = 0

    @tool
    async def tiny(i: int) -> int:
        """Quick."""
        nonlocal peak_tasks
        peak_tasks = max(peak_tasks, len(asyncio.all_tasks()))
        await asyncio.sleep(0)
        return i

    n = 2000
    results = await router_for(tiny).dispatch_many(
        [c("tiny", {"i": i}, id_=str(i)) for i in range(n)], max_concurrency=4
    )
    assert [r.content for r in results] == [str(i) for i in range(n)]
    assert peak_tasks <= 4 + 2  # 4 workers + the test task (+ slack), not 2000


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "4"])
async def test_invalid_concurrency_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        ToolRouter(ToolRegistry(), max_concurrency=bad)
    with pytest.raises(ValueError, match="max_concurrency"):
        await ToolRouter(ToolRegistry()).dispatch_many([c("x")], max_concurrency=bad)


# ---- failure isolation -----------------------------------------------------------------------


async def test_one_failure_never_affects_the_others() -> None:
    @tool
    def ok(x: int) -> int:
        """Fine."""
        return x

    @tool
    def expected() -> str:
        """Expected failure."""
        raise ToolError("nope, by design")

    @tool
    def crash() -> str:
        """Unexpected failure."""
        raise KeyError("secret")

    @tool(timeout_s=0.05)
    async def hang() -> str:
        """Times out."""
        await asyncio.sleep(10)
        return "never"

    @tool
    async def spurious() -> str:
        """Handler-internal cancellation."""
        raise asyncio.CancelledError

    @tool
    def bad_result() -> int:
        """Wrong result type."""
        return "oops"  # type: ignore[return-value]

    calls = [
        c("ok", {"x": 1}, "0"),
        c("nope", "{}", "1"),  # unknown tool
        c("ok", '{"x": ', "2"),  # invalid JSON
        c("ok", {"x": "a"}, "3"),  # invalid arguments
        c("expected", "{}", "4"),
        c("crash", "{}", "5"),
        c("hang", "{}", "6"),
        c("spurious", "{}", "7"),
        c("bad_result", "{}", "8"),
        c("ok", {"x": 9}, "9"),
    ]
    router = router_for(ok, expected, crash, hang, spurious, bad_result)
    results = await router.dispatch_many(calls, max_concurrency=3)

    assert [r.call_id for r in results] == [str(i) for i in range(10)]
    codes = [r.error.code if r.error else None for r in results]
    assert codes == [
        None,
        ErrorCode.UNKNOWN_TOOL,
        ErrorCode.INVALID_JSON,
        ErrorCode.ARGUMENTS_INVALID,
        ErrorCode.TOOL_FAILED,
        ErrorCode.TOOL_FAILED,
        ErrorCode.TOOL_TIMEOUT,
        ErrorCode.CANCELLED,
        ErrorCode.RESULT_INVALID,
        None,
    ]
    assert results[0].content == "1"
    assert results[9].content == "9"  # the last call still ran after all the failures


async def test_a_hung_tool_only_costs_its_own_slot() -> None:
    done_at: dict[str, float] = {}
    start = time.perf_counter()

    @tool(timeout_s=0.4)
    async def hang() -> str:
        """Hangs until its timeout."""
        await asyncio.sleep(10)
        return "never"

    @tool
    async def quick(i: int) -> int:
        """Quick call."""
        await asyncio.sleep(0.01)
        done_at[f"q{i}"] = time.perf_counter() - start
        return i

    calls = [c("hang", "{}", "h")] + [c("quick", {"i": i}, f"q{i}") for i in range(8)]
    results = await router_for(hang, quick).dispatch_many(calls, max_concurrency=2)
    assert results[0].error is not None
    assert results[0].error.code is ErrorCode.TOOL_TIMEOUT
    assert all(r.ok for r in results[1:])
    assert max(done_at.values()) < 0.3  # all quick calls finished while `hang` was still stuck


async def test_sync_handlers_in_a_batch_run_in_parallel_threads() -> None:
    @tool
    def nap(i: int) -> int:
        """Sleep a bit."""
        time.sleep(0.2)
        return i

    start = time.perf_counter()
    results = await router_for(nap).dispatch_many(
        [c("nap", {"i": i}, str(i)) for i in range(8)], max_concurrency=8
    )
    assert time.perf_counter() - start < 1.0  # serial would be 1.6s
    assert [r.content for r in results] == [str(i) for i in range(8)]


async def test_a_broken_registry_yields_results_not_exceptions() -> None:
    class Broken(ToolRegistry):
        def get(self, name: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("registry exploded at /private/path")

    results = await ToolRouter(Broken()).dispatch_many([c("a"), c("b")])
    assert [r.error.code for r in results if r.error] == [ErrorCode.INTERNAL_ERROR] * 2
    assert all("/private/path" not in r.text for r in results)


# ---- write serialization -----------------------------------------------------------------------


def write_and_read_tools() -> tuple[Tool, Tool, Gauge, Gauge, list[str]]:
    writes, reads = Gauge(), Gauge()
    write_order: list[str] = []

    @tool(side_effect=SideEffect.WRITE)
    async def write(name: str) -> str:
        """A WRITE tool."""
        with writes.hold():
            write_order.append(name)
            await asyncio.sleep(0.03)
        return name

    @tool(side_effect=SideEffect.READ)
    async def read(name: str) -> str:
        """A READ tool."""
        with reads.hold():
            await asyncio.sleep(0.03)
        return name

    return write, read, writes, reads, write_order


async def test_writes_are_serialized_in_call_order_while_reads_run_in_parallel() -> None:
    write, read, writes, reads, write_order = write_and_read_tools()
    calls = []
    for i in range(4):
        calls.append(c("write", {"name": f"w{i}"}, f"w{i}"))
        calls.append(c("read", {"name": f"r{i}"}, f"r{i}"))
    results = await router_for(write, read).dispatch_many(calls, max_concurrency=8)

    assert all(r.ok for r in results)
    assert writes.max == 1  # never two writes at once
    assert write_order == ["w0", "w1", "w2", "w3"]  # and in the order the model listed them
    assert reads.max > 1  # reads were not slowed down
    assert [r.call_id for r in results] == [x.id for x in calls]


async def test_write_serialization_can_be_switched_off() -> None:
    write, read, writes, _, _ = write_and_read_tools()
    calls = [c("write", {"name": f"w{i}"}, f"w{i}") for i in range(4)]
    await router_for(write, read).dispatch_many(calls, max_concurrency=4, serialize_writes=False)
    assert writes.max > 1

    writes.max = 0
    router = router_for(write, read, serialize_writes=False)
    await router.dispatch_many(calls, max_concurrency=4)
    assert writes.max > 1  # router-level default honored
    writes.max = 0
    await router.dispatch_many(calls, max_concurrency=4, serialize_writes=True)  # per-call wins
    assert writes.max == 1


async def test_a_failing_write_does_not_block_the_next_write() -> None:
    order: list[str] = []

    @tool(side_effect=SideEffect.WRITE)
    async def write(name: str) -> str:
        """Fails for 'bad'."""
        order.append(name)
        if name == "bad":
            raise ToolError("disk full")
        return name

    results = await router_for(write).dispatch_many(
        [c("write", {"name": n}, n) for n in ("a", "bad", "b")], max_concurrency=3
    )
    assert order == ["a", "bad", "b"]
    assert [r.ok for r in results] == [True, False, True]


# ---- cancellation ------------------------------------------------------------------------------


async def test_cancelling_the_caller_cancels_running_handlers_and_starts_nothing_new() -> None:
    started: list[str] = []
    cleaned: list[str] = []

    @tool(timeout_s=30)
    async def slow(i: int) -> int:
        """Slow call."""
        started.append(str(i))
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append(str(i))
        return i

    router = router_for(slow)
    task = asyncio.create_task(
        router.dispatch_many([c("slow", {"i": i}, str(i)) for i in range(10)], max_concurrency=3)
    )
    await wait_until(lambda: len(started) == 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(cleaned) == sorted(started)  # every running handler was cancelled and cleaned up
    assert len(started) == 3  # the other 7 never began
    await asyncio.sleep(0.05)
    assert asyncio.all_tasks() == {asyncio.current_task()}  # no orphan tasks


async def test_cancelling_the_caller_abandons_running_sync_threads() -> None:
    release = threading.Event()
    started: list[int] = []

    @tool(timeout_s=30)
    def block(i: int) -> int:
        """Block until released."""
        started.append(i)
        release.wait(5)
        return i

    router = router_for(block)
    task = asyncio.create_task(
        router.dispatch_many([c("block", {"i": i}, str(i)) for i in range(6)], max_concurrency=3)
    )
    await wait_until(lambda: len(started) == 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert router.executor.abandoned_threads == 3

    release.set()
    await wait_until(lambda: router.executor.active_threads == 0)
    assert router.executor.abandoned_threads == 0
    assert len(started) == 3  # queued calls never ran, even after the threads were released


def test_keyboard_interrupt_in_one_call_propagates_unwrapped_and_cancels_siblings() -> None:
    cleaned: list[str] = []

    @tool
    async def slow() -> str:
        """Sibling that should be cancelled."""
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append("slow")
        return "never"

    @tool
    async def interrupt() -> str:
        """Raises KeyboardInterrupt on the loop."""
        await asyncio.sleep(0.05)
        raise KeyboardInterrupt

    async def go() -> list[ToolResult]:
        return await router_for(slow, interrupt).dispatch_many(
            [c("slow", id_="s"), c("interrupt", id_="i")], max_concurrency=2
        )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(go())
    assert cleaned == ["slow"]


def test_system_exit_propagates_unwrapped() -> None:
    @tool
    async def leave() -> str:
        """Raises SystemExit on the loop."""
        raise SystemExit(7)

    async def go() -> list[ToolResult]:
        return await router_for(leave).dispatch_many([c("leave")])

    with pytest.raises(SystemExit) as ei:
        asyncio.run(go())
    assert ei.value.code == 7


async def test_a_timeout_inside_a_batch_does_not_cancel_the_batch() -> None:
    @tool(timeout_s=0.05)
    async def hang() -> str:
        """Times out."""
        await asyncio.sleep(10)
        return "never"

    @tool
    async def after(i: int) -> int:
        """Runs after the timeout."""
        await asyncio.sleep(0.02)
        return i

    results = await router_for(hang, after).dispatch_many(
        [c("hang", id_="h"), c("after", {"i": 1}, "a1"), c("after", {"i": 2}, "a2")],
        max_concurrency=1,
    )
    assert [r.ok for r in results] == [False, True, True]
    task = asyncio.current_task()
    assert task is not None
    assert task.cancelling() == 0
