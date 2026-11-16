import asyncio
import logging
import time
from typing import Any

import pytest

from tests.fakes import router_tools as rt
from toolrouter import (
    CallEvent,
    ErrorCode,
    LoggingHook,
    PolicyDecision,
    PolicyRequest,
    ReadOnlyPolicy,
    RecordingHook,
    ResultEvent,
    RouterHook,
    ToolCall,
    ToolRegistry,
    ToolRouter,
    tool,
)
from toolrouter import hooks as hooks_module
from toolrouter import router as router_module


@tool(timeout_s=0.05)
async def too_slow() -> str:
    """Times out."""
    await asyncio.sleep(10)
    return "never"


@tool
async def sleeper() -> str:
    """Sleeps until cancelled."""
    await asyncio.sleep(30)
    return "never"


def registry() -> ToolRegistry:
    return ToolRegistry(
        [rt.echo, rt.add, rt.crash, rt.expected_failure, rt.bad_int, rt.cancelled, rt.delete,
         too_slow, sleeper]
    )  # fmt: skip


def c(name: str, args: Any = "{}", id_: str = "c1") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


# ---- exactly one before and one after, for every outcome ------------------------------------

SCENARIOS: list[tuple[str, str, Any, ErrorCode | None]] = [
    ("success", "echo", {"text": "hi"}, None),
    ("unknown tool", "nope", "{}", ErrorCode.UNKNOWN_TOOL),
    ("invalid json", "echo", '{"text": ', ErrorCode.INVALID_JSON),
    ("too large", "echo", '{"text": "' + "x" * 70_000 + '"}', ErrorCode.ARGUMENTS_TOO_LARGE),
    ("invalid arguments", "echo", {"text": 5}, ErrorCode.ARGUMENTS_INVALID),
    ("tool error", "expected_failure", {"path": "a"}, ErrorCode.TOOL_FAILED),
    ("tool crash", "crash", {"path": "a"}, ErrorCode.TOOL_FAILED),
    ("bad result", "bad_int", {"x": 1}, ErrorCode.RESULT_INVALID),
    ("timeout", "too_slow", "{}", ErrorCode.TOOL_TIMEOUT),
    ("stray cancel", "cancelled", "{}", ErrorCode.CANCELLED),
]


@pytest.mark.parametrize(
    ("label", "name", "args", "code"), SCENARIOS, ids=[s[0] for s in SCENARIOS]
)
async def test_every_outcome_fires_one_before_and_one_after(
    label: str, name: str, args: Any, code: ErrorCode | None
) -> None:
    rec = RecordingHook()
    result = await ToolRouter(registry(), hooks=[rec]).dispatch(c(name, args))
    assert len(rec.before_events) == 1
    assert len(rec.after_events) == 1
    assert rec.before_events[0].call_id == "c1"
    assert rec.after_events[0].result is result
    assert rec.after_events[0].call is rec.before_events[0]  # the same event ties the pair
    assert (result.error.code if result.error else None) is code


async def test_policy_denial_and_policy_failure_also_fire_hooks() -> None:
    for policy_kind in ("deny", "raise"):
        rec = RecordingHook()

        class Raising:
            def check(self, request: PolicyRequest) -> PolicyDecision:
                raise RuntimeError("boom")

        policy: Any = ReadOnlyPolicy() if policy_kind == "deny" else Raising()
        router = ToolRouter(registry(), policy=policy, hooks=[rec])
        result = await router.dispatch(c("delete", {"n": 1}))
        assert not result.ok
        assert (len(rec.before_events), len(rec.after_events)) == (1, 1)
        assert rec.after_events[0].result is result


async def test_a_broken_registry_still_fires_both_hooks() -> None:
    class Broken(ToolRegistry):
        def get(self, name: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("registry exploded")

    rec = RecordingHook()
    result = await ToolRouter(Broken(), hooks=[rec]).dispatch(c("x"))
    assert result.error is not None
    assert result.error.code is ErrorCode.INTERNAL_ERROR
    assert (len(rec.before_events), len(rec.after_events)) == (1, 1)
    assert rec.before_events[0].tool is None


async def test_caller_cancellation_still_reports_to_after_hooks() -> None:
    rec = RecordingHook()
    router = ToolRouter(registry(), hooks=[rec])
    task = asyncio.create_task(router.dispatch(c("sleeper")))
    while not rec.before_events:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(rec.after_events) == 1
    result = rec.after_events[0].result
    assert result.error is not None
    assert result.error.code is ErrorCode.CANCELLED
    assert isinstance(result.cause, asyncio.CancelledError)


async def test_batches_fire_hooks_per_call_with_before_preceding_after() -> None:
    log: list[tuple[str, str]] = []

    class Order(RouterHook):
        def before(self, event: CallEvent) -> None:
            log.append(("before", event.call_id))

        def after(self, event: ResultEvent) -> None:
            log.append(("after", event.call.call_id))

    router = ToolRouter(registry(), hooks=[Order()])
    calls = [c("echo", {"text": str(i)}, id_=f"c{i}") for i in range(6)] + [c("nope", id_="u")]
    await router.dispatch_many(calls, max_concurrency=3)
    assert sorted(log) == sorted([(k, x.id) for x in calls for k in ("before", "after")])
    for call in calls:
        assert log.index(("before", call.id)) < log.index(("after", call.id))


# ---- ordering, sync/async, and containment ----------------------------------------------------


async def test_hooks_run_in_registration_order_around_the_handler() -> None:
    log: list[str] = []

    @tool
    def work() -> str:
        """Work."""
        log.append("handler")
        return "x"

    class Named(RouterHook):
        def __init__(self, name: str) -> None:
            self.name = name

        def before(self, event: CallEvent) -> None:
            log.append(f"{self.name}.before")

        async def after(self, event: ResultEvent) -> None:  # async hooks work too
            await asyncio.sleep(0)
            log.append(f"{self.name}.after")

    await ToolRouter(ToolRegistry([work]), hooks=[Named("A"), Named("B")]).dispatch(c("work"))
    assert log == ["A.before", "B.before", "handler", "A.after", "B.after"]


async def test_the_base_hook_is_a_no_op_and_return_values_are_ignored() -> None:
    class Chatty(RouterHook):
        def before(self, event: CallEvent) -> Any:
            return "ignored"

    r = await ToolRouter(registry(), hooks=[RouterHook(), Chatty()]).dispatch(
        c("echo", {"text": "hi"})
    )
    assert r.content == "hi"


async def test_a_failing_hook_is_contained_logged_and_does_not_stop_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = RecordingHook()

    class Exploding(RouterHook):
        def before(self, event: CallEvent) -> None:
            raise RuntimeError("secret hunter2 inside the hook")

        def after(self, event: ResultEvent) -> None:
            raise ValueError("another secret hunter2")

    with caplog.at_level(logging.WARNING, logger="toolrouter"):
        r = await ToolRouter(registry(), hooks=[Exploding(), rec]).dispatch(
            c("echo", {"text": "hi"})
        )
    assert r.ok
    assert r.content == "hi"  # the result is unaffected
    assert (len(rec.before_events), len(rec.after_events)) == (1, 1)  # later hooks still ran
    logged = caplog.text
    assert "Exploding.before failed (RuntimeError)" in logged
    assert "Exploding.after failed (ValueError)" in logged
    assert "hunter2" not in logged  # exception text is not logged


async def test_a_slow_hook_times_out_without_stalling_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = RecordingHook()

    class Hung(RouterHook):
        async def before(self, event: CallEvent) -> None:
            await asyncio.sleep(30)

    with caplog.at_level(logging.WARNING, logger="toolrouter"):
        start = time.perf_counter()
        r = await ToolRouter(registry(), hooks=[Hung(), rec], hook_timeout_s=0.05).dispatch(
            c("echo", {"text": "hi"})
        )
    assert time.perf_counter() - start < 1.0
    assert r.content == "hi"
    assert "Hung.before timed out" in caplog.text
    assert len(rec.after_events) == 1


async def test_a_hooks_own_timeout_error_is_just_a_failure() -> None:
    class Flaky(RouterHook):
        def before(self, event: CallEvent) -> None:
            raise TimeoutError("hook's own timeout")

    r = await ToolRouter(registry(), hooks=[Flaky()]).dispatch(c("echo", {"text": "hi"}))
    assert r.ok


def test_hook_timeout_is_validated() -> None:
    for bad in (0, -1, True, float("inf"), float("nan"), "5"):
        with pytest.raises(ValueError, match="hook_timeout_s"):
            ToolRouter(ToolRegistry(), hook_timeout_s=bad)  # type: ignore[arg-type]


async def test_no_hooks_means_no_event_building(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("events must not be built when there are no hooks")

    monkeypatch.setattr(router_module, "build_call_event", boom)
    assert (await ToolRouter(registry()).dispatch(c("echo", {"text": "hi"}))).ok


# ---- event contents ---------------------------------------------------------------------------


async def test_events_carry_the_tool_the_raw_call_and_redacted_arguments() -> None:
    rec = RecordingHook()
    call = c("add", '{"a": 1, "b": 2}')
    await ToolRouter(registry(), hooks=[rec]).dispatch(call)
    event = rec.before_events[0]
    assert event.tool is rt.add
    assert event.raw_call is call
    assert event.arguments == {"a": 1, "b": 2}
    assert (event.name, event.call_id) == ("add", "c1")


@pytest.mark.parametrize(
    ("name", "args", "note"),
    [
        ("nope", {"text": "hi"}, "<arguments omitted: unknown tool>"),
        ("echo", '{"text": ', "<arguments omitted: invalid_json>"),
        ("echo", "[1]", "<arguments omitted: invalid_json>"),
        ("echo", '{"text": "' + "x" * 70_000 + '"}', "<arguments omitted: arguments_too_large>"),
    ],
)
async def test_unparseable_or_unattributable_arguments_are_omitted_entirely(
    name: str, args: Any, note: str
) -> None:
    rec = RecordingHook()
    await ToolRouter(registry(), hooks=[rec]).dispatch(c(name, args))
    assert rec.before_events[0].arguments == note


async def test_events_are_immutable_and_arguments_are_a_copy() -> None:
    rec = RecordingHook()
    call = c("echo", {"text": "hi"})
    await ToolRouter(registry(), hooks=[rec]).dispatch(call)
    event = rec.before_events[0]
    with pytest.raises(AttributeError):
        event.arguments = {}  # type: ignore[misc]
    event.arguments["text"] = "tampered"
    assert call.arguments == {"text": "hi"}  # the raw call is untouched


async def test_large_arguments_are_bounded_in_events() -> None:
    rec = RecordingHook()
    await ToolRouter(registry(), hooks=[rec]).dispatch(c("echo", {"text": "y" * 5000}))
    text = rec.before_events[0].arguments["text"]
    assert len(text) < 300
    assert "[+4800 chars]" in text


def test_build_call_event_never_raises() -> None:
    class Broken(ToolRegistry):
        def get(self, name: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("x")

    event = hooks_module.build_call_event(Broken(), c("echo"))
    assert event.tool is None
    assert event.arguments == "<arguments omitted: unknown tool>"


# ---- LoggingHook ------------------------------------------------------------------------------


async def test_logging_hook_logs_success_and_failure_at_their_levels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = ToolRouter(registry(), hooks=[LoggingHook()])
    with caplog.at_level(logging.DEBUG, logger="toolrouter"):
        await router.dispatch(c("add", {"a": 1, "b": 2}, id_="ok1"))
        await router.dispatch(c("add", {"a": "x"}, id_="bad1"))
    by_level = {(r.levelno, r.getMessage().split(" args=")[0]) for r in caplog.records}
    assert any(lvl == logging.INFO and "name=add id=ok1 ok=true" in msg for lvl, msg in by_level)
    failure = [r for r in caplog.records if r.levelno == logging.WARNING][0].getMessage()
    assert "id=bad1 ok=false" in failure
    assert "error=arguments_invalid" in failure
    assert 'args={"a": "x"}' in failure
    started = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(started) == 2
    assert all("tool call started" in r.getMessage() for r in started)


async def test_logging_hook_omits_exception_details_unless_asked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    quiet = LoggingHook()
    verbose = LoggingHook(include_tracebacks=True)
    for hook, expect_exc in ((quiet, False), (verbose, True)):
        caplog.clear()
        router = ToolRouter(registry(), hooks=[hook])
        with caplog.at_level(logging.INFO, logger="toolrouter"):
            await router.dispatch(c("crash", {"path": "/etc/passwd"}))
        record = [r for r in caplog.records if "ok=false" in r.getMessage()][0]
        assert "cause=KeyError" in record.getMessage()
        assert (
            "abc123" in caplog.text
        ) is expect_exc  # the KeyError text appears only in tracebacks
        assert (record.exc_info is not None) is expect_exc


async def test_logging_hook_custom_logger_and_levels(caplog: pytest.LogCaptureFixture) -> None:
    custom = logging.getLogger("audit.tools")
    hook = LoggingHook(custom, level=logging.ERROR, failure_level=logging.CRITICAL)
    with caplog.at_level(logging.DEBUG, logger="audit.tools"):
        router = ToolRouter(registry(), hooks=[hook])
        await router.dispatch(c("echo", {"text": "hi"}))
        await router.dispatch(c("nope"))
    levels = sorted(r.levelno for r in caplog.records if r.name == "audit.tools" and r.levelno > 10)
    assert levels == [logging.ERROR, logging.CRITICAL]


async def test_logging_hook_reports_truncation_and_bounds_arguments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @tool(max_output_chars=60)
    def big(text: str) -> str:
        """Big output."""
        return text * 100

    with caplog.at_level(logging.INFO, logger="toolrouter"):
        await ToolRouter(ToolRegistry([big]), hooks=[LoggingHook()]).dispatch(
            c("big", {"text": "z" * 9000})
        )
    line = [r.getMessage() for r in caplog.records if "ok=true" in r.getMessage()][0]
    assert "truncated=true" in line
    assert len(line) < 600
