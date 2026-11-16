import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

from tests.fakes import router_tools as rt
from toolrouter import (
    ErrorCode,
    Limits,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolRouter,
    tool,
)
from toolrouter.formats import ollama


@pytest.fixture(autouse=True)
def reset_calls() -> Iterator[None]:
    rt.CALLS.clear()
    yield
    rt.CALLS.clear()


def registry() -> ToolRegistry:
    names = {
        "echo", "add", "slow_add", "report", "reports", "stats", "nothing", "when", "pick",
        "maybe", "untyped", "any_typed", "bad_int", "bad_model", "bad_str", "unserializable",
        "expected_failure", "quota", "crash", "lookup", "delete", "dict_model", "wrong_model",
    }  # fmt: skip
    return ToolRegistry([getattr(rt, n) for n in sorted(names)])


def call(name: str, arguments: Any = "{}", id_: str = "c1") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=arguments)


async def run(name: str, arguments: Any = "{}", **router_kw: Any) -> ToolResult:
    return await ToolRouter(registry(), **router_kw).dispatch(call(name, arguments))


# ---- success ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("echo", '{"text": "hi"}', "hi"),
        ("add", '{"a": 2, "b": 3}', "5"),
        ("add", {"a": 2}, "2"),
        ("slow_add", '{"a": 2, "b": 3}', "5"),
        ("report", '{"city": "Oslo"}', '{"city":"Oslo","temp":21.5}'),
        ("reports", '{"n": 2}', '[{"city":"c0","temp":0.0},{"city":"c1","temp":1.0}]'),
        ("stats", "{}", '{"a":1}'),
        ("nothing", "", "null"),
        ("when", "{}", '"2024-01-02T03:04:05"'),
        ("pick", "{}", '"red"'),
        ("maybe", '{"flag": true}', "yes"),
        ("maybe", '{"flag": false}', "null"),
        ("untyped", '{"x": 3}', '{"x":3}'),
        ("any_typed", '{"x": 3}', '{"x":3}'),
    ],
)
async def test_success_results(name: str, arguments: Any, expected: str) -> None:
    r = await run(name, arguments)
    assert r.ok, r.error
    assert r.content == expected
    assert r.error is None
    assert r.call_id == "c1"
    assert r.name == name
    assert not r.truncated
    assert r.text == expected


async def test_duration_uses_the_injected_clock() -> None:
    ticks = iter([10.0, 10.25])
    r = await run("echo", '{"text": "x"}', clock=lambda: next(ticks))
    assert r.duration_ms == pytest.approx(250.0)


async def test_failures_also_report_duration() -> None:
    ticks = iter([1.0, 1.5])
    r = await run("nope", clock=lambda: next(ticks))
    assert not r.ok
    assert r.duration_ms == pytest.approx(500.0)


# ---- each failure mode ---------------------------------------------------------------------


async def test_unknown_tool_lists_available() -> None:
    r = await run("nope")
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.UNKNOWN_TOOL
    assert r.error.retryable
    assert "Unknown tool 'nope'" in r.error.message
    assert "echo" in r.error.message
    assert r.content is None
    assert rt.CALLS == []


async def test_unknown_tool_message_is_bounded() -> None:
    long_name = "x" * 10_000
    r = await run(long_name)
    assert r.error is not None
    assert len(r.error.message) < 1000
    assert "…" in r.error.message
    many = ToolRegistry([tool(name=f"t{i}", description="d")(lambda: None) for i in range(25)])
    r2 = await ToolRouter(many).dispatch(call("zzz"))
    assert r2.error is not None
    assert "(+5 more)" in r2.error.message
    assert "t24" not in r2.error.message


async def test_unknown_tool_with_empty_registry() -> None:
    r = await ToolRouter(ToolRegistry()).dispatch(call("x"))
    assert r.error is not None
    assert "No tools are available" in r.error.message


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ('{"text": ', ErrorCode.INVALID_JSON),
        ("[1]", ErrorCode.INVALID_JSON),
        ('{"text": "a", "text": "b"}', ErrorCode.INVALID_JSON),
        ('{"text": "' + "x" * 70_000 + '"}', ErrorCode.ARGUMENTS_TOO_LARGE),
        ("{}", ErrorCode.ARGUMENTS_INVALID),
        ('{"text": 5}', ErrorCode.ARGUMENTS_INVALID),
        ('{"text": "a", "extra": 1}', ErrorCode.ARGUMENTS_INVALID),
    ],
)
async def test_bad_arguments_never_reach_the_handler(arguments: str, code: ErrorCode) -> None:
    r = await run("echo", arguments)
    assert not r.ok
    assert r.error is not None
    assert r.error.code is code
    assert r.error.retryable
    assert rt.CALLS == []  # handler never ran


async def test_arguments_invalid_carries_issues() -> None:
    r = await run("add", '{"a": "x"}')
    assert r.error is not None
    assert r.error.issues[0].loc == ["a"]
    assert "a:" in r.error.message


async def test_router_limits_are_applied() -> None:
    r = await run("echo", '{"text": "' + "x" * 100 + '"}', limits=Limits(max_bytes=50))
    assert r.error is not None
    assert r.error.code is ErrorCode.ARGUMENTS_TOO_LARGE


async def test_tool_error_message_is_passed_to_the_model() -> None:
    r = await run("expected_failure", '{"path": "a.txt"}')
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert r.error.message == "File not found: a.txt"
    assert rt.CALLS == ["expected_failure"]


async def test_tool_error_subclass_and_limits_on_message() -> None:
    r = await run("quota")
    assert r.error is not None
    assert r.error.message == "Daily quota exceeded; try again tomorrow."

    @tool
    def wordy() -> str:
        """Doc."""
        from toolrouter import ToolError

        raise ToolError("x" * 5000)

    @tool
    def silent() -> str:
        """Doc."""
        from toolrouter import ToolError

        raise ToolError("   ")

    router = ToolRouter(ToolRegistry([wordy, silent]))
    long = await router.dispatch(call("wordy"))
    assert long.error is not None
    assert len(long.error.message) == 500
    blank = await router.dispatch(call("silent"))
    assert blank.error is not None
    assert blank.error.message == "The tool reported a failure."


async def test_unexpected_exception_is_generic_and_kept_as_cause() -> None:
    r = await run("crash", '{"path": "x"}')
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert "KeyError" in r.error.message
    assert "/etc/secret" not in r.error.message
    assert "abc123" not in r.error.message
    assert isinstance(r.cause, KeyError)  # available to logs/hooks
    assert "abc123" not in r.model_dump_json()  # but never serialized


@pytest.mark.parametrize(
    "name", ["bad_int", "bad_model", "wrong_model", "bad_str", "unserializable"]
)
async def test_bad_results_become_result_invalid(name: str) -> None:
    r = await run(name, '{"x": 1}' if name == "bad_int" else "{}")
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.RESULT_INVALID
    assert not r.error.retryable
    assert r.content is None
    assert r.cause is not None


async def test_dict_satisfying_the_declared_model_is_accepted() -> None:
    r = await run("dict_model")
    assert r.ok
    assert r.content == '{"city":"x","temp":1.0}'


async def test_internal_errors_are_contained() -> None:
    class Broken(ToolRegistry):
        def get(self, name: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom with /private/path")

    r = await ToolRouter(Broken()).dispatch(call("x"))
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.INTERNAL_ERROR
    assert "/private/path" not in r.error.message
    assert isinstance(r.cause, RuntimeError)


# ---- BaseExceptions ---------------------------------------------------------------------------


async def test_cancelled_error_from_inside_a_handler_is_a_failed_call_not_a_crash() -> None:
    r = await ToolRouter(ToolRegistry([rt.cancelled])).dispatch(call("cancelled"))
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.CANCELLED
    assert not r.error.retryable
    assert isinstance(r.cause, asyncio.CancelledError)


@pytest.mark.parametrize(
    ("t", "exc"), [(rt.async_interrupt, KeyboardInterrupt), (rt.async_exit_now, SystemExit)]
)
def test_keyboard_interrupt_and_system_exit_on_the_loop_propagate(
    t: Any, exc: type[BaseException]
) -> None:
    async def go() -> ToolResult:
        return await ToolRouter(ToolRegistry([t])).dispatch(call(t.name))

    with pytest.raises(exc):
        asyncio.run(go())


# ---- coerce wiring ------------------------------------------------------------------------


async def test_router_level_coerce_follows_the_write_rule() -> None:
    lenient = {"n": "5"}
    assert (await run("lookup", lenient)).error is not None  # strict by default
    assert (await run("lookup", lenient, coerce=True)).content == "5"
    assert (await run("add", {"a": "2"}, coerce=True)).content == "2"
    r = await run("delete", lenient, coerce=True)  # WRITE never inherits lenient
    assert r.error is not None
    assert r.error.code is ErrorCode.ARGUMENTS_INVALID
    assert "delete" not in rt.CALLS


# ---- result type invariants -----------------------------------------------------------------


def test_tool_result_invariants() -> None:
    with pytest.raises(ValueError):
        ToolResult(call_id="1", name="n", ok=True)  # ok needs content
    with pytest.raises(ValueError):
        ToolResult(call_id="1", name="n", ok=False)  # failed needs error
    from toolrouter.calls import make_error

    err = make_error(ErrorCode.CANCELLED, "x")
    with pytest.raises(ValueError):
        ToolResult(call_id="1", name="n", ok=True, content="c", error=err)
    with pytest.raises(ValueError):
        ToolResult(call_id="1", name="n", ok=False, content="c", error=err)
    assert ToolResult(call_id="1", name="n", ok=True, content="").text == ""
    assert ToolResult(call_id="1", name="n", ok=False, error=err).text == "x"


async def test_result_serializes_without_the_cause() -> None:
    r = await run("crash", '{"path": "x"}')
    data = json.loads(r.model_dump_json())
    assert "cause" not in data
    assert data["error"]["code"] == "tool_failed"


# ---- end to end -------------------------------------------------------------------------------


async def test_provider_message_to_results_preserves_order_and_ids() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}},
            {"function": {"name": "nope", "arguments": {}}},
            {"function": {"name": "echo", "arguments": {"text": "hi"}}},
        ],
    }
    router = ToolRouter(registry())
    results = [await router.dispatch(c) for c in ollama.parse_calls(message)]
    assert [(r.call_id, r.name, r.ok) for r in results] == [
        ("call_0", "add", True),
        ("call_1", "nope", False),
        ("call_2", "echo", True),
    ]
    assert [r.text for r in results if r.ok] == ["3", "hi"]


async def test_concurrent_dispatch_shares_no_state() -> None:
    router = ToolRouter(registry())
    calls = [call("slow_add", {"a": i, "b": 1}, id_=f"c{i}") for i in range(20)]
    results = await asyncio.gather(*(router.dispatch(c) for c in calls))
    assert [r.content for r in results] == [str(i + 1) for i in range(20)]
    assert [r.call_id for r in results] == [f"c{i}" for i in range(20)]


async def test_registry_changes_are_seen_by_an_existing_router() -> None:
    reg = ToolRegistry()
    router = ToolRouter(reg)
    assert (await router.dispatch(call("echo", {"text": "a"}))).error is not None
    reg.register(rt.echo)
    assert (await router.dispatch(call("echo", {"text": "a"}))).content == "a"
