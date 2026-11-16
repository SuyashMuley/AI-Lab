import asyncio
import inspect
from typing import Any

import pytest

from tests.fakes.factories import make_tool
from toolrouter import (
    ALLOW,
    AllOf,
    AllowAll,
    AllowList,
    DenyList,
    ErrorCode,
    MaxSideEffect,
    PolicyDecision,
    PolicyRequest,
    ReadOnlyPolicy,
    SideEffect,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolRouter,
    deny,
    tool,
)
from toolrouter.feedback import model_text


def req(effect: SideEffect = SideEffect.NONE, name: str = "t") -> PolicyRequest:
    t = make_tool(int, side_effect=effect, name=name)
    return PolicyRequest(
        t,
        ToolCall(id="c", name=name, arguments={"x": 1}),
        t.args_model.model_validate({"x": 1}),
    )


async def decide(policy: Any, request: PolicyRequest) -> PolicyDecision:
    out = policy.check(request)
    return await out if inspect.isawaitable(out) else out  # type: ignore[no-any-return]


# ---- built-in policies ----------------------------------------------------------------------


async def test_allow_all() -> None:
    for effect in SideEffect:
        assert (await decide(AllowAll(), req(effect))).allowed


@pytest.mark.parametrize(
    ("level", "effect", "allowed"),
    [
        (SideEffect.NONE, SideEffect.NONE, True),
        (SideEffect.NONE, SideEffect.READ, False),
        (SideEffect.NONE, SideEffect.WRITE, False),
        (SideEffect.READ, SideEffect.NONE, True),
        (SideEffect.READ, SideEffect.READ, True),
        (SideEffect.READ, SideEffect.WRITE, False),
        (SideEffect.WRITE, SideEffect.NONE, True),
        (SideEffect.WRITE, SideEffect.READ, True),
        (SideEffect.WRITE, SideEffect.WRITE, True),
    ],
)
async def test_max_side_effect_truth_table(
    level: SideEffect, effect: SideEffect, allowed: bool
) -> None:
    decision = await decide(MaxSideEffect(level), req(effect))
    assert decision.allowed is allowed
    if not allowed:
        assert effect.value in decision.reason
        assert level.value in decision.reason


@pytest.mark.parametrize(
    ("effect", "allowed"),
    [(SideEffect.NONE, True), (SideEffect.READ, True), (SideEffect.WRITE, False)],
)
async def test_read_only_policy(effect: SideEffect, allowed: bool) -> None:
    decision = await decide(ReadOnlyPolicy(), req(effect))
    assert decision.allowed is allowed
    assert allowed or "read-only" in decision.reason


async def test_allow_list_and_deny_list() -> None:
    allow = AllowList(["a", "b"])
    assert (await decide(allow, req(name="a"))).allowed
    denied = await decide(allow, req(name="c"))
    assert not denied.allowed
    assert "allowed tools" in denied.reason
    assert not (await decide(AllowList([]), req(name="a"))).allowed  # empty list allows nothing

    block = DenyList(["x"])
    assert not (await decide(block, req(name="x"))).allowed
    assert (await decide(block, req(name="y"))).allowed


async def test_max_side_effect_validates_its_level() -> None:
    with pytest.raises(ValueError):
        MaxSideEffect("everything")  # type: ignore[arg-type]


async def test_decision_helpers() -> None:
    assert ALLOW.allowed
    assert deny("because").reason == "because"
    assert not deny().allowed


class Sync:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision, self.calls = decision, 0

    def check(self, request: PolicyRequest) -> PolicyDecision:
        self.calls += 1
        return self.decision


class Async(Sync):
    async def check(self, request: PolicyRequest) -> PolicyDecision:  # type: ignore[override]
        self.calls += 1
        await asyncio.sleep(0)
        return self.decision


async def test_all_of_first_denial_wins_and_stops_evaluating() -> None:
    first, second, third = Sync(ALLOW), Async(deny("second says no")), Sync(deny("third says no"))
    decision = await decide(AllOf(first, second, third), req())
    assert (decision.allowed, decision.reason) == (False, "second says no")
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)


async def test_all_of_allows_when_all_allow_and_when_empty() -> None:
    assert (await decide(AllOf(Sync(ALLOW), Async(ALLOW)), req())).allowed
    assert (await decide(AllOf(), req())).allowed


async def test_all_of_rejects_a_policy_that_returns_garbage() -> None:
    class Bad:
        def check(self, request: PolicyRequest) -> Any:
            return True

    with pytest.raises(TypeError):
        await decide(AllOf(Bad()), req())


# ---- inside the router ----------------------------------------------------------------------


CALLS: list[str] = []


def make_tools() -> tuple[Tool, Tool, Tool]:
    CALLS.clear()

    @tool(side_effect=SideEffect.WRITE)
    def write_file(path: str) -> str:
        """Writes."""
        CALLS.append("write_file")
        return "written"

    @tool(side_effect=SideEffect.READ)
    def read_file(path: str) -> str:
        """Reads."""
        CALLS.append("read_file")
        return "contents"

    @tool
    def pure(x: int = 0) -> int:
        """Pure."""
        CALLS.append("pure")
        return x

    return write_file, read_file, pure


def call(name: str, args: Any = None, id_: str = "c") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments={"path": "/tmp/a"} if args is None else args)


async def test_read_only_router_denies_writes_without_running_them() -> None:
    write_file, read_file, pure = make_tools()
    router = ToolRouter(ToolRegistry([write_file, read_file, pure]), policy=ReadOnlyPolicy())
    denied = await router.dispatch(call("write_file"))
    assert not denied.ok
    assert denied.error is not None
    assert denied.error.code is ErrorCode.POLICY_DENIED
    assert not denied.error.retryable
    assert denied.error.message == (
        "Tool 'write_file' is not permitted: "
        "it modifies external state and this session is read-only"
    )
    assert CALLS == []  # never ran
    assert model_text(denied).startswith("Error (policy_denied): ")

    assert (await router.dispatch(call("read_file"))).content == "contents"
    assert (await router.dispatch(call("pure", {"x": 4}))).content == "4"
    assert CALLS == ["read_file", "pure"]


async def test_default_policy_allows_everything() -> None:
    write_file, _, _ = make_tools()
    r = await ToolRouter(ToolRegistry([write_file])).dispatch(call("write_file"))
    assert r.content == "written"


async def test_policy_sees_validated_arguments_and_only_for_valid_calls() -> None:
    write_file, read_file, _ = make_tools()
    seen: list[PolicyRequest] = []

    class Spy:
        def check(self, request: PolicyRequest) -> PolicyDecision:
            seen.append(request)
            return ALLOW

    router = ToolRouter(ToolRegistry([write_file, read_file]), policy=Spy())
    await router.dispatch(call("read_file", '{"path": "/x"}'))
    assert len(seen) == 1
    assert isinstance(seen[0].args, read_file.args_model)
    assert seen[0].args.path == "/x"  # type: ignore[attr-defined]
    assert seen[0].tool is read_file

    await router.dispatch(call("nope"))  # unknown tool
    await router.dispatch(call("read_file", '{"path": '))  # invalid JSON
    await router.dispatch(call("read_file", {"path": 5}))  # invalid arguments
    assert len(seen) == 1  # the policy was never consulted for calls that never validated


@pytest.mark.parametrize("is_async", [False, True])
async def test_argument_dependent_policy(is_async: bool) -> None:
    write_file, _, _ = make_tools()

    class NoEtc:
        def check(self, request: PolicyRequest) -> PolicyDecision:
            path = request.args.path  # type: ignore[attr-defined]
            return deny("paths under /etc are off limits") if path.startswith("/etc") else ALLOW

    class NoEtcAsync(NoEtc):
        async def check(self, request: PolicyRequest) -> PolicyDecision:  # type: ignore[override]
            await asyncio.sleep(0)
            return NoEtc.check(self, request)

    router = ToolRouter(ToolRegistry([write_file]), policy=NoEtcAsync() if is_async else NoEtc())
    bad = await router.dispatch(call("write_file", {"path": "/etc/passwd"}))
    assert bad.error is not None
    assert bad.error.code is ErrorCode.POLICY_DENIED
    assert "off limits" in bad.error.message
    assert "/etc/passwd" not in bad.error.message  # the denial never echoes the arguments
    assert CALLS == []
    assert (await router.dispatch(call("write_file", {"path": "/tmp/ok"}))).ok


async def test_a_policy_that_raises_fails_closed_and_never_leaks() -> None:
    write_file, _, _ = make_tools()

    class Broken:
        def check(self, request: PolicyRequest) -> PolicyDecision:
            raise RuntimeError("db password=hunter2 unreachable")

    r = await ToolRouter(ToolRegistry([write_file]), policy=Broken()).dispatch(call("write_file"))
    assert r.error is not None
    assert r.error.code is ErrorCode.INTERNAL_ERROR
    assert "hunter2" not in r.error.message
    assert "policy check failed" in r.error.message
    assert isinstance(r.cause, RuntimeError)
    assert CALLS == []


@pytest.mark.parametrize("garbage", [True, False, None, "allow", {"allowed": True}, 1])
async def test_a_policy_returning_a_non_decision_fails_closed(garbage: Any) -> None:
    write_file, _, _ = make_tools()

    class Sloppy:
        def check(self, request: PolicyRequest) -> Any:
            return garbage

    r = await ToolRouter(ToolRegistry([write_file]), policy=Sloppy()).dispatch(call("write_file"))
    assert r.error is not None
    assert r.error.code is ErrorCode.INTERNAL_ERROR
    assert CALLS == []


async def test_denial_reasons_are_bounded_and_default_sensibly() -> None:
    write_file, _, _ = make_tools()

    def router_with(reason: str) -> ToolRouter:
        class P:
            def check(self, request: PolicyRequest) -> PolicyDecision:
                return deny(reason)

        return ToolRouter(ToolRegistry([write_file]), policy=P())

    long = await router_with("x" * 5000).dispatch(call("write_file"))
    assert long.error is not None
    assert len(long.error.message) < 400
    blank = await router_with("   ").dispatch(call("write_file"))
    assert blank.error is not None
    assert blank.error.message == "Tool 'write_file' is not permitted by policy."


async def test_cancellation_inside_a_policy() -> None:
    write_file, _, _ = make_tools()

    class Spurious:
        async def check(self, request: PolicyRequest) -> PolicyDecision:
            raise asyncio.CancelledError

    r = await ToolRouter(ToolRegistry([write_file]), policy=Spurious()).dispatch(call("write_file"))
    assert r.error is not None
    assert r.error.code is ErrorCode.CANCELLED  # a stray cancel is a failed call, not a crash
    assert CALLS == []

    started = asyncio.Event()

    class Slow:
        async def check(self, request: PolicyRequest) -> PolicyDecision:
            started.set()
            await asyncio.sleep(30)
            return ALLOW

    router = ToolRouter(ToolRegistry([write_file]), policy=Slow())
    task = asyncio.create_task(router.dispatch(call("write_file")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # a real cancel still propagates
        await task
    assert CALLS == []


async def test_denials_in_a_batch_do_not_disturb_other_calls() -> None:
    write_file, read_file, pure = make_tools()
    router = ToolRouter(ToolRegistry([write_file, read_file, pure]), policy=ReadOnlyPolicy())
    results = await router.dispatch_many(
        [
            call("write_file", id_="w1"),
            call("read_file", id_="r1"),
            call("write_file", id_="w2"),
            call("pure", {"x": 3}, id_="p1"),
        ]
    )
    assert [r.call_id for r in results] == ["w1", "r1", "w2", "p1"]
    assert [r.error.code if r.error else None for r in results] == [
        ErrorCode.POLICY_DENIED,
        None,
        ErrorCode.POLICY_DENIED,
        None,
    ]
    assert sorted(CALLS) == ["pure", "read_file"]


async def test_denied_calls_use_no_worker_threads() -> None:
    write_file, _, _ = make_tools()
    router = ToolRouter(ToolRegistry([write_file]), policy=ReadOnlyPolicy())
    for _ in range(50):
        await router.dispatch(call("write_file"))
    assert router.executor.active_threads == 0
