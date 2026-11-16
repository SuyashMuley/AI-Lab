"""Nothing a handler raises may leak into what the model (or a serialized result) can see."""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from toolrouter import ErrorCode, Tool, ToolCall, ToolError, ToolRegistry, ToolRouter, tool

SECRETS = ("hunter2", "/home/alice", "id_rsa", "SECRET-CHAIN", "token-abc123", "10.0.0.5")


class Leaky(Exception):
    def __str__(self) -> str:
        return "hunter2 leaked via __str__"

    def __repr__(self) -> str:
        return "Leaky('hunter2 leaked via __repr__')"


class Exploding(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str() blew up with hunter2")


def _chained() -> None:
    try:
        raise ValueError("SECRET-CHAIN")
    except ValueError as inner:
        raise RuntimeError("outer failure") from inner


RAISERS: dict[str, Callable[[], None]] = {
    "value_error": lambda: (_ for _ in ()).throw(
        ValueError("password=hunter2 in /home/alice/.ssh/id_rsa")
    ),
    "os_error": lambda: (_ for _ in ()).throw(
        FileNotFoundError(2, "No such file", "/home/alice/.ssh/id_rsa")
    ),
    "key_error": lambda: (_ for _ in ()).throw(KeyError("token-abc123")),
    "leaky_str_and_repr": lambda: (_ for _ in ()).throw(Leaky()),
    "exploding_str": lambda: (_ for _ in ()).throw(Exploding()),
    "chained": _chained,
    "timeout_with_host": lambda: (_ for _ in ()).throw(TimeoutError("connect 10.0.0.5:5432")),
    "huge_message": lambda: (_ for _ in ()).throw(RuntimeError("hunter2 " * 100_000)),
}


def make_tool(kind: str, is_async: bool) -> Tool:
    raiser = RAISERS[kind]
    if is_async:

        @tool(name="boom")
        async def boom_async() -> str:
            """Raises."""
            raiser()
            return ""

        return boom_async

    @tool(name="boom")
    def boom_sync() -> str:
        """Raises."""
        raiser()
        return ""

    return boom_sync


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", sorted(RAISERS))
async def test_unexpected_exceptions_never_leak(kind: str, is_async: bool) -> None:
    router = ToolRouter(ToolRegistry([make_tool(kind, is_async)]))
    r = await router.dispatch(ToolCall(id="c", name="boom", arguments="{}"))

    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    visible = r.error.message + r.model_dump_json() + r.text
    for secret in SECRETS:
        assert secret not in visible, f"{secret!r} leaked for {kind}"
    assert "Traceback" not in visible
    assert 'File "' not in visible
    assert len(r.error.message) < 200  # bounded, however large the exception text was
    assert r.cause is not None  # ...while the real exception stays available to logs


async def test_tool_error_messages_pass_through_but_are_bounded() -> None:
    @tool
    def friendly() -> str:
        """Raises a model-safe error."""
        raise ToolError("City 'Atlantis' not found. Try a real city.")

    r = await ToolRouter(ToolRegistry([friendly])).dispatch(
        ToolCall(id="c", name="friendly", arguments="{}")
    )
    assert r.error is not None
    assert r.error.message == "City 'Atlantis' not found. Try a real city."


class ExplodingToolError(ToolError):
    def __str__(self) -> str:
        raise RuntimeError("str() blew up")


async def test_a_hostile_tool_error_str_cannot_break_the_router() -> None:
    @tool
    def hostile() -> str:
        """Raises a ToolError whose __str__ raises."""
        raise ExplodingToolError("x")

    r = await ToolRouter(ToolRegistry([hostile])).dispatch(
        ToolCall(id="c", name="hostile", arguments="{}")
    )
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED  # not INTERNAL_ERROR
    assert r.error.message == "The tool reported a failure."


async def test_failures_carry_no_traceback_text_even_in_deep_stacks() -> None:
    def level3() -> None:
        raise ValueError("deep secret hunter2")

    def level2() -> None:
        level3()

    @tool
    def deep() -> str:
        """Deep stack."""
        level2()
        return ""

    r = await ToolRouter(ToolRegistry([deep])).dispatch(
        ToolCall(id="c", name="deep", arguments="{}")
    )
    blob: Any = r.model_dump_json()
    assert "level3" not in blob
    assert "hunter2" not in blob
    assert r.cause is not None
    assert r.cause.__traceback__ is not None  # full traceback retained for logs only


async def test_concurrent_failures_do_not_cross_contaminate() -> None:
    @tool
    async def a() -> str:
        """Fails with a's secret."""
        await asyncio.sleep(0.01)
        raise ValueError("secret-of-a")

    @tool
    async def b() -> str:
        """Fails with b's secret."""
        raise ValueError("secret-of-b")

    router = ToolRouter(ToolRegistry([a, b]))
    ra, rb = await asyncio.gather(
        router.dispatch(ToolCall(id="1", name="a", arguments="{}")),
        router.dispatch(ToolCall(id="2", name="b", arguments="{}")),
    )
    assert ra.error is not None
    assert rb.error is not None
    assert (ra.call_id, rb.call_id) == ("1", "2")
    assert "secret" not in ra.error.message + rb.error.message
    assert isinstance(ra.cause, ValueError)
    assert str(ra.cause) == "secret-of-a"
    assert str(rb.cause) == "secret-of-b"
