"""Real toolrouter tools for the loop's tests. Handlers record what ran in CALLS (a spy)."""

import asyncio

from toolrouter import SideEffect, ToolError, ToolRegistry, tool

CALLS: list[str] = []


@tool
def add(a: int, b: int = 0) -> int:
    """Add two numbers."""
    CALLS.append("add")
    return a + b


@tool
def echo(text: str) -> str:
    """Echo text back."""
    CALLS.append("echo")
    return text


@tool(side_effect=SideEffect.READ)
async def slow(ms: int, label: str = "") -> str:
    """Wait a little, then report."""
    CALLS.append("slow")
    await asyncio.sleep(ms / 1000)
    return f"slow-{label or ms}"


@tool
def boom() -> str:
    """Always fails in an expected way."""
    CALLS.append("boom")
    raise ToolError("backend down")


@tool
def crash(path: str) -> str:
    """Fails in an unexpected way whose text must never reach the model."""
    CALLS.append("crash")
    raise KeyError("/etc/secret/token-abc123")


@tool(side_effect=SideEffect.WRITE)
def save(name: str) -> str:
    """A WRITE tool (denied by the default read-only policy)."""
    CALLS.append("save")
    return f"saved {name}"


@tool(max_output_chars=60)
def big() -> str:
    """Returns far more than its output cap."""
    CALLS.append("big")
    return "x" * 500


@tool(timeout_s=0.05)
async def hang() -> str:
    """Never finishes within its timeout."""
    CALLS.append("hang")
    await asyncio.sleep(30)
    return "never"


@tool(timeout_s=30)
async def wait_forever() -> str:
    """Waits until cancelled (for cancellation tests)."""
    CALLS.append("wait_forever")
    await asyncio.sleep(30)
    return "never"


ALL = [add, echo, slow, boom, crash, save, big, hang, wait_forever]


def registry(*tools: object) -> ToolRegistry:
    return ToolRegistry(list(tools) if tools else ALL)  # type: ignore[arg-type]
