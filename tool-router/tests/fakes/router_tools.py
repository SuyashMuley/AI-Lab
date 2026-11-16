"""Handlers for router tests. Every handler records its name in CALLS (a spy)."""

import asyncio
import datetime
from enum import StrEnum
from typing import Any

from toolrouter import SideEffect, StrictModel, ToolError, tool

CALLS: list[str] = []


class Report(StrictModel):
    city: str
    temp: float


class Other(StrictModel):
    city: str
    temp: float


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class Opaque:
    pass


@tool
def echo(text: str) -> str:
    """Echo text."""
    CALLS.append("echo")
    return text


@tool
def add(a: int, b: int = 0) -> int:
    """Add numbers."""
    CALLS.append("add")
    return a + b


@tool
async def slow_add(a: int, b: int) -> int:
    """Add numbers asynchronously."""
    CALLS.append("slow_add")
    await asyncio.sleep(0)
    return a + b


@tool
def report(city: str) -> Report:
    """Weather report as a model."""
    CALLS.append("report")
    return Report(city=city, temp=21.5)


@tool
def reports(n: int) -> list[Report]:
    """Several reports."""
    CALLS.append("reports")
    return [Report(city=f"c{i}", temp=float(i)) for i in range(n)]


@tool
def stats() -> dict[str, int]:
    """A dict result."""
    CALLS.append("stats")
    return {"a": 1}


@tool
def nothing() -> None:
    """Returns None."""
    CALLS.append("nothing")


@tool
def when() -> datetime.datetime:
    """A datetime result."""
    CALLS.append("when")
    return datetime.datetime(2024, 1, 2, 3, 4, 5)


@tool
def pick() -> Color:
    """An enum result."""
    CALLS.append("pick")
    return Color.RED


@tool
def maybe(flag: bool) -> str | None:
    """Optional string result."""
    CALLS.append("maybe")
    return "yes" if flag else None


@tool
def untyped(x: int):  # type: ignore[no-untyped-def]
    """Unannotated return."""
    CALLS.append("untyped")
    return {"x": x}


@tool
def any_typed(x: int) -> Any:
    """Any return."""
    CALLS.append("any_typed")
    return {"x": x}


@tool
def bad_int(x: int) -> int:
    """Declares int, returns str."""
    CALLS.append("bad_int")
    return "oops"  # type: ignore[return-value]


@tool
def dict_model() -> Report:
    """Declares a model, returns a dict that satisfies it (accepted, validated strictly)."""
    CALLS.append("dict_model")
    return {"city": "x", "temp": 1.0}  # type: ignore[return-value]


@tool
def bad_model() -> Report:
    """Declares a model, returns a dict that does NOT satisfy it."""
    CALLS.append("bad_model")
    return {"city": 1, "temp": "hot"}  # type: ignore[return-value]


@tool
def wrong_model() -> Report:
    """Declares one model, returns a different model class."""
    CALLS.append("wrong_model")
    return Other(city="x", temp=1.0)  # type: ignore[return-value]


@tool
def bad_str() -> str:
    """Declares str, returns int."""
    CALLS.append("bad_str")
    return 5  # type: ignore[return-value]


@tool
def unserializable():  # type: ignore[no-untyped-def]
    """Unannotated, returns something that is not JSON."""
    CALLS.append("unserializable")
    return Opaque()


@tool
def expected_failure(path: str) -> str:
    """Raises a model-safe error."""
    CALLS.append("expected_failure")
    raise ToolError(f"File not found: {path}")


class QuotaExceeded(ToolError):
    pass


@tool
def quota() -> str:
    """Raises a ToolError subclass."""
    CALLS.append("quota")
    raise QuotaExceeded("Daily quota exceeded; try again tomorrow.")


@tool
def crash(path: str) -> str:
    """Raises an unexpected exception whose text must never reach the model."""
    CALLS.append("crash")
    raise KeyError("/etc/secret/token-abc123")


@tool
async def cancelled() -> str:
    """Raises CancelledError from inside the handler (the caller is not being cancelled)."""
    CALLS.append("cancelled")
    raise asyncio.CancelledError


@tool
def interrupt() -> str:
    """Raises KeyboardInterrupt."""
    CALLS.append("interrupt")
    raise KeyboardInterrupt


@tool
def exit_now() -> str:
    """Raises SystemExit."""
    CALLS.append("exit_now")
    raise SystemExit(3)


@tool
async def async_interrupt() -> str:
    """Raises KeyboardInterrupt on the event loop."""
    CALLS.append("async_interrupt")
    raise KeyboardInterrupt


@tool
async def async_exit_now() -> str:
    """Raises SystemExit on the event loop."""
    CALLS.append("async_exit_now")
    raise SystemExit(3)


@tool(side_effect=SideEffect.READ)
def lookup(n: int) -> int:
    """READ tool."""
    CALLS.append("lookup")
    return n


@tool(side_effect=SideEffect.WRITE)
def delete(n: int) -> int:
    """WRITE tool."""
    CALLS.append("delete")
    return n


SAFE = [echo, add, slow_add, report, reports, stats, nothing, when, pick, maybe, untyped, lookup]
