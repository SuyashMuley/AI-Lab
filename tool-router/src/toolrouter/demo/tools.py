"""Safe example tools for the demo CLI.

(No `from __future__ import annotations` here: the tools are built inside a factory and their
annotations refer to local names, which must be real objects for the schema builder.)
"""

import ast
import datetime
import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from toolrouter import Confined, SideEffect, Tool, ToolError, tool

MAX_EXPRESSION_CHARS = 200
MAX_NODES = 100
MAX_RESULT_BITS = 10_000
MAX_READ_BYTES = 100_000
MAX_WRITE_CHARS = 50_000  # below the router's 64 KB argument cap, so this limit is the one hit
MAX_LISTED = 200

BinOp = Callable[[Any, Any], Any]


def _pow(base: float, exp: float) -> float:
    if isinstance(base, int) and isinstance(exp, int) and exp >= 0:
        if base.bit_length() * exp > MAX_RESULT_BITS:
            raise ToolError("That number is too large to compute.")
        return base**exp  # type: ignore[no-any-return]
    try:
        return math.pow(base, exp)
    except (OverflowError, ValueError):
        raise ToolError("That power cannot be computed.") from None


_BINARY: dict[type, BinOp] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED = "Only numbers and + - * / // % ** ( ) are allowed."


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value  # type: ignore[return-value]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        result = _BINARY[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise ToolError("That number is too large to compute.")
        return result  # type: ignore[no-any-return]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))  # type: ignore[no-any-return]
    raise ToolError(_ALLOWED)


@tool
def calculator(
    expression: Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)],
) -> str:
    """Evaluate an arithmetic expression, e.g. "17 * 23 + 4". Supports + - * / // % ** and ( )."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ToolError(f"That is not a valid arithmetic expression. {_ALLOWED}") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ToolError("That expression is too complicated.")
    try:
        result = _eval(tree.body)
        if isinstance(result, float) and not math.isfinite(result):
            raise ToolError("The result is not a finite number.")
        return str(result)
    except ZeroDivisionError:
        raise ToolError("Division by zero.") from None
    except OverflowError:
        raise ToolError("That number is too large to compute.") from None


@tool(side_effect=SideEffect.READ)
def get_time(timezone: str = "UTC") -> str:
    """Current date and time in an IANA timezone such as "Europe/Oslo" or "UTC"."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        shown = timezone[:50]
        raise ToolError(
            f"Unknown timezone {shown!r}. Use an IANA name such as 'Europe/Oslo' or 'UTC'."
        ) from None
    return datetime.datetime.now(zone).isoformat(timespec="seconds")


def make_file_tools(files_dir: Path) -> list[Tool]:
    """read_file, list_dir (READ) and write_file (WRITE), all confined to `files_dir`."""
    root = Path(files_dir).resolve()
    InDir = Annotated[Path, Confined(root)]

    def shown(path: Path) -> str:
        try:
            return str(path.relative_to(root)) or "."
        except ValueError:
            return "<path>"

    @tool(side_effect=SideEffect.READ)
    def read_file(path: InDir) -> str:
        """Read a text file (first 100 KB), given a path relative to the working directory."""
        if path.is_dir():
            raise ToolError(f"{shown(path)} is a directory; use list_dir instead.")
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_READ_BYTES + 1)
        except FileNotFoundError:
            raise ToolError(f"File not found: {shown(path)}") from None
        except PermissionError:
            raise ToolError(f"Permission denied: {shown(path)}") from None
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return text + ("\n[file continues beyond 100 KB]" if len(data) > MAX_READ_BYTES else "")

    @tool(side_effect=SideEffect.READ)
    def list_dir(path: InDir | None = None) -> list[str]:
        """List a directory (default: the working directory). Subdirectories end with '/'."""
        target = root if path is None else path
        if not target.is_dir():
            raise ToolError(f"Not a directory: {shown(target)}")
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        except PermissionError:
            raise ToolError(f"Permission denied: {shown(target)}") from None
        if len(entries) > MAX_LISTED:
            return [*entries[:MAX_LISTED], f"... and {len(entries) - MAX_LISTED} more"]
        return entries

    @tool(side_effect=SideEffect.WRITE)
    def write_file(
        path: InDir,
        content: Annotated[str, Field(max_length=MAX_WRITE_CHARS)],
        overwrite: bool = False,
    ) -> str:
        """Write a text file inside the working directory. Refuses to replace an existing file
        unless overwrite is true."""
        if path.is_dir():
            raise ToolError(f"{shown(path)} is a directory.")
        if path.exists() and not overwrite:
            raise ToolError(
                f"File already exists: {shown(path)}. Pass overwrite=true to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)  # stays inside root: `path` is confined
        try:
            with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
                handle.write(content)
        except FileExistsError:
            raise ToolError(f"File already exists: {shown(path)}.") from None
        except PermissionError:
            raise ToolError(f"Permission denied: {shown(path)}") from None
        return f"Wrote {len(content)} characters to {shown(path)}"

    return [read_file, list_dir, write_file]


def make_demo_tools(files_dir: Path) -> list[Tool]:
    return [calculator, get_time, *make_file_tools(files_dir)]
