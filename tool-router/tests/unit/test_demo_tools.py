import os
import time
from pathlib import Path
from typing import Any

import pytest

from toolrouter import (
    ErrorCode,
    ReadOnlyPolicy,
    SideEffect,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolRouter,
)
from toolrouter.demo.tools import (
    MAX_LISTED,
    MAX_READ_BYTES,
    calculator,
    get_time,
    make_demo_tools,
    make_file_tools,
)


async def run(tool: Any, args: Any, *tools: Any, **kw: Any) -> ToolResult:
    router = ToolRouter(ToolRegistry([tool, *tools]), **kw)
    return await router.dispatch(ToolCall(id="c", name=tool.name, arguments=args))


async def calc(expression: str) -> ToolResult:
    return await run(calculator, {"expression": expression})


# ---- calculator ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("17 * 23 + 4", "395"),
        ("2 ** 10", "1024"),
        ("10 / 4", "2.5"),
        ("10 // 3", "3"),
        ("7 % 3", "1"),
        ("-5 + 2", "-3"),
        ("+4", "4"),
        ("(1 + 2) * 3", "9"),
        ("2 ** -1", "0.5"),
        ("1e3 + 1", "1001.0"),
        ("  3   +   4  ", "7"),
        ("2 + 3 * 4 ** 2", "50"),
        ("2 ** 3 ** 2", "512"),
    ],
)
async def test_calculator_evaluates(expression: str, expected: str) -> None:
    r = await calc(expression)
    assert r.ok, r.error
    assert r.content == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "abs(-1)",
        "a + 1",
        "'a' * 3",
        "True + 1",
        "1j",
        "[1, 2][0]",
        "(1).__class__",
        "lambda: 1",
        "1 if 1 else 2",
        "x := 5",
        "import os",
        "1 +",
        "1 +* 2",
    ],
)
async def test_calculator_rejects_anything_but_arithmetic(expression: str) -> None:
    r = await calc(expression)
    assert not r.ok
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert "pwned" not in r.error.message


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "Division by zero."),
        ("1 // 0", "Division by zero."),
        ("5 % 0", "Division by zero."),
        ("9 ** 9 ** 9", "too large"),
        ("2 ** 100000", "too large"),
        ("10 ** 10 ** 10", "too large"),
        ("(10 ** 3000) * (10 ** 3000)", "too large"),
        ("1e308 * 10", "not a finite number"),
        ("(-8) ** 0.5", "cannot be computed"),
        ("0.0 ** -1", "cannot be computed"),
        ("+".join(["1"] * 40), "too complicated"),
    ],
)
async def test_calculator_guards_against_blowups(expression: str, message: str) -> None:
    start = time.perf_counter()
    r = await calc(expression)
    assert time.perf_counter() - start < 1.0  # refused, not computed
    assert r.error is not None
    assert message in r.error.message


async def test_calculator_input_limits() -> None:
    assert (await calc("")).error is not None
    long = await calc("1+" * 150 + "1")
    assert long.error is not None
    assert long.error.code is ErrorCode.ARGUMENTS_INVALID  # over 200 characters
    nested = await calc("(" * 100 + "1" + ")" * 100)
    assert nested.ok or nested.error is not None  # either answer is fine; it must not crash


# ---- get_time -----------------------------------------------------------------------------------


async def test_get_time() -> None:
    utc = await run(get_time, {})
    assert utc.ok
    assert utc.content is not None
    assert utc.content.endswith("+00:00")
    oslo = await run(get_time, {"timezone": "Europe/Oslo"})
    assert oslo.content is not None
    assert oslo.content[-6:] in ("+01:00", "+02:00")
    assert get_time.side_effect is SideEffect.READ


@pytest.mark.parametrize("zone", ["Mars/Phobos", "../etc/passwd", "", "/etc/passwd", "x" * 500])
async def test_get_time_rejects_bad_zones_without_echoing_them_unbounded(zone: str) -> None:
    r = await run(get_time, {"timezone": zone})
    assert r.error is not None
    assert r.error.code is ErrorCode.TOOL_FAILED
    assert "Unknown timezone" in r.error.message
    assert len(r.error.message) < 200


# ---- file tools ---------------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    (root / "sub").mkdir(parents=True)
    (root / "notes.txt").write_text("hello from notes")
    (root / "sub" / "deep.txt").write_text("deep")
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    return root


def file_tools(root: Path) -> dict[str, Any]:
    return {t.name: t for t in make_file_tools(root)}


async def invoke(root: Path, name: str, args: Any, **kw: Any) -> ToolResult:
    tools = file_tools(root)
    router = ToolRouter(ToolRegistry(list(tools.values())), **kw)
    return await router.dispatch(ToolCall(id="c", name=name, arguments=args))


async def test_read_file(files: Path) -> None:
    r = await invoke(files, "read_file", {"path": "notes.txt"})
    assert r.content == "hello from notes"
    assert (await invoke(files, "read_file", {"path": "sub/deep.txt"})).content == "deep"
    assert (await invoke(files, "read_file", {"path": "sub/../notes.txt"})).ok


async def test_read_file_errors_use_relative_paths_only(files: Path, tmp_path: Path) -> None:
    missing = await invoke(files, "read_file", {"path": "nope.txt"})
    assert missing.error is not None
    assert missing.error.message == "File not found: nope.txt"
    directory = await invoke(files, "read_file", {"path": "sub"})
    assert directory.error is not None
    assert directory.error.message == "sub is a directory; use list_dir instead."
    for r in (missing, directory):
        assert str(tmp_path) not in r.text  # the real location never reaches the model


@pytest.mark.parametrize(
    "path", ["../secret.txt", "/etc/passwd", "sub/../../secret.txt", "..", "/", "~/.ssh/id_rsa"]
)
async def test_read_file_cannot_escape_the_directory(files: Path, path: str) -> None:
    r = await invoke(files, "read_file", {"path": path})
    assert not r.ok
    assert "TOP SECRET" not in r.text
    assert "root:" not in r.text


async def test_read_file_cannot_follow_symlinks_out(files: Path, tmp_path: Path) -> None:
    os.symlink(tmp_path / "secret.txt", files / "link.txt")
    os.symlink(tmp_path, files / "outside")
    for path in ("link.txt", "outside/secret.txt"):
        r = await invoke(files, "read_file", {"path": path})
        assert r.error is not None
        assert r.error.code is ErrorCode.ARGUMENTS_INVALID
        assert "TOP SECRET" not in r.text


async def test_read_file_caps_what_it_reads(files: Path) -> None:
    (files / "big.txt").write_text("x" * (MAX_READ_BYTES + 5000))
    r = await invoke(files, "read_file", {"path": "big.txt"})
    assert r.ok
    assert r.content is not None
    assert "[file continues beyond 100 KB]" in r.content or r.truncated
    (files / "bin.dat").write_bytes(b"\xff\xfe\x00binary")
    assert (await invoke(files, "read_file", {"path": "bin.dat"})).ok  # undecodable bytes: replaced


async def test_list_dir(files: Path) -> None:
    assert (await invoke(files, "list_dir", {})).content == '["notes.txt","sub/"]'
    assert (await invoke(files, "list_dir", {"path": "sub"})).content == '["deep.txt"]'
    assert (await invoke(files, "list_dir", {"path": "."})).content == '["notes.txt","sub/"]'
    bad = await invoke(files, "list_dir", {"path": "notes.txt"})
    assert bad.error is not None
    assert bad.error.message == "Not a directory: notes.txt"
    assert (await invoke(files, "list_dir", {"path": "../"})).error is not None


async def test_list_dir_caps_huge_directories(files: Path) -> None:
    for i in range(MAX_LISTED + 25):
        (files / "sub" / f"f{i:04d}.txt").write_text("")
    r = await invoke(files, "list_dir", {"path": "sub"})
    assert r.ok
    assert r.content is not None
    assert "and 26 more" in r.content  # 225 files + deep.txt, minus the 200 shown


async def test_write_file(files: Path) -> None:
    r = await invoke(files, "write_file", {"path": "new.txt", "content": "hi"})
    assert r.content == "Wrote 2 characters to new.txt"
    assert (files / "new.txt").read_text() == "hi"


async def test_write_file_creates_parent_directories_inside_the_root(files: Path) -> None:
    r = await invoke(files, "write_file", {"path": "a/b/c.txt", "content": "x"})
    assert r.ok
    assert (files / "a" / "b" / "c.txt").read_text() == "x"


async def test_write_file_refuses_to_overwrite_unless_asked(files: Path) -> None:
    r = await invoke(files, "write_file", {"path": "notes.txt", "content": "new"})
    assert r.error is not None
    assert "File already exists: notes.txt" in r.error.message
    assert "overwrite=true" in r.error.message
    assert (files / "notes.txt").read_text() == "hello from notes"  # untouched
    ok = await invoke(
        files, "write_file", {"path": "notes.txt", "content": "new", "overwrite": True}
    )
    assert ok.ok
    assert (files / "notes.txt").read_text() == "new"


async def test_write_file_cannot_escape_or_write_through_symlinks(
    files: Path, tmp_path: Path
) -> None:
    for path in ("../evil.txt", "/tmp/toolrouter-evil.txt", "sub/../../evil.txt"):
        r = await invoke(files, "write_file", {"path": path, "content": "x"})
        assert not r.ok
    os.symlink(tmp_path / "secret.txt", files / "link.txt")
    r = await invoke(
        files, "write_file", {"path": "link.txt", "content": "pwned", "overwrite": True}
    )
    assert not r.ok
    assert (tmp_path / "secret.txt").read_text() == "TOP SECRET"
    assert not (tmp_path / "evil.txt").exists()
    assert not Path("/tmp/toolrouter-evil.txt").exists()


async def test_write_file_limits_and_directories(files: Path) -> None:
    big = await invoke(files, "write_file", {"path": "big.txt", "content": "x" * 50_001})
    assert big.error is not None
    assert big.error.code is ErrorCode.ARGUMENTS_INVALID
    assert not (files / "big.txt").exists()
    onto_dir = await invoke(files, "write_file", {"path": "sub", "content": "x"})
    assert onto_dir.error is not None
    assert onto_dir.error.message == "sub is a directory."


async def test_read_only_policy_blocks_writes_but_not_reads(files: Path) -> None:
    denied = await invoke(
        files, "write_file", {"path": "x.txt", "content": "x"}, policy=ReadOnlyPolicy()
    )
    assert denied.error is not None
    assert denied.error.code is ErrorCode.POLICY_DENIED
    assert not (files / "x.txt").exists()
    assert (await invoke(files, "read_file", {"path": "notes.txt"}, policy=ReadOnlyPolicy())).ok


def test_demo_tool_set_and_side_effects(files: Path) -> None:
    tools = {t.name: t for t in make_demo_tools(files)}
    assert list(tools) == ["calculator", "get_time", "read_file", "list_dir", "write_file"]
    assert tools["write_file"].side_effect is SideEffect.WRITE
    assert all(
        tools[n].side_effect is not SideEffect.WRITE
        for n in ("calculator", "get_time", "read_file", "list_dir")
    )
    ToolRegistry(list(tools.values()))  # schemas export cleanly
