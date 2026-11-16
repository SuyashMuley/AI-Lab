import os
from pathlib import Path
from typing import Annotated

import pytest

from toolrouter import Confined, ErrorCode, ToolCallError, ToolRegistry, tool, validate_call
from toolrouter.schema import tool_schema


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a")
    (root / "sub" / "b.txt").write_text("b")
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "passwd").write_text("top secret")
    return root


def read_tool(root: Path, **kw: bool):  # type: ignore[no-untyped-def]
    @tool
    def read_file(path: Annotated[Path, Confined(root, **kw)]) -> str:
        """Read a file."""
        return path.read_text()

    return read_file


def check(t, path: str) -> Path:  # type: ignore[no-untyped-def]
    return validate_call(t, {"path": path}).path  # type: ignore[no-any-return,attr-defined]


def rejected(t, path: str) -> str:  # type: ignore[no-untyped-def]
    with pytest.raises(ToolCallError) as ei:
        validate_call(t, {"path": path})
    assert ei.value.info.code is ErrorCode.ARGUMENTS_INVALID
    return ei.value.info.message


def test_relative_paths_resolve_inside_root(sandbox: Path) -> None:
    t = read_tool(sandbox)
    assert check(t, "a.txt") == (sandbox / "a.txt").resolve()
    assert check(t, "sub/b.txt") == (sandbox / "sub" / "b.txt").resolve()
    assert check(t, "sub/../a.txt") == (sandbox / "a.txt").resolve()
    assert check(t, ".") == sandbox.resolve()


def test_absolute_path_inside_root_allowed(sandbox: Path) -> None:
    assert check(read_tool(sandbox), str(sandbox / "a.txt")) == (sandbox / "a.txt").resolve()


@pytest.mark.parametrize(
    "bad", ["../secret/passwd", "sub/../../secret/passwd", "..", "../data/../secret", "sub/../.."]
)
def test_dotdot_escapes_rejected(sandbox: Path, bad: str) -> None:
    assert "outside the allowed directory" in rejected(read_tool(sandbox), bad)


def test_absolute_path_outside_rejected(sandbox: Path) -> None:
    t = read_tool(sandbox)
    assert "outside" in rejected(t, str(sandbox.parent / "secret" / "passwd"))
    assert "outside" in rejected(t, "/etc/passwd")


def test_sibling_directory_with_same_prefix_rejected(sandbox: Path) -> None:
    evil = sandbox.parent / "data-evil"
    evil.mkdir()
    (evil / "x").write_text("x")
    assert "outside" in rejected(read_tool(sandbox), str(evil / "x"))


def test_symlink_file_pointing_outside_rejected(sandbox: Path) -> None:
    os.symlink(sandbox.parent / "secret" / "passwd", sandbox / "link")
    assert "outside" in rejected(read_tool(sandbox), "link")


def test_symlinked_directory_pointing_outside_rejected(sandbox: Path) -> None:
    os.symlink(sandbox.parent / "secret", sandbox / "dirlink")
    assert "outside" in rejected(read_tool(sandbox), "dirlink/passwd")


def test_dangling_symlink_to_outside_rejected(sandbox: Path) -> None:
    os.symlink(sandbox.parent / "nowhere", sandbox / "dangling")
    assert "outside" in rejected(read_tool(sandbox), "dangling")


def test_symlink_staying_inside_is_allowed_and_resolved(sandbox: Path) -> None:
    os.symlink(sandbox / "a.txt", sandbox / "alias")
    assert check(read_tool(sandbox), "alias") == (sandbox / "a.txt").resolve()


def test_symlinked_root_is_handled(sandbox: Path, tmp_path: Path) -> None:
    link_root = tmp_path / "rootlink"
    os.symlink(sandbox, link_root)
    t = read_tool(link_root)
    assert check(t, "a.txt") == (sandbox / "a.txt").resolve()
    assert "outside" in rejected(t, "../secret/passwd")


def test_nonexistent_path_inside_root_allowed_by_default(sandbox: Path) -> None:
    assert check(read_tool(sandbox), "new/file.txt") == (sandbox / "new" / "file.txt").resolve()


def test_must_exist(sandbox: Path) -> None:
    t = read_tool(sandbox, must_exist=True)
    assert check(t, "a.txt") == (sandbox / "a.txt").resolve()
    assert "does not exist" in rejected(t, "missing.txt")


def test_null_byte_rejected(sandbox: Path) -> None:
    assert "null byte" in rejected(read_tool(sandbox), "a.txt\x00.png")


def test_error_never_reveals_root_location(sandbox: Path) -> None:
    msg = rejected(read_tool(sandbox), "../secret/passwd")
    assert str(sandbox) not in msg
    assert str(sandbox.parent) not in msg


def test_non_string_value_still_a_type_error(sandbox: Path) -> None:
    t = read_tool(sandbox)
    with pytest.raises(ToolCallError) as ei:
        validate_call(t, {"path": 5})
    assert ei.value.info.code is ErrorCode.ARGUMENTS_INVALID


async def test_handler_receives_the_resolved_real_path(sandbox: Path) -> None:
    t = read_tool(sandbox)
    args = validate_call(t, {"path": "sub/../a.txt"})
    assert await t.invoke(args) == "a"


def test_schema_is_a_plain_path_string(sandbox: Path) -> None:
    t = read_tool(sandbox)
    reg = ToolRegistry([t])
    assert tool_schema(reg["read_file"])["properties"]["path"] == {
        "type": "string",
        "format": "path",
    }
