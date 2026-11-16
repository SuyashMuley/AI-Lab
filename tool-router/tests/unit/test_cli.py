import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest
from typer.testing import CliRunner

from toolrouter import cli
from toolrouter.demo.chat import ModelTimeoutError

runner = CliRunner()


@pytest.fixture
def files(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "notes.txt").write_text("hello from notes")
    return root


def invoke(*args: str, input: str | None = None) -> Any:
    return runner.invoke(cli.app, list(args), input=input)


# ---- list / schema ----


def test_list_shows_every_demo_tool_with_effect_and_signature(files: Path) -> None:
    r = invoke("list", "--files-dir", str(files))
    assert r.exit_code == 0
    for name in ("calculator", "get_time", "read_file", "list_dir", "write_file"):
        assert name in r.stdout
    assert "write_file  [write]" in r.stdout
    assert "calculator  [none]" in r.stdout
    assert "calculator(expression: string (1-200 chars))" in r.stdout


@pytest.mark.parametrize(
    ("fmt", "check"),
    [
        ("ollama", lambda p: p[0]["type"] == "function" and "parameters" in p[0]["function"]),
        ("anthropic", lambda p: set(p[0]) == {"name", "description", "input_schema"}),
        ("openai", lambda p: p[0]["function"]["parameters"]["type"] == "object"),
        ("openai-responses", lambda p: "function" not in p[0] and "parameters" in p[0]),
    ],
)
def test_schema_formats(files: Path, fmt: str, check: Any) -> None:
    r = invoke("schema", "--format", fmt, "--files-dir", str(files))
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert len(payload) == 5
    assert check(payload)


def test_schema_for_one_tool_and_plain_json(files: Path) -> None:
    one = json.loads(invoke("schema", "calculator", "--files-dir", str(files)).stdout)
    assert [t["function"]["name"] for t in one] == ["calculator"]
    plain = json.loads(invoke("schema", "--format", "json", "--files-dir", str(files)).stdout)
    assert set(plain) == {"calculator", "get_time", "read_file", "list_dir", "write_file"}
    assert plain["calculator"]["type"] == "object"


def test_schema_unknown_tool_is_a_usage_error(files: Path) -> None:
    r = invoke("schema", "nope", "--files-dir", str(files))
    assert r.exit_code == 2
    assert "Unknown tool 'nope'" in r.stderr
    assert "calculator" in r.stderr


# ---- call ----


def test_call_success_prints_the_content(files: Path) -> None:
    r = invoke("call", "calculator", "--args", '{"expression": "2+2"}', "--files-dir", str(files))
    assert r.exit_code == 0
    assert r.stdout.strip() == "4"


def test_call_failure_exits_1_with_the_model_facing_message_on_stderr(files: Path) -> None:
    r = invoke("call", "calculator", "--args", '{"expression": "1/0"}', "--files-dir", str(files))
    assert r.exit_code == 1
    assert r.stdout == ""
    assert "Error (tool_failed): Division by zero." in r.stderr


@pytest.mark.parametrize(
    ("tool", "args", "code"),
    [
        ("calculator", '{"expression": ', "invalid_json"),
        ("calculator", "{}", "arguments_invalid"),
        ("nope", "{}", "unknown_tool"),
    ],
)
def test_call_reports_each_kind_of_bad_call(files: Path, tool: str, args: str, code: str) -> None:
    r = invoke("call", tool, "--args", args, "--files-dir", str(files))
    assert r.exit_code == 1
    assert f"Error ({code})" in r.stderr


def test_call_json_output(files: Path) -> None:
    ok = json.loads(
        invoke(
            "call", "calculator", "-a", '{"expression": "6*7"}', "--json", "--files-dir", str(files)
        ).stdout
    )
    assert (ok["ok"], ok["content"], ok["error"]) == (True, "42", None)
    bad = invoke("call", "nope", "--json", "--files-dir", str(files))
    assert bad.exit_code == 1
    assert json.loads(bad.stdout)["error"]["code"] == "unknown_tool"


def test_call_read_only_denies_writes(files: Path) -> None:
    args = '{"path": "x.txt", "content": "x"}'
    denied = invoke("call", "write_file", "-a", args, "--read-only", "--files-dir", str(files))
    assert denied.exit_code == 1
    assert "policy_denied" in denied.stderr
    assert not (files / "x.txt").exists()
    allowed = invoke("call", "write_file", "-a", args, "--files-dir", str(files))
    assert allowed.exit_code == 0
    assert (files / "x.txt").read_text() == "x"


def test_call_cannot_escape_the_files_dir(files: Path, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    r = invoke("call", "read_file", "-a", '{"path": "../secret.txt"}', "--files-dir", str(files))
    assert r.exit_code == 1
    assert "outside the allowed directory" in r.stderr
    assert "TOP SECRET" not in r.stdout + r.stderr


# ---- --tools ----


USER_TOOLS = '''
from typing import Annotated
from toolrouter import Secret, SideEffect, tool

@tool
def double(n: int) -> int:
    """Double a number."""
    return n * 2

@tool(side_effect=SideEffect.READ)
def lookup(key: str, api_key: Annotated[str, Secret()]) -> str:
    """Look something up."""
    return f"value-for-{key}"
'''


def write_tools(tmp_path: Path, text: str = USER_TOOLS, name: str = "my_tools.py") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_tools_from_a_python_file_replace_the_demo_tools(tmp_path: Path) -> None:
    path = write_tools(tmp_path)
    listing = invoke("list", "--tools", str(path)).stdout
    assert "double" in listing
    assert "lookup" in listing
    assert "calculator" not in listing
    assert invoke("call", "double", "-a", '{"n": 21}', "--tools", str(path)).stdout.strip() == "42"


def test_coerce_flag_reaches_user_tools(tmp_path: Path) -> None:
    path = write_tools(tmp_path)
    assert invoke("call", "double", "-a", '{"n": "21"}', "--tools", str(path)).exit_code == 1
    ok = invoke("call", "double", "-a", '{"n": "21"}', "--coerce", "--tools", str(path))
    assert ok.stdout.strip() == "42"


def test_tools_from_a_dotted_module_name() -> None:
    r = invoke("list", "--tools", "tests.fakes.sample_tools")
    assert r.exit_code == 0
    assert "get_weather" in r.stdout


def test_tools_loading_errors_are_usage_errors(tmp_path: Path) -> None:
    assert invoke("list", "--tools", str(tmp_path / "missing.py")).exit_code == 2
    assert invoke("list", "--tools", "no.such.module").exit_code == 2
    broken = write_tools(tmp_path, "def (:\n", "broken.py")
    assert invoke("list", "--tools", str(broken)).exit_code == 2
    empty = write_tools(tmp_path, "x = 1\n", "empty.py")
    r = invoke("list", "--tools", str(empty))
    assert r.exit_code == 2
    assert "No tools found" in r.stderr
    bad_tool = write_tools(
        tmp_path, "from toolrouter import tool\n@tool\ndef f(x): 'doc'\n", "bad.py"
    )
    assert invoke("list", "--tools", str(bad_tool)).exit_code == 2


def test_duplicate_tool_names_across_modules_are_rejected(tmp_path: Path) -> None:
    a = write_tools(tmp_path, name="a.py")
    b = write_tools(tmp_path, name="b.py")
    r = invoke("list", "--tools", str(a), "--tools", str(b))
    assert r.exit_code == 2
    assert "already registered" in r.stderr


# ---- chat (with a scripted model instead of Ollama) ----


def scripted(
    monkeypatch: pytest.MonkeyPatch, *replies: Any, record: dict[str, Any] | None = None
) -> None:
    pending = list(replies)

    def factory(model: str, host: str | None, max_tokens: int) -> Any:
        if record is not None:
            record.update(model=model, host=host, max_tokens=max_tokens)

        async def chat_fn(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
            reply = pending.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        return chat_fn

    monkeypatch.setattr(cli, "_ollama_chat_fn", factory)


def asks(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def says(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def test_chat_prints_the_answer_to_stdout_and_the_trace_to_stderr(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch, asks("calculator", expression="17*23+4"), says("It is 395."))
    r = invoke("chat", "What is 17*23+4?", "--files-dir", str(files))
    assert r.exit_code == 0
    assert r.stdout.strip() == "It is 395."
    assert '→ calculator({"expression": "17*23+4"})' in r.stderr
    assert "✓ calculator" in r.stderr
    assert "395" in r.stderr
    assert "turns=2 tool_calls=1 repairs=0 failures: none" in r.stderr


def test_chat_quiet_prints_only_the_answer(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted(monkeypatch, asks("calculator", expression="1+1"), says("2"))
    r = invoke("chat", "q", "--quiet", "--files-dir", str(files))
    assert r.stdout.strip() == "2"
    assert r.stderr == ""


def test_chat_passes_options_to_the_model_function(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    scripted(monkeypatch, says("ok"), record=seen)
    invoke(
        "chat",
        "q",
        "-m",
        "qwen3",
        "--host",
        "http://h:1",
        "--max-tokens",
        "77",
        "--files-dir",
        str(files),
    )
    assert seen == {"model": "qwen3", "host": "http://h:1", "max_tokens": 77}


def test_chat_is_read_only_by_default(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write = asks("write_file", path="hi.txt", content="hello")
    scripted(monkeypatch, write, says("I could not write it."))
    r = invoke("chat", "write hi", "--files-dir", str(files))
    assert r.exit_code == 0
    assert "✗ write_file" in r.stderr
    assert "policy_denied" in r.stderr
    assert not (files / "hi.txt").exists()


def test_chat_allow_writes_lets_writes_through(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = asks("write_file", path="hi.txt", content="hello")
    scripted(monkeypatch, write, says("Done."))
    r = invoke("chat", "write hi", "--allow-writes", "--files-dir", str(files))
    assert r.exit_code == 0
    assert (files / "hi.txt").read_text() == "hello"


def test_chat_trace_redacts_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_tools(tmp_path)
    scripted(monkeypatch, asks("lookup", key="k", api_key="sk-live-SUPERSECRET"), says("found"))
    r = invoke("chat", "look it up", "--tools", str(path))
    assert r.exit_code == 0
    assert "[REDACTED]" in r.stderr
    assert "SUPERSECRET" not in r.stdout + r.stderr


def test_chat_gives_up_with_exit_4(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = asks("calculator", nonsense=1)
    scripted(monkeypatch, bad, bad, bad, says("unreachable"))
    r = invoke("chat", "q", "--max-repairs", "2", "--files-dir", str(files))
    assert r.exit_code == 4
    assert r.stdout == ""
    assert "could not fix its tool calls after 2 repair attempts" in r.stderr
    assert "repairs=3" in r.stderr


def test_chat_max_turns(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ok = asks("calculator", expression="1+1")
    scripted(monkeypatch, ok, ok, ok, ok)
    r = invoke("chat", "q", "--max-turns", "2", "--files-dir", str(files))
    assert r.exit_code == 4
    assert "no final answer within 2 turns" in r.stderr


@pytest.mark.parametrize(
    ("error", "needle"),
    [
        (ollama.ResponseError("model 'x' not found", 404), "ollama pull llama3.2"),
        (
            ollama.ResponseError("registry.ollama.ai/library/x does not support tools", 400),
            "cannot use tools",
        ),
        (ollama.ResponseError("boom", 500), "Ollama error (500)"),
        (ConnectionError("refused"), "Cannot reach Ollama"),
        (httpx.ConnectError("refused"), "Cannot reach Ollama"),
        (ModelTimeoutError(5), "did not answer within 5s"),
    ],
)
def test_chat_maps_model_failures_to_exit_3(
    files: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, needle: str
) -> None:
    scripted(monkeypatch, error)
    r = invoke("chat", "q", "--files-dir", str(files))
    assert r.exit_code == 3
    assert needle in r.stderr
    assert "Traceback" not in r.stderr


def test_chat_does_not_swallow_unexpected_errors(
    files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch, RuntimeError("a real bug"))
    r = invoke("chat", "q", "--files-dir", str(files))
    assert r.exit_code != 0
    assert isinstance(r.exception, RuntimeError)


def test_chat_ctrl_c_exits_130(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted(monkeypatch, KeyboardInterrupt())
    r = invoke("chat", "q", "--files-dir", str(files))
    assert r.exit_code == 130
    assert "cancelled" in r.stderr


# ---- packaging boundaries ----


def test_the_core_package_does_not_import_the_cli_dependencies() -> None:
    code = (
        "import sys, toolrouter, toolrouter.demo.chat, toolrouter.demo.tools\n"
        "bad = [m for m in ('typer', 'rich', 'ollama', 'httpx') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_module_entry_point_works() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "toolrouter", "--help"], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert "chat" in done.stdout
