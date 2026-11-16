import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from toolrouter import (
    ReadOnlyPolicy,
    RecordingHook,
    ToolCall,
    ToolRegistry,
    ToolRouter,
    cli,
)
from toolrouter.formats import anthropic, ollama, openai

ROOT = Path(__file__).parents[2]
TOOLS = str(ROOT / "examples" / "tools.py")
runner = CliRunner()


def invoke(*args: str) -> Any:
    return runner.invoke(cli.app, list(args))


def test_the_example_tools_load_and_describe_themselves() -> None:
    r = invoke("list", "--tools", TOOLS)
    assert r.exit_code == 0
    for name in ("get_forecast", "convert_currency", "read_note", "send_email"):
        assert name in r.stdout
    assert "send_email  [write]" in r.stdout
    assert "get_forecast  [read]" in r.stdout
    assert 'get_forecast(city: string, units?: "c" | "f" = "c")' in r.stdout


def test_example_schemas_export_for_every_provider() -> None:
    registry = ToolRegistry(_example_tools())
    for payload in (
        anthropic.tools(registry),
        openai.tools(registry),
        openai.tools(registry, api="responses"),
        ollama.tools(registry),
    ):
        json.dumps(payload)
        assert len(payload) == 4
    email = json.dumps(anthropic.tool(registry["send_email"]))
    assert "sk-" not in email
    assert "writeOnly" not in email


def _example_tools() -> list[Any]:
    return cli._tools_from_module(TOOLS)


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("get_forecast", {"city": "Oslo", "units": "f"}, '"temperature":39.2'),
        ("get_forecast", {"city": "Cairo"}, '"conditions":"sunny"'),
        ("convert_currency", {"amount": 100, "source": "USD", "target": "NOK"}, "1080.00 NOK"),
        ("read_note", {"path": "welcome.txt"}, "Welcome to the toolrouter examples."),
    ],
)
def test_example_tools_work(tool: str, args: dict[str, Any], expected: str) -> None:
    r = invoke("call", tool, "--tools", TOOLS, "--args", json.dumps(args))
    assert r.exit_code == 0, r.stderr
    assert expected in r.stdout


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("get_forecast", {"city": "Atlantis"}, "Known cities: Cairo, Lima, Oslo."),
        ("convert_currency", {"amount": -5, "source": "USD", "target": "EUR"}, "greater than 0"),
        ("convert_currency", {"amount": 5, "source": "GBP", "target": "EUR"}, "'USD', 'EUR' or"),
        ("read_note", {"path": "../tools.py"}, "outside the allowed directory"),
        ("read_note", {"path": "/etc/passwd"}, "outside the allowed directory"),
        ("read_note", {"path": "missing.txt"}, "No such note: missing.txt"),
    ],
)
def test_example_tools_reject_bad_calls_helpfully(
    tool: str, args: dict[str, Any], expected: str
) -> None:
    r = invoke("call", tool, "--tools", TOOLS, "--args", json.dumps(args))
    assert r.exit_code == 1
    assert expected in r.stderr


async def test_send_email_is_a_write_with_a_secret_key() -> None:
    registry = ToolRegistry(_example_tools())
    recorder = RecordingHook()
    args = {"to": "a@b.co", "subject": "hi", "body": "x", "api_key": "sk-SUPERSECRET"}

    denied = await ToolRouter(registry, policy=ReadOnlyPolicy(), hooks=[recorder]).dispatch(
        ToolCall(id="1", name="send_email", arguments=args)
    )
    assert denied.error is not None
    assert denied.error.code.value == "policy_denied"
    assert recorder.before_events[0].arguments["api_key"] == "[REDACTED]"
    assert "SUPERSECRET" not in json.dumps(recorder.before_events[0].arguments)

    allowed = await ToolRouter(registry).dispatch(
        ToolCall(id="2", name="send_email", arguments=args)
    )
    assert allowed.content == "Queued email to a@b.co: 'hi'"

    rejected = await ToolRouter(registry).dispatch(
        ToolCall(id="3", name="send_email", arguments={**args, "api_key": "nope"})
    )
    assert rejected.error is not None
    assert rejected.error.message == "The API key was rejected."

    invalid = await ToolRouter(registry).dispatch(
        ToolCall(id="4", name="send_email", arguments={**args, "to": "not-an-email"})
    )
    assert invalid.error is not None
    assert "String should match pattern" in invalid.error.message


def test_the_quickstart_script_runs_offline_and_tells_the_story() -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert '"temperature":39.2' in out  # a call that worked
    assert "Known cities: Cairo, Lima, Oslo." in out  # a tool's own refusal
    assert "Input should be a valid number (got '100')" in out  # strict validation
    assert "outside the allowed directory" in out  # the confined path
    assert "policy_denied" in out  # the read-only policy
    assert "Did you mean 'get_forecast'?" in out  # the typo
    assert "6 tool messages ready to send back" in out
    assert "sk-123" not in out + done.stderr  # the secret never shows, even in the log lines
    assert "[REDACTED]" in done.stderr
