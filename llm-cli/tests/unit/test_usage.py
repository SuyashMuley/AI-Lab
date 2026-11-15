from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.core.models import Usage
from llmcli.services.pricing import estimate_cost
from llmcli.ui.console import format_usage
from tests.fakes.provider import FakeProvider

runner = CliRunner()


def test_format_usage() -> None:
    line = format_usage(Usage(input_tokens=10, output_tokens=20, latency_ms=2000), cost=0.0123)
    assert "10 in / 20 out" in line
    assert "2.00s" in line
    assert "10.0 tok/s" in line
    assert "~$0.0123 (estimate)" in line


def test_format_usage_without_cost_or_latency() -> None:
    assert format_usage(Usage(input_tokens=1, output_tokens=2)) == "tokens: 1 in / 2 out"


def test_cost_estimate_uses_wildcards_and_returns_none_when_unknown() -> None:
    assert estimate_cost("ollama", "anything", Usage(input_tokens=5, output_tokens=5)) == 0
    assert estimate_cost("nobody", "x", Usage(input_tokens=5)) is None


@pytest.mark.parametrize("stream", [False, True])
def test_usage_goes_to_stderr_not_stdout(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    args = ["ask", "hi", "--usage"] + (["--stream"] if stream else [])
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0
    assert "tokens:" in r.stderr
    assert "tokens:" not in r.stdout
    assert "Hello, world" in r.stdout


def test_no_usage_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    assert "tokens:" not in runner.invoke(cli.app, ["ask", "hi"]).stderr
