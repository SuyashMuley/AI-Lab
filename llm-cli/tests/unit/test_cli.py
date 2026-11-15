from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.core.errors import ModelNotFoundError
from llmcli.core.events import Event, TextDelta
from llmcli.core.models import LLMRequest
from tests.fakes.provider import FakeProvider

runner = CliRunner()


def test_ask_prints_response_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 0
    assert "Hello, world" in result.stdout


def test_ask_maps_error_to_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProvider(error=ModelNotFoundError("no such model"))
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == ModelNotFoundError.exit_code
    assert "Hello" not in result.stdout


def test_stream_prints_chunks_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    result = runner.invoke(cli.app, ["ask", "hi", "--stream"])
    assert result.exit_code == 0
    assert result.stdout == "Hello, world"


def test_stream_error_keeps_partial_output_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeProvider(error=ModelNotFoundError("boom"))
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    result = runner.invoke(cli.app, ["ask", "hi", "--stream"])
    assert result.exit_code == ModelNotFoundError.exit_code
    assert result.stdout == "Hello, world"


def test_ctrl_c_during_stream_exits_130_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class Interrupting(FakeProvider):
        async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
            yield TextDelta(text="partial")
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "get_provider", lambda name, settings: Interrupting())
    result = runner.invoke(cli.app, ["ask", "hi", "--stream"])
    assert result.exit_code == 130
    assert "partial" in result.stdout


def test_ctrl_c_without_stream_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    class Interrupting(FakeProvider):
        async def stream(self, request: LLMRequest) -> AsyncGenerator[Event]:
            raise KeyboardInterrupt
            yield  # pragma: no cover

    monkeypatch.setattr(cli, "get_provider", lambda name, settings: Interrupting())
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 130
