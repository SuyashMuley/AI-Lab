from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.core.errors import InvalidRequestError, SchemaValidationError
from llmcli.core.models import LLMRequest, Message
from llmcli.services.llm import LLMService
from llmcli.services.structured import SchemaMismatch, parse_and_validate
from tests.fakes.provider import FakeProvider

runner = CliRunner()
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def req() -> LLMRequest:
    return LLMRequest(model="m", messages=[Message.of("user", "who?")])


def test_parse_handles_fences_and_prose() -> None:
    assert parse_and_validate('```json\n{"name":"a","age":1}\n```', SCHEMA)["age"] == 1
    assert parse_and_validate('Sure! {"name":"a","age":1} hope that helps', SCHEMA)["name"] == "a"


def test_parse_rejects_bad_json_and_schema_mismatch() -> None:
    with pytest.raises(SchemaMismatch, match="not valid JSON"):
        parse_and_validate("nope", SCHEMA)
    with pytest.raises(SchemaMismatch, match="age"):
        parse_and_validate('{"name":"a","age":"old"}', SCHEMA)


async def test_valid_first_try_and_native_schema_passed() -> None:
    fake = FakeProvider(responses=[['{"name":"Ada","age":36}']])
    data, resp = await LLMService(fake).generate_structured(req(), SCHEMA)
    assert data == {"name": "Ada", "age": 36}
    assert fake.requests[0].response_schema == SCHEMA
    assert "JSON schema" in (fake.requests[0].system or "")
    assert resp.usage.latency_ms is not None


async def test_non_native_provider_gets_prompt_only() -> None:
    fake = FakeProvider(responses=[['{"name":"Ada","age":36}']], native_schema=False)
    await LLMService(fake).generate_structured(req(), SCHEMA)
    assert fake.requests[0].response_schema is None
    assert "JSON schema" in (fake.requests[0].system or "")


async def test_invalid_then_valid_retries_with_error_fed_back() -> None:
    fake = FakeProvider(responses=[['{"name":"Ada","age":"old"}'], ['{"name":"Ada","age":36}']])
    data, resp = await LLMService(fake).generate_structured(req(), SCHEMA)
    assert data["age"] == 36
    assert len(fake.requests) == 2
    retry_msgs = fake.requests[1].messages
    assert retry_msgs[-2].role == "assistant"
    assert "age" in retry_msgs[-1].text  # the validation error is sent back
    assert resp.usage.input_tokens == 6  # usage summed across attempts


async def test_exhausted_retries_raise_clear_error() -> None:
    fake = FakeProvider(responses=[["not json"]])
    with pytest.raises(SchemaValidationError, match="after 3 attempts") as ei:
        await LLMService(fake, schema_retries=2).generate_structured(req(), SCHEMA)
    assert len(fake.requests) == 3
    assert ei.value.last_output == "not json"


async def test_invalid_schema_definition_rejected_before_calling_model() -> None:
    fake = FakeProvider()
    with pytest.raises(InvalidRequestError):
        await LLMService(fake).generate_structured(req(), {"type": "nonsense"})
    assert fake.requests == []


def schema_file(tmp_path: Path) -> Path:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(SCHEMA))
    return p


def test_cli_schema_prints_validated_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeProvider(responses=[['{"age": 36, "name": "Ada"}']])
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    r = runner.invoke(cli.app, ["ask", "who?", "--schema", str(schema_file(tmp_path)), "--usage"])
    assert r.exit_code == 0
    assert json.loads(r.stdout) == {"name": "Ada", "age": 36}
    assert "tokens:" in r.stderr


def test_cli_schema_failure_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeProvider(responses=[["nope"]])
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    r = runner.invoke(cli.app, ["ask", "who?", "--schema", str(schema_file(tmp_path))])
    assert r.exit_code == SchemaValidationError.exit_code
    assert r.stdout == ""
    assert "did not match" in r.stderr


def test_cli_schema_bad_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    missing = runner.invoke(cli.app, ["ask", "x", "--schema", str(tmp_path / "nope.json")])
    assert missing.exit_code == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert runner.invoke(cli.app, ["ask", "x", "--schema", str(bad)]).exit_code == 2


def test_cli_schema_with_stream_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: FakeProvider())
    r = runner.invoke(cli.app, ["ask", "x", "--stream", "--schema", str(schema_file(tmp_path))])
    assert r.exit_code == 2
