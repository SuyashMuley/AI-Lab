from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.config.settings import Settings, project_config_path, user_config_path
from tests.fakes.provider import FakeProvider

runner = CliRunner()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_defaults() -> None:
    assert Settings().model == "llama3.2"


def test_precedence_flags_env_project_user_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    write(user_config_path(), 'model = "from-user"\nprovider = "user-prov"\nollama_host = "u"\n')
    assert Settings().model == "from-user"

    write(project_config_path(), 'model = "from-project"\nprovider = "proj-prov"\n')
    s = Settings()
    assert (s.model, s.provider, s.ollama_host) == ("from-project", "proj-prov", "u")

    monkeypatch.setenv("LLMCLI_MODEL", "from-env")
    assert Settings().model == "from-env"
    assert Settings().provider == "proj-prov"  # env didn't set it

    assert Settings(model="from-flag").model == "from-flag"


def test_secrets_in_config_files_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    write(user_config_path(), 'api_key = "sk-from-file"\n')
    assert Settings().api_key is None
    monkeypatch.setenv("LLMCLI_API_KEY", "sk-from-env")
    assert Settings().api_key is not None
    assert Settings().api_key.get_secret_value() == "sk-from-env"  # type: ignore[union-attr]


def test_config_set_get_roundtrip() -> None:
    r = runner.invoke(cli.app, ["config", "set", "model", "llama3.2:1b"])
    assert r.exit_code == 0
    assert runner.invoke(cli.app, ["config", "get", "model"]).stdout.strip() == "llama3.2:1b"
    assert "llama3.2:1b" in user_config_path().read_text()


def test_config_set_project_writes_project_file() -> None:
    runner.invoke(cli.app, ["config", "set", "timeout", "12.5", "--project"])
    assert "12.5" in project_config_path().read_text()
    assert runner.invoke(cli.app, ["config", "get", "timeout"]).stdout.strip() == "12.5"


def test_config_set_validates_values_and_keys() -> None:
    assert runner.invoke(cli.app, ["config", "set", "timeout", "abc"]).exit_code == 2
    assert runner.invoke(cli.app, ["config", "set", "timeout", "-1"]).exit_code == 2
    assert runner.invoke(cli.app, ["config", "set", "nope", "x"]).exit_code == 2
    assert not user_config_path().exists()


def test_config_set_refuses_secrets() -> None:
    r = runner.invoke(cli.app, ["config", "set", "api_key", "sk-supersecret"])
    assert r.exit_code == 2
    assert "sk-supersecret" not in r.stdout + r.stderr
    assert "LLMCLI_API_KEY" in r.stderr
    assert not user_config_path().exists()


def test_secrets_never_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCLI_API_KEY", "sk-supersecret")
    for args in (["config", "list"], ["config", "get", "api_key"]):
        r = runner.invoke(cli.app, args)
        assert "sk-supersecret" not in r.stdout + r.stderr
    assert "api_key = (set)" in runner.invoke(cli.app, ["config", "list"]).stdout
    assert "sk-supersecret" not in repr(Settings())


def test_invalid_env_value_gives_clean_error_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMCLI_API_KEY", "sk-supersecret")
    monkeypatch.setenv("LLMCLI_TIMEOUT", "soon")
    r = runner.invoke(cli.app, ["ask", "hi"])
    assert r.exit_code == 2
    assert "timeout" in r.stderr
    assert "sk-supersecret" not in r.stdout + r.stderr
    assert "Traceback" not in r.stderr


def test_alias_resolves_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    runner.invoke(cli.app, ["config", "set", "alias.fast", "llama3.2:1b"])
    fake = FakeProvider()
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    assert runner.invoke(cli.app, ["ask", "hi", "-m", "fast"]).exit_code == 0
    assert fake.requests[0].model == "llama3.2:1b"
    assert "alias.fast = llama3.2:1b" in runner.invoke(cli.app, ["config", "list"]).stdout


def test_config_unset() -> None:
    runner.invoke(cli.app, ["config", "set", "model", "x"])
    runner.invoke(cli.app, ["config", "unset", "model"])
    assert runner.invoke(cli.app, ["config", "get", "model"]).stdout.strip() == "llama3.2"
