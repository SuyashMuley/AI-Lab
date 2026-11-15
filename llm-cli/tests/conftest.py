from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No test may see the developer's real env vars, config files or .env."""
    for key in list(os.environ):
        if key.startswith("LLMCLI_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LLMCLI_CONFIG_HOME", str(tmp_path / "userconf"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work
