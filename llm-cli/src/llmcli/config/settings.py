"""Settings. Precedence: CLI flags > env vars > project config > user config > defaults.

Secrets come from environment variables only; config files can never supply them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

SECRET_FIELDS = frozenset({"api_key"})


def user_config_path() -> Path:
    home = os.environ.get("LLMCLI_CONFIG_HOME")
    base = Path(home) if home else Path.home() / ".config" / "llmcli"
    return base / "config.toml"


def data_path() -> Path:
    """Directory for local data (conversation history)."""
    home = os.environ.get("LLMCLI_DATA_HOME")
    return Path(home) if home else Path.home() / ".local" / "share" / "llmcli"


def project_config_path() -> Path:
    return Path.cwd() / ".llmcli.toml"


class _ConfigFileSource(TomlConfigSettingsSource):
    """TOML source that ignores secret fields, so secrets stay env-only."""

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in super().__call__().items() if k not in SECRET_FIELDS}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMCLI_", env_file=".env", extra="ignore")

    provider: str = "ollama"
    model: str = "llama3.2"
    ollama_host: str | None = None
    timeout: float = Field(default=60.0, gt=0, description="Seconds of provider silence.")
    max_retries: int = Field(default=2, ge=0, description="Retries on 429/5xx/connection.")
    schema_retries: int = Field(default=2, ge=0, description="Retries on schema mismatch.")
    save_history: bool = Field(default=True, description="Save conversations (SQLite).")
    aliases: dict[str, str] = Field(default_factory=dict)
    api_key: SecretStr | None = None  # env only (LLMCLI_API_KEY); for remote providers

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win. init_settings carries CLI flags.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _ConfigFileSource(settings_cls, toml_file=project_config_path()),
            _ConfigFileSource(settings_cls, toml_file=user_config_path()),
        )
