"""Read/write the TOML config files behind `llm config`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, cast

import tomli_w
from pydantic import TypeAdapter, ValidationError

from llmcli.config.settings import SECRET_FIELDS, Settings
from llmcli.core.errors import InvalidRequestError

SETTABLE = sorted(k for k in Settings.model_fields if k not in SECRET_FIELDS | {"aliases"})
ALIAS_PREFIXES = ("alias.", "aliases.")


def load_table(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise InvalidRequestError(f"Malformed config file {path}: {exc}") from exc


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode())


def _check_key(key: str) -> tuple[str, str | None]:
    """Return (field, alias_name). Rejects secrets and unknown keys."""
    for prefix in ALIAS_PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            return "aliases", key[len(prefix) :]
    root = key.split(".")[0]
    if root in SECRET_FIELDS:
        raise InvalidRequestError(
            f"{root!r} is a secret and cannot be stored in a config file. "
            f"Set the LLMCLI_{root.upper()} environment variable instead."
        )
    if key not in SETTABLE:
        raise InvalidRequestError(
            f"Unknown config key {key!r}. Valid: {', '.join(SETTABLE)}, alias.<name>"
        )
    return key, None


def set_key(path: Path, key: str, raw: str) -> Any:
    field, alias = _check_key(key)
    data = load_table(path)
    if alias is not None:
        data.setdefault("aliases", {})[alias] = raw
        value: Any = raw
    else:
        info = Settings.model_fields[field]
        # The field's type plus its constraints (gt/ge), e.g. timeout > 0.
        typ = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
        try:
            value = TypeAdapter(cast(Any, typ)).validate_python(raw)
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"]
            raise InvalidRequestError(f"Invalid value for {field}: {reason}") from None
        data[field] = value
    _save(path, data)
    return value


def unset_key(path: Path, key: str) -> bool:
    field, alias = _check_key(key)
    data = load_table(path)
    if alias is not None:
        removed = data.get("aliases", {}).pop(alias, None) is not None
    else:
        removed = data.pop(field, None) is not None
    if removed:
        _save(path, data)
    return removed
