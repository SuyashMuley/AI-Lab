"""Tools with secret arguments. SENTINEL marks values that must never leak."""

from typing import Annotated, Any

from pydantic import SecretStr

from toolrouter import Secret, StrictModel, tool

SENTINEL = "ZXQV-SECRET-ZXQV"


class Creds(StrictModel):
    user: str
    password: SecretStr


class Item(StrictModel):
    name: str
    token: Annotated[str, Secret()]


@tool
def login(user: str, password: SecretStr) -> str:
    """Log in."""
    return f"hello {user}"


@tool
def call_api(url: str, api_key: Annotated[str, Secret()], retries: int = 3) -> str:
    """Call an API."""
    return url


@tool
def with_default(note: str = "n", token: Annotated[str, Secret()] = "hunter2-default") -> str:
    """A secret with a default value."""
    return note


@tool
def nested(creds: Creds, items: list[Item]) -> str:
    """Nested secrets."""
    return creds.user


@tool
def optional_secret(note: str, key: SecretStr | None = None) -> str:
    """An optional secret."""
    return note


@tool
def secret_map(headers: dict[str, SecretStr], label: str) -> str:
    """A map of secrets."""
    return label


@tool
def union_tool(target: Item | Creds) -> str:
    """A union of models, each with a secret."""
    return "ok"


@tool
def freeform(meta: dict[str, Any], label: str = "x") -> str:
    """Free-form metadata (values are not secret)."""
    return label


@tool
def secret_list(tokens: list[Annotated[str, Secret()]]) -> str:
    """A list of secrets."""
    return str(len(tokens))


ALL = [
    login,
    call_api,
    with_default,
    nested,
    optional_secret,
    secret_map,
    union_tool,
    freeform,
    secret_list,
]
