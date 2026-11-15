"""JSON-schema helpers for structured output (M8)."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import exceptions as js_exc
from jsonschema import validators

from llmcli.core.errors import InvalidRequestError


class SchemaMismatch(Exception):
    """Output was not valid JSON, or did not satisfy the schema."""


def check_schema(schema: dict[str, Any]) -> None:
    try:
        validators.validator_for(schema).check_schema(schema)
    except js_exc.SchemaError as exc:
        raise InvalidRequestError(f"Invalid JSON schema: {exc.message}") from exc


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse(text: str) -> Any:
    candidates = [text.strip()]
    if m := _FENCE.search(text):
        candidates.append(m.group(1).strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # Last resort: first JSON value embedded in prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise SchemaMismatch("response was not valid JSON")


def parse_and_validate(text: str, schema: dict[str, Any]) -> Any:
    data = _parse(text)
    validator = validators.validator_for(schema)(schema)
    if error := js_exc.best_match(validator.iter_errors(data)):
        where = ".".join(str(p) for p in error.absolute_path) or "<root>"
        raise SchemaMismatch(f"{where}: {error.message}")
    return data


def build_system(existing: str | None, schema: dict[str, Any]) -> str:
    instruction = (
        "Respond with a single JSON value that conforms to this JSON schema, "
        "and nothing else (no prose, no code fences):\n" + json.dumps(schema)
    )
    return f"{existing}\n\n{instruction}" if existing else instruction


def retry_prompt(error: str) -> str:
    return (
        f"Your previous response was invalid: {error}\n"
        "Respond again with only valid JSON that conforms to the schema."
    )
