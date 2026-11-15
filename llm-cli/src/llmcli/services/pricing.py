"""Optional cost estimate from a data-file price table. Always an estimate."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from functools import cache
from importlib import resources

from llmcli.core.models import Usage


@cache
def _table() -> dict[str, dict[str, float]]:
    raw = resources.files("llmcli.data").joinpath("prices.json").read_text("utf-8")
    prices: dict[str, dict[str, float]] = json.loads(raw)["prices"]
    return prices


def estimate_cost(provider: str, model: str, usage: Usage) -> float | None:
    """USD estimate, or None when the model has no entry in the table."""
    key = f"{provider}/{model}"
    for pattern, price in _table().items():
        if fnmatch(key, pattern):
            return (
                usage.input_tokens * price["input"] + usage.output_tokens * price["output"]
            ) / 1e6
    return None
