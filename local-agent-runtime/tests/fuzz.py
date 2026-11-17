"""Scale the property (fuzz) tests: `FUZZ_SCALE=20 uv run pytest tests/property`."""

import os

from hypothesis import settings


def fuzz(examples: int) -> settings:
    scale = max(1, int(os.environ.get("FUZZ_SCALE", "1")))
    return settings(max_examples=examples * scale, deadline=None)
