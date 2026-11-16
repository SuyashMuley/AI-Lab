"""Console entry point: a clear message, not a traceback, when the CLI extras are missing."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from toolrouter.cli import app
    except ModuleNotFoundError as exc:
        if exc.name in {"typer", "rich", "ollama", "httpx"}:
            sys.stderr.write(
                f"error: the toolrouter CLI needs an optional dependency ({exc.name}).\n"
                "Install it with:  pip install 'toolrouter[cli]'\n"
            )
            raise SystemExit(1) from None
        raise
    app(prog_name="toolrouter")
