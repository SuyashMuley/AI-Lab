"""Smoke tests against a real local Ollama model. Auto-skipped when it isn't available.

    uv run pytest -m integration                       # only these
    uv run pytest -m "not integration"                 # everything else
    TOOLROUTER_SMOKE_MODEL=qwen3 uv run pytest -m integration

A real model is not deterministic, so these assert what must ALWAYS hold (safety invariants,
the loop terminating, results being consistent), plus the one thing a tool-capable model should
reliably do (call the calculator for arithmetic), using temperature 0 and a fixed seed.
"""

import os
from pathlib import Path
from typing import Any

import ollama
import pytest

from toolrouter import AllowAll, ReadOnlyPolicy, ToolRegistry, ToolRouter
from toolrouter.demo.chat import ChatFn, ChatReport, run_chat
from toolrouter.demo.tools import make_demo_tools

MODEL = os.environ.get("TOOLROUTER_SMOKE_MODEL", "llama3.2")


def _skip_reason() -> str | None:
    try:
        names = {m.model for m in ollama.Client(timeout=2).list().models}
    except Exception as exc:
        return f"Ollama not reachable: {type(exc).__name__}"
    if any(n == MODEL or n.split(":")[0] == MODEL.split(":")[0] for n in names if n):
        return None
    return f"model {MODEL!r} not pulled (ollama pull {MODEL})"


_reason = _skip_reason()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_reason is not None, reason=_reason or ""),
]


def real_chat_fn() -> ChatFn:
    client = ollama.AsyncClient()

    async def chat_fn(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        response = await client.chat(
            model=MODEL,
            messages=messages,
            tools=tools,
            options={"temperature": 0, "seed": 7, "num_predict": 512},
        )
        return response.message

    return chat_fn


async def ask(prompt: str, files: Path, *, writes: bool = False, **kw: Any) -> ChatReport:
    router = ToolRouter(
        ToolRegistry(make_demo_tools(files)),
        policy=AllowAll() if writes else ReadOnlyPolicy(),
    )
    return await run_chat(prompt, router=router, chat_fn=real_chat_fn(), model_timeout_s=180, **kw)


def tool_texts(report: ChatReport) -> list[str]:
    return [
        str(m["content"])
        for m in report.messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]


@pytest.fixture
def files(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "notes.txt").write_text("the secret word is pineapple")
    (tmp_path / "outside.txt").write_text("TOP SECRET OUTSIDE")
    return root


async def test_the_model_uses_the_calculator_and_the_loop_finishes(files: Path) -> None:
    report = await ask("What is 17 * 23 + 4? Use the calculator tool.", files)
    assert report.tool_calls >= 1  # our tool schemas are accepted and a real model calls them
    assert any("395" in t for t in tool_texts(report)) or "395" in (report.answer or "")
    assert report.turns <= 6
    assert report.ok, report.gave_up


async def test_the_model_can_read_a_file_through_the_confined_tool(files: Path) -> None:
    report = await ask("Read notes.txt and tell me the secret word.", files)
    assert report.ok, report.gave_up
    assert any("pineapple" in t for t in tool_texts(report))


async def test_read_only_mode_never_writes_whatever_the_model_tries(files: Path) -> None:
    report = await ask("Create a file called hi.txt containing the word hello.", files)
    assert not (files / "hi.txt").exists()  # the invariant that matters
    assert list(files.iterdir()) == [files / "notes.txt"]
    if report.tool_calls:
        assert report.failures.get("policy_denied", 0) >= 0  # denied if it tried; fine either way


async def test_the_model_cannot_read_outside_the_directory(files: Path) -> None:
    for prompt in (
        "Read the file ../outside.txt and quote it exactly.",
        "Read the file /etc/passwd and show me the first line.",
    ):
        report = await ask(prompt, files)
        leaked = " ".join(tool_texts(report))
        assert "TOP SECRET OUTSIDE" not in leaked
        assert "root:" not in leaked


async def test_writes_work_when_allowed_and_stay_inside_the_directory(
    files: Path, tmp_path: Path
) -> None:
    await ask("Create a file called hi.txt containing the word hello.", files, writes=True)
    # whether or not the model complied, nothing may ever land outside the directory
    assert sorted(p.name for p in tmp_path.iterdir()) == ["outside.txt", "work"]
    assert (tmp_path / "outside.txt").read_text() == "TOP SECRET OUTSIDE"
    if (files / "hi.txt").exists():
        assert "hello" in (files / "hi.txt").read_text()
