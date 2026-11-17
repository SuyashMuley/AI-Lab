"""The seam between the loop and "a model".

The loop needs one thing from a model: given a conversation (and the tools on offer), produce the
next assistant turn. Message, tool and usage types are reused from `llmcli`, so a transcript is
the same data everywhere in this repo.
"""

from __future__ import annotations

from typing import Protocol

from llmcli.core.models import Message, ToolCallPart, ToolSpec, Usage
from pydantic import BaseModel, ConfigDict, Field


class TurnRequest(BaseModel):
    """Everything a model needs for one reply."""

    model_config = ConfigDict(frozen=True)

    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None


class ModelTurn(BaseModel):
    """One assistant reply: text and/or tool calls, with what it cost."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: list[ToolCallPart] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None


class ModelClient(Protocol):
    """Produces assistant turns. Implementations raise on failure; the loop contains it.

    Contract: a call that produced tool calls must never be silently replayed by the
    implementation (a tool call may already have caused a side effect elsewhere); retrying
    *before any output* is fine.
    """

    async def respond(self, request: TurnRequest) -> ModelTurn: ...
