"""Provider-agnostic request/response types. SDK types never leak past providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["user", "assistant", "system", "tool"]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    """A tool call the model made (assistant messages only).

    `arguments` is kept raw, exactly as the provider produced it: a dict, or JSON text for
    providers that send text. Validating it is the tool layer's job, not ours.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class ToolResultPart(BaseModel):
    """The outcome of a tool call, sent back to the model (tool messages only).

    `is_error` is informational: providers with an error flag use it; for the others (Ollama)
    the caller should make failures obvious in `content`.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str = Field(min_length=1)
    content: str
    is_error: bool = False


Part = Annotated[TextPart | ToolCallPart | ToolResultPart, Field(discriminator="type")]


class ToolSpec(BaseModel):
    """A tool the model may call: provider-agnostic (a JSON schema for the parameters)."""

    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Message(BaseModel):
    role: Role
    content: list[Part]

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> Any:
        # `str` shorthand: Message(role="user", content="hi")
        return [{"type": "text", "text": v}] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _parts_match_role(self) -> Message:
        """Keep conversations well formed: calls come from the assistant, results from tools."""
        for part in self.content:
            if isinstance(part, ToolCallPart) and self.role != "assistant":
                raise ValueError("tool calls can only appear in assistant messages")
            if isinstance(part, ToolResultPart) and self.role != "tool":
                raise ValueError("tool results can only appear in tool messages")
            if self.role == "tool" and not isinstance(part, ToolResultPart):
                raise ValueError("tool messages can only contain tool results")
        return self

    @classmethod
    def of(cls, role: Role, text: str) -> Message:
        return cls(role=role, content=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Sequence[ToolCallPart] = ()) -> Message:
        """An assistant turn: optional text, then any tool calls it made."""
        parts: list[Part] = [TextPart(text=text)] if text else []
        parts.extend(tool_calls)
        return cls(role="assistant", content=parts)

    @classmethod
    def tool_results(cls, results: Sequence[ToolResultPart]) -> Message:
        """One tool message carrying the results for a turn's calls, in order."""
        return cls(role="tool", content=list(results))

    @property
    def text(self) -> str:
        """The text parts only (tool calls and results are not text)."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.content if isinstance(p, ToolCallPart)]

    @property
    def results(self) -> list[ToolResultPart]:
        return [p for p in self.content if isinstance(p, ToolResultPart)]


class LLMRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, Any] | None = None
    tools: list[ToolSpec] | None = None  # tools the model may call (cannot combine with a schema)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum token counts (latency is not additive; callers set it)."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class LLMResponse(BaseModel):
    text: str
    model: str
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    tool_calls: list[ToolCallPart] = Field(default_factory=list)
