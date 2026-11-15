"""Provider-agnostic request/response types. SDK types never leak past providers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: list[TextPart]

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> Any:
        # `str` shorthand: Message(role="user", content="hi")
        return [{"text": v}] if isinstance(v, str) else v

    @classmethod
    def of(cls, role: Literal["user", "assistant", "system"], text: str) -> Message:
        return cls(role=role, content=[TextPart(text=text)])

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.content)


class LLMRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, Any] | None = None


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
