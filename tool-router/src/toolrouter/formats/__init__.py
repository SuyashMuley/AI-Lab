"""Pure functions translating tools to each provider's JSON shape. No provider SDKs here."""

from toolrouter.formats import anthropic, ollama, openai

__all__ = ["anthropic", "ollama", "openai"]
