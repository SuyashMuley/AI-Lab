"""Normalized error hierarchy. Providers raise these with the SDK error as __cause__."""


class LLMError(Exception):
    """Base class for all errors surfaced by llmcli."""

    exit_code = 1


class AuthenticationError(LLMError):
    exit_code = 3


class RateLimitError(LLMError):
    exit_code = 4


class InvalidRequestError(LLMError):
    exit_code = 2


class ProviderUnavailableError(LLMError):
    exit_code = 5


class ModelNotFoundError(LLMError):
    exit_code = 6


class LLMTimeoutError(LLMError):
    exit_code = 7


class SchemaValidationError(LLMError):
    """Model output never matched the requested JSON schema."""

    exit_code = 8

    def __init__(self, message: str, last_output: str = "") -> None:
        super().__init__(message)
        self.last_output = last_output
