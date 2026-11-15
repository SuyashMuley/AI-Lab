# LLM CLI: Plan

A small, well-tested Python CLI for talking to LLMs. Scope: **model APIs, streaming, token usage, structured output.**
Single package. No agent frameworks. No separate `llm-core` package (extract later only if another project needs it).

## Goals
- `llm ask "prompt"` works against real providers.
- Streaming output with clean Ctrl+C handling.
- Token usage and latency reporting.
- Structured output validated against a JSON schema.
- Clean internal boundaries so the code is easy to extend or extract later.

## Non-goals (for now)
- Agents, tools, memory, RAG, MCP, workflows
- Multi-package workspace, middleware hooks, record/replay provider
- Conversation history (SQLite is optional, M9)

## Stack
| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| CLI | Typer |
| Terminal output | Rich |
| LLM access | Official provider SDKs |
| Models/config | Pydantic + Pydantic Settings |
| Async | asyncio (single `asyncio.run()` at the CLI boundary) |
| Tests | pytest + pytest-asyncio |
| Packaging | uv |
| Quality | Ruff + mypy (or Pyright) |

## Layout
```
LLM CLI/
├── pyproject.toml
├── README.md
├── PLAN.md
├── .env.example
├── src/llmcli/
│   ├── cli.py                # Typer commands only
│   ├── core/
│   │   ├── models.py         # Message, LLMRequest, LLMResponse, Usage
│   │   ├── events.py         # TextDelta, UsageEvent, Done, ErrorEvent
│   │   ├── errors.py         # LLMError hierarchy
│   │   └── protocols.py      # LLMProvider protocol
│   ├── providers/            # one adapter per provider + registry
│   ├── services/llm.py       # generate/stream, retries, timeouts, usage
│   ├── config/settings.py
│   └── ui/console.py
└── tests/{unit,integration,fakes}/
```

## Design decisions
1. **Own the types.** CLI and services never touch SDK types. Providers adapt SDK <-> internal models.
2. **Pydantic models**, not dataclasses (config, schema validation and future JSON serialization).
3. **Message content as parts** (`list[TextPart]` in V1, with a `str` shorthand) so tool parts can be added later without breaking the contract.
4. **Streaming = typed events** ending in a `Done(finish_reason, response)` or `ErrorEvent`. A helper folds a stream into an `LLMResponse`, so `generate()` and `stream()` share code.
5. **Errors normalized**: `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderUnavailableError`, `ModelNotFoundError`, `LLMTimeoutError`. Original exception kept as `__cause__`.
6. **Retries owned by us**: disable SDK retries, use bounded exponential backoff with jitter on 429/5xx/connection errors. Never retry invalid requests. Never retry after partial streamed output.
7. **Output hygiene**: response data on stdout, usage and diagnostics on stderr, so pipes work. Detect TTY.
8. **Config precedence**: CLI flags > env vars > project config > user config > defaults. Secrets via env vars only, never logged.
9. **Usage**: input, output and cached tokens, plus `latency_ms`. Cost is an optional estimate from a data-file price table, clearly labeled as an estimate.

## Milestones
| # | Deliverable | Acceptance criteria |
|---|---|---|
| M1 | Scaffold + minimal `llm ask` | One provider returns a response; project installs with `uv` |
| M2 | Provider interface + service layer | CLI calls the service only; `FakeProvider` powers tests |
| M3 | Second provider | Same command works with both; the second is chosen to differ in shape from the first |
| M4 | Streaming | `--stream` prints incrementally; Ctrl+C cancels cleanly; `Done` event ends the stream |
| M5 | Reliability | Timeouts, bounded retries, normalized errors, correct exit codes |
| M6 | Configuration | Precedence works; `llm config set/get`; model aliases; secrets never printed |
| M7 | Usage reporting | `--usage` shows tokens and latency on stderr; optional cost estimate |
| M8 | Structured output | `--schema file.json` uses native provider support where available, validates, and retries with the error on failure |
| M9 (optional) | Conversation sessions | SQLite persistence; `--continue` |
| M10 | Polish | README, smoke tests per real provider, installable via `uv tool install` |

**Usable version: M4. Solid foundation: M8.**

## Testing
- `FakeProvider` for deterministic unit tests (no network).
- Stream chunks render in order; `Done` always terminates.
- Bad credentials give `AuthenticationError`.
- 429 retries are bounded; invalid requests are not retried.
- No retry after partial stream output.
- Config precedence is respected.
- Ctrl+C does not report success.
- Secrets never appear in logs or tracebacks.
- Schema validation failure triggers a retry, then a clear error.
- One optional integration smoke test per real provider (skipped without keys).

## Open questions
- Which provider first, and do you have keys?
- Which second provider: Anthropic (different shape, better abstraction test) or Ollama (local, free)?
- Type checker: mypy or Pyright?
- Is session persistence (M9) wanted in this project?
