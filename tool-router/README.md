# toolrouter

A small, strict Python library that sits between an LLM and your code.

Models emit tool calls as **untrusted text**. toolrouter lets you define tools with Pydantic, strictly validates every call, runs the handlers safely, and hands the model a result or an error it can act on.

> **The promise:** whatever a model sends, the router never crashes, never runs a handler with unvalidated arguments, and always returns a result the model can read.

- **Strict by default.** Unknown fields and wrong types are errors, not guesses (`"42"` is not an `int`).
- **Safe execution.** Timeouts, sync handlers off the event loop, bounded output, sanitized errors, cancellation that works.
- **Errors that repair themselves.** Messages say what was wrong *and* what a valid call looks like, with did-you-mean for tool names.
- **Parallel calls** with a concurrency bound, stable ordering, and write calls kept in order.
- **Policy and hooks** for permissions and audit; secrets are redacted everywhere.
- **Provider-agnostic.** Tool payloads, call parsing and result messages for Anthropic, OpenAI and Ollama. The core needs only `pydantic`.

## Install

Requires Python 3.12+.

```bash
pip install toolrouter              # the library (pydantic only)
pip install "toolrouter[cli]"       # plus the demo CLI (typer, rich, ollama)
```

From a checkout (this repo is not on PyPI): `uv tool install ".[cli]"`, or `uv sync` for development.

## Quick start

```python
# readme-test
import asyncio
from typing import Literal

from toolrouter import SideEffect, ToolCall, ToolError, ToolRegistry, ToolRouter, tool


@tool(side_effect=SideEffect.READ)
def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Args:
        city: City name.
        units: "c" for Celsius, "f" for Fahrenheit.
    """
    if city.lower() != "oslo":
        raise ToolError(f"No data for {city!r}.")
    return "4 degrees, light snow" if units == "c" else "39 degrees, light snow"


registry = ToolRegistry([get_weather])
router = ToolRouter(registry)


async def main() -> None:
    good = ToolCall(id="1", name="get_weather", arguments={"city": "Oslo"})
    result = await router.dispatch(good)
    print(result.ok, result.content)

    bad = ToolCall(id="2", name="get_weather", arguments='{"city": "Oslo", "units": "kelvin"}')
    result = await router.dispatch(bad)
    print(result.ok, result.error.code.value)
    print(result.text)


asyncio.run(main())
```

```text
True 4 degrees, light snow
False arguments_invalid
Invalid arguments for 'get_weather': units: Input should be 'c' or 'f' (got 'kelvin'). Expected: get_weather(city: string, units?: "c" | "f" = "c")
```

The second call never reached the handler. The error text is what you send back to the model; it contains everything needed to fix the call.

## Defining tools

A tool is a function with **fully annotated parameters**. The signature becomes a strict argument schema, the docstring summary becomes the description, and the `Args:` section becomes per-parameter descriptions.

```python
@tool  # bare...
def add(a: int, b: int = 0) -> int:
    """Add two numbers."""


@tool(
    side_effect=SideEffect.WRITE,  # ...or with options
    timeout_s=10,
    max_output_chars=5_000,
    coerce=False,
    name="save",
    description="...",
)
async def save(path: str, content: str) -> str: ...
```

- **Async or sync.** Sync handlers run in worker threads, never on the event loop.
- **Constraints** come from `Annotated[int, Field(ge=1, le=50)]` and friends.
- **Explicit models.** `@tool(model=MyArgs, name=..., description=...)` hands the handler the validated model whole. Models must forbid unknown fields (subclass `StrictModel`); nested models are checked too.
- **Return values** may be `str` (sent as-is), Pydantic models, lists, dicts, numbers, `None` (sent as JSON). The value is validated against the return annotation, so a bug in a handler becomes a `result_invalid` error, not garbage in the conversation.
- **Expected failures:** `raise ToolError("message for the model")`. Its message is passed through; any other exception becomes a generic `tool_failed` naming only the exception type.
- **Definition-time errors** (`ToolDefinitionError`): bad names, missing description or annotations, `*args`/`**kwargs`, reserved parameter names, recursive models, schemas that can't be expressed.

See [`examples/tools.py`](examples/tools.py) for async handlers, confined paths, secrets, explicit models and `ToolError`, and [`examples/quickstart.py`](examples/quickstart.py) for an offline walkthrough.

## Strict by default

Arguments are validated in Pydantic's JSON mode with unknown fields forbidden:

| JSON the model sent | Field type | strict (default) | `coerce=True` |
|---|---|---|---|
| `42` | `int` | ok | ok |
| `"42"` | `int` | rejected | `42` |
| `1.0` | `int` | rejected | `1` |
| `1` or `"true"` | `bool` | rejected | `True` |
| `"red"` | `Enum` / `datetime` / `UUID` / `Path` | ok | ok |
| `{"a": 1, "extra": 2}` | model with field `a` | rejected | rejected |
| (missing field) | required | rejected | rejected |

Turn on leniency per router (`ToolRouter(registry, coerce=True)`) or per tool (`@tool(coerce=True)`; a per-tool setting always wins). **`WRITE` tools never inherit a lenient router default**; they must opt in themselves.

Input limits apply before parsing: 64 KiB of arguments and nesting depth 32 (configurable via `Limits`). Duplicate JSON keys, `NaN`/`Infinity`, non-object top levels and lone surrogates are rejected; an empty argument string means "no arguments".

## Errors

`dispatch()` never raises for anything a model can cause. Every failure is a `ToolResult` with `ok=False` and an `ErrorInfo`:

| `code` | When | Model may retry |
|---|---|---|
| `unknown_tool` | name not registered (with did-you-mean) | yes |
| `invalid_json` | arguments aren't a JSON object | yes |
| `arguments_too_large` | over the size or depth limit | yes |
| `arguments_invalid` | schema violation (`issues` lists each one) | yes |
| `policy_denied` | a policy said no | no |
| `tool_timeout` | exceeded the tool's `timeout_s` | yes |
| `tool_failed` | the handler raised | yes |
| `result_invalid` | the return value didn't match its annotation | no |
| `cancelled` | a `CancelledError` came from *inside* a handler | no |
| `internal_error` | a bug on our side (a crashing validator or policy) | no |

Only `Exception` is converted. If the **calling task** is cancelled, `CancelledError` propagates (asyncio requires it); `KeyboardInterrupt`/`SystemExit` on the event loop propagate too.

What a model finally reads comes from `toolrouter.feedback.model_text(result)`: the content on success (`(no output)` if empty) and `Error (<code>): <message>` on failure.

## Safe execution

- **Timeouts** are per tool (`timeout_s`, default 30) and cover the whole call. A handler's own `TimeoutError` is not mistaken for ours.
- **Sync handlers** run in per-call daemon threads with the caller's `contextvars`, bounded by `max_threads` (default 32). When saturated, new sync calls fail fast instead of queueing.
- **A thread cannot be killed.** On timeout or cancellation the caller stops waiting, the result returns on time, and the thread is counted in `router.executor.abandoned_threads` until it ends; its late result is discarded. Real isolation for runaway code needs a sandbox, which this library is not.
- **Output** is capped at `max_output_chars` (default 20,000), marker included; truncated results are still `ok` with `truncated=True`.
- **Handler exceptions** never leak: no tracebacks, paths or secrets reach the model. The original exception is on `result.cause` for your logs.

## Parallel calls

```python
results = await router.dispatch_many(calls, max_concurrency=4)  # same order as `calls`
```

At most `max_concurrency` calls run at once, one failure never affects another, and a 10,000-call batch costs a handful of tasks. Calls to `WRITE` tools in one batch run one at a time **in the order the model listed them** (`serialize_writes=False` to opt out); reads stay parallel. Cancelling the caller cancels every running handler and starts nothing new.

## Policy, hooks and secrets

```python
# readme-test
import asyncio
from typing import Annotated

from toolrouter import (
    ReadOnlyPolicy,
    RecordingHook,
    Secret,
    SideEffect,
    ToolCall,
    ToolRegistry,
    ToolRouter,
    tool,
)


@tool(side_effect=SideEffect.WRITE)
def send_message(to: str, text: str, api_key: Annotated[str, Secret()]) -> str:
    """Send a message."""
    return f"sent to {to}"


recorder = RecordingHook()
safe_router = ToolRouter(
    ToolRegistry([send_message]),
    policy=ReadOnlyPolicy(),  # reads allowed, WRITE tools denied
    hooks=[recorder],  # one before + one after per call, whatever the outcome
)


async def main() -> None:
    call = ToolCall(
        id="1",
        name="send_message",
        arguments={"to": "ann", "text": "hi", "api_key": "sk-live-123"},
    )
    result = await safe_router.dispatch(call)
    print(result.error.code.value)
    print(recorder.before_events[0].arguments)  # what a log would show


asyncio.run(main())
```

```text
policy_denied
{'to': 'ann', 'text': 'hi', 'api_key': '[REDACTED]'}
```

- **Policies** (`ReadOnlyPolicy`, `MaxSideEffect`, `AllowList`, `DenyList`, `AllOf`, or your own with a `check(request)` method, sync or async) see the validated arguments before the handler runs. They **fail closed**: a policy that raises, or returns anything but a `PolicyDecision`, blocks the call.
- **Hooks** (`RouterHook`, `RecordingHook`, `LoggingHook`) fire exactly once before and once after every dispatch, including unknown tools, invalid arguments, denials, timeouts and caller cancellation. A hook that raises or exceeds `hook_timeout_s` is logged and skipped; it can never change a result.
- **Secrets:** mark a field with `Annotated[str, Secret()]` or use `SecretStr`. Secrets are redacted from hook events, logs and validation errors, and a secret's default is never sent to a provider. Undeclared fields and unparseable arguments are never echoed. See *Limitations* for what redaction does not cover.
- **`Confined`** keeps a path argument inside a directory (no `..`, no absolute escapes, no symlink escapes): `Annotated[Path, Confined("/srv/data")]`. It is a validation-time guard, not a sandbox.

## Using it with a provider

Every provider module (`toolrouter.formats.anthropic`, `.openai`, `.ollama`) has the same three functions: `tools(registry)`, `parse_calls(response)`, `result_messages(results)`. Calls parse from SDK objects or plain dicts, and never raise on odd shapes.

```python
# readme-test
import asyncio

from toolrouter.formats import ollama

# `message` is what an Ollama chat response contains (a model produced this):
message = {
    "role": "assistant",
    "tool_calls": [
        {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}},
        {"function": {"name": "get_weather", "arguments": {"city": "Atlantis"}}},
    ],
}


async def main() -> None:
    payload = ollama.tools(registry)  # pass as `tools=` when you call the model
    calls = ollama.parse_calls(message)  # -> list[ToolCall] (ids generated: call_0, ...)
    results = await router.dispatch_many(calls)
    for m in ollama.result_messages(results):  # append these to the conversation
        print(m["role"], m["tool_name"], "|", m["content"])
    print(payload[0]["function"]["name"])


asyncio.run(main())
```

```text
tool get_weather | 4 degrees, light snow
tool get_weather | Error (tool_failed): No data for 'Atlantis'.
get_weather
```

Anthropic and OpenAI follow the same shape:

```python
# Anthropic
tools = anthropic.tools(registry)
response = client.messages.create(model=..., max_tokens=1024, tools=tools, messages=messages)
calls = anthropic.parse_calls(response.content)
results = await router.dispatch_many(calls)
messages += [
    {"role": "assistant", "content": response.content},
    *anthropic.result_messages(results),
]  # ONE user message with every tool_result

# OpenAI (Chat Completions; use api="responses" for the Responses API)
tools = openai.tools(registry)
message = (
    client.chat.completions.create(model=..., tools=tools, messages=messages).choices[0].message
)
calls = openai.parse_calls(message)
results = await router.dispatch_many(calls)
messages += [message, *openai.result_messages(results)]
```

> **What is verified:** the Ollama path is exercised against a real local model (`llama3.2`) in this repo's smoke tests. The Anthropic and OpenAI payload shapes follow their documentation and are covered by unit and snapshot tests, but have not been run against the live APIs. OpenAI `strict` structured-output mode is intentionally not emitted yet.

## The `toolrouter` CLI

```bash
toolrouter list                              # tools, side effects, signatures
toolrouter schema calculator --format anthropic     # ollama | anthropic | openai | openai-responses | json
toolrouter call calculator --args '{"expression": "17 * 23"}'   # one call, handled exactly like a model's
toolrouter chat "What time is it in Oslo?"   # a local model uses the tools (needs Ollama)
toolrouter list --tools examples/tools.py    # your own tools (this imports and runs that file)
```

`chat` is **read-only unless `--allow-writes`**, bounds every model reply (`--model-timeout`, `--max-tokens`), and gives a model a small repair budget (`--max-repairs`, default 2) that counts only failures it can fix by changing its call. The answer goes to stdout; a redacted trace and a summary go to stderr.

Exit codes: `0` ok, `1` a `call` failed, `2` usage error, `3` model unreachable / missing / can't use tools / timed out, `4` the model gave up, `130` interrupted.

## Guarantees and limitations

What the test suite pins down (including fuzzing arbitrary model output through the whole pipeline):

- A handler only ever runs with validated arguments, and only after the policy allowed it.
- `dispatch` returns a `ToolResult` for every model-caused failure; hooks see every outcome.
- Planted secrets never appear in events, logs, error messages, result messages or provider payloads.

What it does **not** do:

- **No sandbox.** A stuck sync handler can't be killed (it is abandoned and counted); `Confined` is checked at validation time, so a symlink swapped in afterwards isn't caught. Run untrusted code in a real sandbox.
- **Secret redaction is schema-driven.** A secret the model puts in a *declared, non-secret* field (say, an API key pasted into a `query: str`) cannot be recognised. Inside union branches, undeclared fields are not redacted.
- **Policies run after validation**, so a denied tool's signature can still appear in validation errors, and unknown-tool messages list every registered tool. Register only what a model may use.
- **Hooks that block the event loop with sync code can't be timed out.**
- **Not an agent framework.** The loop in `toolrouter.demo.chat` is a deliberately tiny demonstration of a bounded repair budget.
- Verified live only against Ollama + `llama3.2` (see above). On that model, 36 tool calls per mode across 10 tasks needed 1 repair in strict mode and none with `coerce`: a small sample, but it suggests strict-by-default is cheap.

## Development

```bash
uv sync --all-groups
uv run pytest -q                       # everything (real-model smoke tests skip without Ollama)
uv run pytest -m "not integration"     # offline only
uv run pytest -m integration           # real-model smoke tests (needs Ollama + llama3.2)
FUZZ_SCALE=20 uv run pytest tests/property     # 20x the fuzzing
uv run ruff check . && uv run ruff format --check . && uv run mypy      # mypy is strict
```

Snapshot tests (provider payloads, repair messages) live in `tests/snapshots/`: a missing or different snapshot fails; regenerate deliberately with `UPDATE_SNAPSHOTS=1` and review the diff. Every runnable Python block in this README is executed by `tests/unit/test_readme.py`.

Design notes and the milestone history are in [`PLAN.md`](PLAN.md).
