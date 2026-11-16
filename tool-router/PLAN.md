# Strict Pydantic Tool Router: Plan

A small, heavily-tested Python library (plus a demo CLI) that sits between an LLM and your code. It **defines tools with Pydantic schemas, exposes them to models, strictly validates the tool calls models produce, executes the handlers safely, and returns structured results or model-readable errors.**

The core promise: **whatever bytes a model emits, the router never crashes, never runs a handler with unvalidated arguments, and always returns a result the model can act on.**

Scope: tool definition, schema export, strict validation, routing, safe execution, error feedback, policy/audit hooks. No agent loop, no memory, no provider SDK in the core.

## Goals
- Define a tool from a typed Python function or an explicit Pydantic model.
- Export tool schemas in the formats Anthropic, OpenAI, and Ollama expect.
- Parse tool calls from raw model output (JSON string or dict) and validate them **strictly** (no unknown fields, no silent type coercion by default).
- Dispatch to sync or async handlers with timeouts, cancellation, output limits, and sanitized errors.
- Run parallel tool calls with bounded concurrency and stable result ordering.
- Turn every failure into a structured, model-readable error that makes self-repair likely.
- Policy and hook points for permissions and audit, without building a security product here.

## Non-goals (for now)
- The agent loop / ReAct / planning (that is **Local Agent Runtime**, project 3)
- Memory, RAG, context budgeting (projects 4-5)
- MCP server/gateway (Phase 2), sandboxed execution (Phase 3), full security gateway (Phase 4)
- Calling LLM providers from the core library (only the demo CLI does, via Ollama)
- Tool discovery over the network, plugin marketplaces, hot reload

## How it relates to `llm-cli`
Standalone package with its own `pyproject.toml`. **It does not import `llmcli`.** The core is provider-agnostic: it consumes and produces plain dicts/JSON in each provider's tool format. The llm-cli currently only has text message parts, so wiring tool-call parts through it belongs to project 3, where the loop needs both. That is also the moment to decide whether to extract a shared `llm-core` (the llm-cli plan deferred this on purpose).

## Stack
| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| Validation / schemas | Pydantic v2 (strict mode) |
| Async | plain asyncio (`TaskGroup`, `timeout`, `to_thread`); no anyio |
| Demo CLI | Typer + Rich |
| Demo model access | Official `ollama` SDK (demo only, not in core) |
| Tests | pytest + pytest-asyncio + **Hypothesis** (fuzzing model output) |
| Packaging | uv |
| Quality | Ruff + mypy (strict) |

## Architecture

### Pipeline (one tool call)
```
 raw model output                                                   to the model
 (provider format)                                                       ▲
        │                                                                │
        ▼                                                                │
 ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐
 │ formats/*   │─►│ lookup   │─►│ parse +   │─►│ policy │─►│ execute │─►│ result   │
 │ parse calls │  │ registry │  │ validate  │  │ + hooks│  │ timeout │  │ envelope │
 │ → ToolCall  │  │          │  │ (strict)  │  │        │  │ threads │  │ + format │
 └─────────────┘  └────┬─────┘  └─────┬─────┘  └───┬────┘  └────┬────┘  └────┬─────┘
                       │              │            │            │            │
                  UnknownTool   InvalidJSON /  PolicyDenied  ToolTimeout /   OutputTruncated /
                  (+did-you-     ArgumentsInvalid            ToolExecution   ResultInvalid
                   mean)         (+issues)                   Error
                       └──────────────┴────────────┴────────────┴───────────► ToolResult(ok=False, error=…)
```
Every stage failure short-circuits into a `ToolResult` with `ok=False`. `dispatch()` **never raises** for anything a model can cause.

### Layout
```
tool-router/
├── pyproject.toml
├── README.md
├── PLAN.md
├── src/toolrouter/
│   ├── tool.py          # Tool (frozen dataclass), SideEffect, @tool decorator
│   ├── signature.py     # function signature + docstring -> strict args model
│   ├── docstring.py     # Google-style docstring parser (summary + param descriptions)
│   ├── models.py        # StrictModel base + recursive "no unknown fields" schema check
│   ├── registry.py      # ToolRegistry: register/get/list, name rules, duplicates
│   ├── schema.py        # JSON-schema normalization (inline $refs, strict-mode fixes)
│   ├── argtypes.py      # strict argument types: Confined (path confined to a root)
│   ├── calls.py         # ToolCall, ToolResult, ErrorInfo, ValidationIssue (Pydantic)
│   ├── errors.py        # ErrorCode enum + internal exception types
│   ├── validate.py      # parse JSON, size/depth limits, strict/lax validation -> issues
│   ├── executor.py      # run handler: timeout, daemon-thread offload for sync code, abandonment
│   ├── results.py       # validate + serialize handler return values (exact str raw, else JSON)
│   ├── router.py        # ToolRouter.dispatch / dispatch_many
│   ├── policy.py        # Policy protocol, SideEffect levels, built-in policies
│   ├── hooks.py         # RouterHook protocol (before/after), logging hook with redaction
│   ├── feedback.py      # error -> model-readable text, did-you-mean
│   ├── formats/         # anthropic.py, openai.py, ollama.py (tools out, calls in, results out)
│   └── cli.py           # demo: list, schema, call, chat
├── examples/tools.py    # calculator, read_file (confined path), get_time
└── tests/{unit,property,integration,fakes}/
```

### Core types (sketch)
```python
class SideEffect(StrEnum):
    NONE = "none"  # pure
    READ = "read"  # reads external state
    WRITE = "write"  # mutates external state


@dataclass(frozen=True)
class Tool:  # what the model sees + how we run it (built by @tool)
    name: str  # ^[a-zA-Z][a-zA-Z0-9_-]{0,63}$ (works across providers)
    description: str  # required, non-empty
    args_model: type[BaseModel]  # extra="forbid" enforced at definition time, recursively
    handler: Callable[..., Any]
    side_effect: SideEffect = SideEffect.NONE
    timeout_s: float = 30.0
    max_output_chars: int = 20_000
    coerce: bool | None = None  # None = inherit the router default; True/False = override
    result_type: Any | None = None  # return annotation: Pydantic model, str, or JSON-able type
    takes_model: bool = False  # explicit-model tools get the model instance whole


class ToolCall(BaseModel):
    id: str  # generated if the provider has none (Ollama)
    name: str
    arguments: str | dict[str, Any]  # raw: string JSON or already-parsed


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    content: str | None  # serialized success value (truncated if needed)
    error: ErrorInfo | None
    truncated: bool = False
    duration_ms: float


class ErrorInfo(BaseModel):
    code: ErrorCode  # stable machine-readable enum
    message: str  # safe for the model to read
    issues: list[ValidationIssue] = []  # for ARGUMENTS_INVALID: loc, msg, type
    retryable: bool  # should the model try again with changes?
```

### Public API (target)
```python
@tool(side_effect=SideEffect.READ, timeout_s=5)
def get_weather(city: str, units: Literal["c", "f"] = "c") -> WeatherReport:
    """Current weather for a city."""


registry = ToolRegistry([get_weather, calculator])
router = ToolRouter(registry, policy=ReadOnlyPolicy(), hooks=[AuditLog()], coerce=False)

tools_payload = anthropic.tools(registry)  # or openai.tools / ollama.tools
calls = anthropic.parse_calls(response_content_blocks)  # -> list[ToolCall]
results = await router.dispatch_many(calls, max_concurrency=4)
messages = anthropic.result_messages(results)  # back into the conversation
```

## Design decisions
1. **Strict by default, coercion is a router-level setting with per-tool override.** Args models must forbid unknown fields (checked recursively, including nested models, when the tool is defined) and are validated in strict mode at call time, so `coerce` can toggle strictness without redefining models: an unknown field or `"42"` for an `int` is an error, not a guess. `ToolRouter(coerce=True)` switches on lenient coercion for all tools (useful with small local models); a tool can override with `coerce=True/False`. **Safety rule: tools with `side_effect=WRITE` never inherit a lenient router default; they must opt in explicitly.** That keeps a global switch from loosening validation on destructive tools.
2. **The router never raises on model-caused failures.** Bad JSON, unknown tool, invalid args, timeouts, handler exceptions all become `ToolResult(ok=False)`. Exceptions are reserved for programmer errors (duplicate tool name, invalid spec) and surface at registration time.
3. **Errors are for two audiences.** `ErrorInfo.code` is a stable enum for code; `message`/`issues` are written for the model. Internals (tracebacks, file paths, secrets) never reach the model; they go to hooks/logs only (`ToolResult.cause` carries the original exception, excluded from serialization). Handlers signal *expected* failures with `ToolError("message for the model")`, whose message is passed through; any other exception becomes a generic `TOOL_FAILED` naming only the exception type.
4. **Validation happens before everything else with side effects.** A handler only ever sees an instance of its validated args model.
5. **Own the types; keep providers at the edge.** The core speaks `Tool` / `ToolCall` / `ToolResult`. `formats/*` are pure functions translating to and from each provider's JSON shape. No provider SDK is imported by the core.
6. **One normalized schema pipeline.** Pydantic's JSON schema is post-processed once (inline `$ref`/`$defs`, drop `title` noise, force `additionalProperties: false`, normalize optionals). Provider adapters then apply their quirks (OpenAI Chat vs Responses envelopes, Anthropic `input_schema`, Ollama). Recursive models are rejected at registration because they can't be inlined. **OpenAI `strict` mode is deferred:** it needs every property required and optionals nullable, so it can't ship until validation (T3) has a story for the nulls models then send for omitted optionals. Schemas are snapshot-tested (`UPDATE_SNAPSHOTS=1` to regenerate; a missing snapshot fails).
7. **Execution is bounded.** Per-tool timeout (covering the whole call, and never confused with a handler's own `TimeoutError`), output cap with a visible truncation marker (the marker counts inside the cap). Sync handlers run in **per-call daemon threads with the caller's contextvars**, not a pool: a stuck handler can't wedge a shared pool, can't block interpreter exit, and a call that already timed out can never start later. Concurrency is bounded by `max_threads` (default 32); when saturated, new sync calls **fail fast** (retryable `TOOL_FAILED`) instead of queueing silently. A thread cannot be killed: on timeout or cancellation the caller stops waiting, the thread is tracked as *abandoned* until it ends, and its late result is discarded (real isolation arrives in the Sandboxed Execution project).
8. **Parallel calls are independent, except that writes are ordered.** `dispatch_many` uses `max_concurrency` worker tasks pulling from the batch (a 10,000-call batch costs a handful of tasks, not 10,000), preserves call order in the results, and one failure never affects siblings. Calls to WRITE tools in the same batch run one at a time in the order the model listed them (`serialize_writes`, on by default, switchable per router or per batch), while READ/NONE calls stay fully parallel; a waiting write holds its concurrency slot. Cancelling the caller cancels every running handler and starts nothing new (a `TaskGroup` guarantees no orphan tasks).
9. **Policy is a seam, not a product.** A `Policy` can allow/deny a call based on tool, side effect, and validated args. Built-ins: `AllowAll`, `ReadOnlyPolicy`, `AllowList`. Real policy enforcement is the later Security Gateway project.
10. **Hooks for observability.** `before`/`after` hooks receive the call and result. The built-in logging hook redacts values of fields marked secret and truncates large payloads. The later Observability project builds on this.
11. **Untrusted input limits.** Max argument payload size (default 64 KB) and max JSON nesting depth are checked before validation, so a hostile or runaway model cannot cost unbounded CPU or memory. Non-standard input is rejected outright: duplicate keys, `NaN`/`Infinity`, non-object top levels, lone surrogates. An empty argument string means "no arguments" (models often send `""` for zero-arg tools); required fields are still required. Validation always runs in Pydantic's **JSON mode** (dict input is serialized first, so both input forms get identical checks): strict JSON mode accepts strings for enums, datetimes, UUIDs and paths but rejects `"42"` for an int and `1` for a bool, whereas Python-mode strict would wrongly reject enums.
12. **Deterministic IDs and ordering.** Missing call IDs are generated (`call_<n>`); results always reference the originating `call_id`.
13. **Tools may return Pydantic models.** The return annotation becomes the result type: models are serialized with `model_dump_json`, `str` passes through, other JSON-able types go through a `TypeAdapter`. The returned value is validated against the annotation, so a handler bug surfaces as `RESULT_INVALID` instead of garbage reaching the model. Results are output-only and never coerced. An exact `str` is returned raw, everything else as JSON (`None` becomes `null`); a dict that strictly satisfies a declared model is accepted. Unannotated or `Any` returns are serialized, not validated.
14. **`ConfinedPath` ships in the library** (`toolrouter.argtypes`, as the `Confined` annotation marker). A path type that resolves symlinks and `..` and rejects anything outside a configured root, so file tools get safe paths from validation rather than ad-hoc checks. It is a validation-time guard, not a sandbox (that is Phase 3).
15. **Plain asyncio, no anyio.** Python 3.12 has `asyncio.TaskGroup`, `asyncio.timeout`, and `asyncio.to_thread`, which cover structured concurrency, timeouts, and thread offload. It avoids a dependency and matches llm-cli.

## Error taxonomy
| Code | When | Retryable by model | Notes |
|---|---|---|---|
| `UNKNOWN_TOOL` | name not registered | yes | message includes did-you-mean suggestions |
| `INVALID_JSON` | arguments string isn't JSON / not an object | yes | message points at the parse error position |
| `ARGUMENTS_TOO_LARGE` | payload or depth over limit | yes | |
| `ARGUMENTS_INVALID` | schema violation | yes | `issues[]` with loc/msg/type; offending values truncated |
| `POLICY_DENIED` | policy said no | no | reason stated; no internals *(T8)* |
| `TOOL_TIMEOUT` | exceeded `timeout_s` | maybe | |
| `TOOL_FAILED` | handler raised | maybe | only the sanitized message; full traceback to hooks |
| `INTERNAL_ERROR` | a bug on our side, e.g. a custom validator that crashes | no | generic message; never blames the model or leaks internals |
| `RESULT_INVALID` | return value can't be serialized / fails result model | no | programmer error, logged loudly |
| `CANCELLED` | a `CancelledError` was raised *inside* a handler while the caller was not being cancelled | no | if the caller itself is cancelled, `CancelledError` propagates (asyncio requires it) |

Output truncation is not an error: `ok=True`, `truncated=True`, with a marker like `[truncated 18,204 chars]`.

## Milestones
| # | Deliverable | Acceptance criteria |
|---|---|---|
| T1 | Scaffold + tool definition | `@tool` builds a strict args model from a function signature and docstring; explicit-model tools also work; invalid names/empty descriptions rejected at definition time; `uv sync` works |
| T2 | Registry + schema export | Registry rejects duplicates; normalized JSON schemas match snapshots; Anthropic, OpenAI, and Ollama tool payloads produced |
| T3 | Call parsing + strict validation | **Done.** Provider call parsing (`parse_calls` for Anthropic, OpenAI chat/responses, Ollama, tolerant of SDK objects or dicts); JSON string or dict in; unknown/extra fields, wrong types, and missing fields all produce `ARGUMENTS_INVALID` with usable `issues`; size/depth limits enforced; router-level and per-tool `coerce` with the WRITE-tool safety rule; `ConfinedPath` rejects traversal and symlink escapes; behavior table of strict vs coerce documented by tests |
| T4 | Router dispatch + result envelope | **Done.** `dispatch()` returns a `ToolResult` for every model-caused failure except timeout/cancellation (T5) and policy denial (T8): unknown tool, invalid JSON, oversized, invalid arguments, tool failure, invalid result, internal error. It never raises for them (only `Exception` is caught; `CancelledError`/`KeyboardInterrupt`/`SystemExit` propagate); handler only runs after validation; Pydantic/str/JSON return values serialized and validated against the return annotation |
| T5 | Execution safety | **Done.** Per-tool timeouts; sync handlers off the event loop; cancellation works; output truncation; handler exceptions sanitized (no traceback/paths to the model) |
| T6 | Parallel dispatch | **Done.** `dispatch_many` honors `max_concurrency`, preserves order, isolates failures, cancels cleanly |
| T7 | Model-facing feedback | Errors render as concise repair prompts; did-you-mean for tool names; provider-specific result messages (Anthropic `tool_result`, OpenAI `tool` role, Ollama `tool`) |
| T8 | Policy + hooks | Side-effect levels enforced by built-in policies; `before`/`after` hooks fire for every outcome incl. failures; log redaction of secret fields |
| T9 | Demo CLI + real model | `toolrouter chat` runs a real Ollama tool-calling model through the router with a small repair loop (max N attempts); `list`, `schema`, `call` subcommands; smoke tests skip without Ollama |
| T10 | Polish | README, property/fuzz test suite green, examples, installable with `uv tool install`, mypy strict clean |

**Usable library: T4. Production-grade core: T8. Proven against a real model: T9.**

## Testing
- `FakeTool` factories and scripted "model outputs" for deterministic unit tests (no network).
- **Strictness table**: for each arg type (int, float, bool, str, list, nested model, enum, Literal, optional), assert exactly which JSON inputs are accepted/rejected in strict mode and in `coerce=True`.
- **Property-based (Hypothesis)**: for arbitrary JSON values (and arbitrary bytes-as-strings) as arguments to arbitrary registered tools, `dispatch()` never raises, always returns a `ToolResult`, and never invokes a handler unless validation passed.
- Schema snapshot tests per provider; schemas round-trip: every generated example validates.
- Handler never sees invalid args (spy handler asserts call count 0 on failures).
- Timeouts fire on async and sync handlers; the event loop stays responsive during a blocking sync handler.
- Exceptions inside handlers never leak tracebacks, absolute paths, or secret values into `ToolResult`.
- `dispatch_many`: order preserved, concurrency bounded (observed max in-flight), one failure doesn't affect siblings, cancel mid-flight leaves no orphan tasks.
- Oversized payloads and deeply nested JSON rejected quickly (time-bounded test).
- Policy denies before execution; hooks see failures too; redaction applied to logs.
- Did-you-mean suggestions are deterministic.
- Optional smoke test with real Ollama tool calling (skipped when unavailable).

## Risks
- **Provider schema quirks drift.** Isolated in `formats/` with snapshot tests so a provider change is a one-file fix.
- **Strict mode frustrates weaker models** (e.g. numbers as strings). That is what the feedback format and the per-tool `coerce` opt-in are for; T9 measures how often repair is needed on a small local model.
- **Can't kill a stuck sync thread.** Mitigated, not solved: results return on time, stuck threads are counted (`router.executor.abandoned_threads`), daemon threads never block exit, and a saturated executor fails fast. A handler that swallows `CancelledError` can also outlive its async timeout. Real isolation needs the sandboxed-execution project.
- **Scope creep into the agent loop.** The only loop in this project is the minimal repair loop in the demo CLI; the real one is project 3.

## Resolved questions
| Question | Decision |
|---|---|
| `coerce` scope | Router-level default **plus** per-tool override; WRITE tools never inherit a lenient default |
| Strict path type | Ships in the library as `ConfinedPath` |
| Result models | Pydantic models allowed (and `str` / JSON-able values) |
| anyio vs asyncio | Plain asyncio |
| Ollama model for T9 | `llama3.2` (3B); pull it before T9. Fallback to a larger tool-capable model (e.g. a Qwen 2.5/3 7B-class) if the repair rate is too high |
