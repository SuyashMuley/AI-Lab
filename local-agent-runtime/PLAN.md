# Local Agent Runtime + Self-Healing ReAct: Plan

A small, heavily-tested Python runtime that lets a **local model pursue a goal with tools**: it reasons, acts, observes, and repeats until it can answer. The point of the project is not the loop (a loop is 30 lines); it is everything that keeps the loop **bounded, observable, and able to recover when things go wrong**: a model that rambles, repeats itself, calls tools wrongly, times out, overflows its context, or a process that crashes mid-run.

> **The promise:** a run always terminates inside its budgets with an honest result (`completed`, `partial`, `failed` or `cancelled`), every recovery it performed is visible in an event log, and a crashed run can be resumed without repeating a side effect.

Scope: the agent loop, self-healing, budgets, final-output validation, a context-management *seam*, persistence and resume, a CLI. No memory, no RAG, no multi-agent, no human approval UI, no sandbox (those are later projects).

## Goals
- A ReAct-style loop (think, act, observe) over **native tool calling**, executing tools through `toolrouter`.
- **Self-healing** that is bounded, visible, and honest: a fault taxonomy with detectors and recoveries, never an infinite retry.
- Hard **budgets**: steps, tool calls, tokens, wall time; graceful wrap-up (a best-effort `partial` answer) instead of a stack trace.
- Validated **final output** (optional JSON schema / Pydantic model) with bounded repair.
- An **event-sourced run log**: replayable, persistable, resumable, and the future input of the Observability project.
- A **context-management seam** with a sane default, so the Context Budget Assembler (project 4) and Memory engine (project 5) plug in without a rewrite.
- A public `localagent.testing` module (scripted and *chaos* models) that later projects (evals, orchestrator) reuse.

## Non-goals (for now)
- Memory, RAG, retrieval, context *budgeting* algorithms (projects 4-5), beyond the seam and a default.
- Multi-agent coordination, planning/decomposition frameworks, an async orchestrator (Phase 4).
- Sandboxed execution, human-in-the-loop approvals, a security gateway (Phase 3-4); only the policy seam `toolrouter` already has.
- Text-format ReAct ("Thought:/Action:" parsing) for models without tool calling (see open questions).
- Provider breadth: Ollama only, like the earlier projects; the interface keeps others possible.

## How it fits the earlier projects

```
        llm-cli  (project 1)                    toolrouter  (project 2)
  providers, errors, retries, timeouts,    define / validate / route / execute tools,
  usage, structured output                 policy, hooks, redaction, repair messages
            \                                      /
             \                                    /
              +-------- local-agent-runtime ------+
                 loop, self-healing, budgets, context seam,
                 event log, persistence, resume, CLI
```

- `toolrouter` is used **as is** (it was built provider-agnostic for exactly this): the registry, `ToolRouter.dispatch_many`, `ToolRouter` policy and hooks, `feedback.model_text`, and the Ollama format helpers.
- `llm-cli` needs a small **additive extension** first (milestone R0 below): tool-call parts, a tool role, a tool-call stream event, `tools` on the request, and Ollama tool mapping. Everything else the runtime needs from a model layer (normalized errors, bounded retries, idle timeouts, usage, native structured output) is already there and tested.

## Stack
| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| Async | asyncio (`TaskGroup`, `timeout`), matching the earlier projects |
| Models / config | Pydantic v2 |
| Persistence | SQLite via the stdlib (`sqlite3`), no ORM |
| CLI | Typer + Rich |
| Tool execution | `toolrouter` (path dependency) |
| Model access | `llmcli` (path dependency) over the official `ollama` SDK |
| Tests | pytest + pytest-asyncio + Hypothesis |
| Packaging | uv |
| Quality | Ruff + mypy (strict) |

## Architecture

### Functional core, imperative shell
The loop is split so the interesting logic is **pure and cheap to test**, and the effects are thin:

```
                 ┌──────────────────────── pure ────────────────────────┐
  RunEvent ────► │  reduce(state, event) -> state                        │
  (the log)      │  decide(state)        -> Action                       │
                 │     CallModel | DispatchTools | Nudge | Finalize |    │
                 │     WrapUp | Fail                                     │
                 └───────────────▲──────────────────────────┬────────────┘
                                 │ new events               │ Action
                 ┌───────────────┴──────────── effects ─────▼────────────┐
                 │ ModelClient.respond()   ToolRouter.dispatch_many()    │
                 │ EventStore.append()     Clock / cancellation          │
                 └───────────────────────────────────────────────────────┘
```

- **State is a fold over events.** `RunState` (messages, counters, budgets used, failure/recovery counters, pending calls) is never mutated by effects; every effect *appends an event* and the reducer derives the next state. Replaying the log reproduces the state exactly, which is what makes persistence, resume, `replay`, and property tests easy.
- **`decide` is a pure function of state**: given the state it names the next action, including every self-healing move. All detectors (repetition, no-progress, circuit breakers, budgets) live here, so each recovery is a unit test with no model and no clock.
- The shell (`AgentRuntime`) executes one action, appends the resulting events (to memory and, optionally, SQLite), and loops. It owns time, cancellation, and I/O.

### Layout
```
local-agent-runtime/
├── pyproject.toml
├── README.md
├── PLAN.md
├── src/localagent/
│   ├── agent.py        # Agent (config), AgentRuntime: run(), stream(), resume()
│   ├── budgets.py      # Budgets (grows an axis per milestone that enforces it)
│   ├── tooling.py      # the bridge to toolrouter: tool specs, system prompt, outcomes
│   ├── events.py       # RunEvent union (the event log schema)
│   ├── state.py        # RunState, Budgets, reduce()
│   ├── decide.py       # decide(state) -> Action; escalation ladder
│   ├── recovery.py     # fault taxonomy, detectors (empty/repeat/no-progress), circuit breaker
│   ├── models.py       # ModelClient protocol, ModelTurn, llmcli adapter, FallbackModel wrapper
│   ├── context.py      # ContextStrategy protocol + WindowStrategy (default)
│   ├── output.py       # final-answer validation, wrap-up turn
│   ├── store.py        # SQLite event store: append, load, list, resume
│   ├── result.py       # RunResult, RunStatus, Recovery record
│   ├── cli.py          # `localagent run | resume | runs | replay | tools`
│   └── testing.py      # ScriptedModel, FakeClock, turn()/call() helpers (public); ChaosModel, FlakyTool arrive with R4-R5
└── tests/{unit,property,chaos,integration}/
```

### Core types (sketch)
```python
class Budgets(BaseModel):  # every run is bounded on every axis
    max_steps: int = 12  # model calls
    max_tool_calls: int = 30
    max_tokens: int | None = 50_000  # input + output, summed from usage
    max_seconds: float | None = 300  # wall clock for the whole run
    model_timeout_s: float = 120  # per model reply
    max_recoveries: int = 8  # total self-healing moves per run (no thrash)


class Agent(BaseModel):
    model: str
    system: str = DEFAULT_SYSTEM
    tools: ToolRegistry
    policy: Policy = ReadOnlyPolicy()  # safe by default; writes are opt-in
    budgets: Budgets = Budgets()
    output: type[BaseModel] | dict | None = None  # validated final answer, optional
    context: ContextStrategy = WindowStrategy()


class ModelClient(Protocol):  # what the loop needs from "a model"
    async def respond(self, request: TurnRequest) -> ModelTurn: ...

    # TurnRequest: messages, tools payload, options (num_ctx, temperature, max_tokens)
    # ModelTurn: text, tool_calls, usage, finish_reason   (streaming variant yields deltas)


class RunResult(BaseModel):
    status: Literal["completed", "partial", "failed", "cancelled"]
    answer: str | None
    output: Any | None  # the parsed/validated final output, if requested
    steps: int
    tool_calls: int
    usage: Usage
    elapsed_s: float
    recoveries: list[Recovery]  # what self-healing did, in order
    error: str | None  # why it failed (model-safe, no internals)
    run_id: str  # resumable / replayable
```

### Event log (the single source of truth)
`RunStarted` · `ModelRequested` · `ModelResponded(text, tool_calls, usage)` · `ToolsDispatching(call ids)` · `ToolResulted(result)` · `RecoveryApplied(kind, reason, detail)` · `ContextTrimmed(dropped, kept)` · `BudgetWarning` · `FinalAnswer` · `RunFinished(status)`. Each carries a monotonically increasing `seq` and a timestamp. Persisted events store **redacted** tool arguments (secret fields masked using `toolrouter`'s schema-driven redaction) and bounded payloads.

## Self-healing: the fault taxonomy

Every recovery is **bounded** (its own counter plus the global `max_recoveries`), **recorded** (`RecoveryApplied`), and **honest** (it shows up in `RunResult.recoveries`; a run that needed rescuing is not reported as if it hadn't).

| Fault | Detector | Recovery (bounded) |
|---|---|---|
| Model unreachable / 5xx / timeout | normalized `llmcli` error class | retry with backoff (3 tries), then `failed` (resumable) |
| Empty turn (no text, no tool calls) | turn inspection | nudge ("continue, or give your final answer"), max 2 |
| Malformed / unknown / invalid tool call | `toolrouter` error codes | the router's repair message goes back to the model; consecutive-repair budget (as in the T9 demo loop) |
| Tool keeps failing | per-tool consecutive failures | circuit breaker: after 3, **withdraw the tool** from the offered set and tell the model |
| Repeating itself | same `(tool, args)` and same result seen again | warn at the 2nd repeat, withdraw the tool at the 3rd |
| No progress | state signature unchanged for N steps | escalation ladder: nudge, then restrict tools, then wrap-up |
| Invalid final output (schema) | validation of the answer | feed the errors back and retry (2), else `partial` |
| Context overflow | `prompt_eval_count` near `num_ctx`, or a provider error | `ContextStrategy` shrinks history; if still over, wrap-up |
| Budget exhaustion | counters in state | **wrap-up turn** (tools disabled): a best-effort answer, status `partial` |
| Cancellation / deadline | cancellation token, clock | cancel in-flight model and tool work, checkpoint, `cancelled` |
| Process crash | resume from the log | replay events; **WRITE tools are at-most-once** (below) |

**Escalation ladder.** Repeated or unproductive behaviour escalates instead of repeating the same nudge: *nudge* → *restrict tools* → *wrap-up (no tools)* → *fail*. The ladder is data in `decide`, not scattered `if`s.

**Resume semantics.** The log records `ToolsDispatching` before a batch runs and `ToolResulted` after. On resume, a batch with no results is re-run only for `NONE`/`READ` tools; a `WRITE` call whose outcome is unknown is **never re-executed**: the model is told "the run was interrupted while this call was in progress; its outcome is unknown, verify before retrying". Model calls are at-least-once (they are idempotent for our purposes).

## Design decisions
1. **Native tool calling, not text parsing.** The loop speaks `toolrouter`'s JSON tool calls end to end. "ReAct" here means the reason-act-observe cycle; the "thought" is the text a model emits alongside its calls.
2. **Event sourcing with a pure core.** State is a fold over events; `decide` is pure. Persistence, resume, replay, the observability hand-off and most tests all fall out of this one choice.
3. **Bounded on every axis, with graceful degradation.** Steps, tool calls, tokens, wall time, per-reply timeout, and total recoveries are all budgets. Hitting one triggers a wrap-up turn and a `partial` result, not an exception.
4. **Recovery is visible and honest.** No silent retries: every recovery is an event and a `RunResult.recoveries` entry.
5. **Safe by default.** `ReadOnlyPolicy` unless writes are enabled; a run never re-executes an uncertain WRITE; persisted logs redact secrets; the database is created `0600`.
6. **Own the context seam, not the context algorithm.** `ContextStrategy.prepare(state) -> messages` is the contract; the default is a window that keeps the system prompt, the task, and recent turns, truncates old tool outputs head-and-tail, and never separates a tool call from its results. Smarter strategies are project 4.
7. **Tell Ollama how much context it has.** Ollama silently truncates input beyond `num_ctx` (default 4096 tokens), which in an agent loop quietly deletes the system prompt and early observations. The Ollama adapter sets `num_ctx` explicitly and the runtime compares `prompt_eval_count` against it to detect pressure.
8. **Model access goes through a small `ModelClient` protocol.** The `llmcli` adapter is the default implementation; tests use `ScriptedModel` and `ChaosModel`, so the loop is tested without a network.
9. **Concurrency without shared state.** One `AgentRuntime` can host many runs; each run owns its state, log and budgets.
10. **Chaos testing is a first-class deliverable.** `ChaosModel` and `FlakyTool` ship in `localagent.testing` so every recovery path is exercised by fuzzing and reusable by later projects.

## Milestones
| # | Deliverable | Acceptance criteria |
|---|---|---|
| R0 | **Prerequisite, in `llm-cli`:** tool calling | **Done.** (llm-cli 0.2.0, plan row M11; the 67 original offline tests pass unchanged, plus 44 new offline tests and 3 real-model tests.)  Additive and backwards compatible: tool-call/tool-result message parts and a `tool` role; `tools` on `LLMRequest`; a `ToolCallEvent` in the stream and `tool_calls` on `LLMResponse`; Ollama maps tools both ways; `FakeProvider` supports it; no retry once any output (including a tool call) has been streamed; every existing llm-cli test still passes; one real-model smoke test |
| R1 | Scaffold + model seam + minimal loop | **Done.**  `ModelClient`, `ModelTurn`, `ScriptedModel`; a no-tools run returns a `RunResult`; `uv sync` works |
| R2 | The ReAct loop on `toolrouter` | **Done.**  Think, act, observe over scripted models; budgets (steps, tool calls); pure `reduce`/`decide`; in-memory event log; `RunResult`; replaying the log reproduces the state |
| R3 | Ollama client | Via `llmcli`: native tool calling, usage, `num_ctx` set explicitly, per-reply timeout, transient-error retry with backoff; streaming deltas; smoke test against a real model |
| R4 | Self-healing I (model-side faults) | Empty-turn nudges, bounded model retries, malformed-call repair budget; `FallbackModel(primary, secondary)` wrapper switches to the pluggable secondary when the primary keeps failing, and records it; each recovery is an event; none can loop forever |
| R5 | Self-healing II (behavioural faults) | Repetition and no-progress detection, per-tool circuit breaker, the escalation ladder; chaos tests show no run exceeds its budgets |
| R6 | Output validation + wrap-up | Optional structured final output with bounded repair; wrap-up turn on budget exhaustion yields `partial` with an honest status |
| R7 | Context management seam | `ContextStrategy` + default `WindowStrategy`; token accounting from usage; overflow detection and shrink; never splits a call from its results |
| R8 | Persistence and resume | SQLite event store; resume from any point; WRITE at-most-once; **crash-consistency test**: crash after every event index, resume, and the outcome matches an uninterrupted run |
| R9 | Cancellation, concurrency, CLI | Clean cancel and wall-clock deadline; many concurrent runs; `localagent run/resume/runs/replay/tools` with a live event stream; real-model smoke tests; recovery rates measured on a real model |
| R10 | Polish | Executable README, examples, packaging verified with `uv tool install`, fuzz scaling (`FUZZ_SCALE`), chaos suite documented |

**Usable: R3. Self-healing complete: R6. Production-grade core: R8.**

## Testing
- **Pure-core unit tests:** `reduce`/`decide` over hand-built states; each detector and each rung of the escalation ladder has a test with no model, no clock, no I/O.
- **Scripted scenarios** (`ScriptedModel`): happy path; every fault in the taxonomy and its recovery; repair then success; budget exhaustion then wrap-up; structured-output repair.
- **Chaos / property tests (Hypothesis):** a `ChaosModel` emits arbitrary junk (empty turns, malformed JSON, unknown tools, huge outputs, repeated calls, errors, hangs) and `FlakyTool`s fail randomly. Properties that must always hold:
  - the run terminates and returns a `RunResult`; it never raises (except real cancellation);
  - model calls, tool calls, tokens and time stay within budgets (with a fake clock);
  - recoveries never exceed `max_recoveries`;
  - the event log is consistent: monotonic `seq`, every dispatched call is resolved exactly once, `RunFinished` is last and unique, and `reduce(log)` equals the live state.
- **Crash-consistency:** for a deterministic scenario, crash after event *k* for every *k*, resume, and assert the final result and transcript equal the uninterrupted run, with each WRITE tool executed at most once in total.
- **Safety:** read-only by default; secrets never appear in the persisted log; an interrupted WRITE is never re-run.
- **Cancellation:** cancel at every step; nothing keeps running; checkpoint is consistent.
- **Real-model smoke tests** (auto-skipped without Ollama): the loop completes a tool task, safety invariants hold, and recovery rates are *measured*, not assumed.
- **Mutation checks** on the guarantees (budget enforcement, WRITE at-most-once, wrap-up) as in the earlier projects.

## Risks
- **Small models loop, ramble, or ignore tools.** That is the reason the project exists; the escalation ladder and the chaos suite are the answer, and real-model runs measure how often each recovery fires.
- **Ollama's silent context truncation** (decision 7). Mitigated by setting `num_ctx` and tracking `prompt_eval_count`; the real-model tests assert the system prompt survives a long run.
- **Resuming side effects.** The at-most-once rule for WRITE tools is conservative by design; the cost is that the model must re-verify after a crash.
- **Changing a finished project (R0).** Kept additive and backwards compatible, with the existing llm-cli suite as the safety net.
- **Scope creep into context budgeting, memory and approvals.** Each has a named seam and a later project; this one ships the seams and a default only.
- **Flaky real-model tests.** Assert invariants and use fixed seeds and temperature 0 where an outcome is asserted; keep correctness in the deterministic tiers.

## Decisions (resolved)
| # | Question | Decision |
|---|---|---|
| 1 | How the runtime reaches the model | Extend `llm-cli` **additively** (R0); the runtime depends on `llmcli` and `toolrouter` via `uv` path dependencies |
| 2 | Tool-use style | **Native tool calling only** (no text-ReAct fallback) |
| 3 | Safety posture | **Read-only unless writes are enabled** |
| 4 | Names | Folder `local-agent-runtime/`, package `localagent`, CLI `localagent` |
| 5 | Model fallback | A **`FallbackModel` wrapper** around `ModelClient` with a pluggable secondary model; outside the core loop (built in R4) |
| 6 | Persistence and resume | **In v1** (R8) |
| 7 | Test models | `llama3.2` (3B) for the default tier; a stronger tool-capable model (7B-class Qwen) to compare recovery rates when measuring in R9. Ollama models must be (re)pulled first |

## R0 findings (what the real model showed)
- **Ollama streams each tool call whole, in one non-final chunk** (`done=False`), with the usage counts in the final chunk. It assigns no call ids, so `llmcli` numbers them per response (`call_0`, `call_1`, ...); the runtime must therefore treat ids as unique *per model turn* and namespace them per run.
- **Results are matched to calls by tool name and order**, not id, and there is no error flag, so failure text must be explicit in the result content. `toolrouter`'s `model_text` already does this (`Error (<code>): ...`).
- **A tool call counts as output.** `llmcli` never retries a request after a tool call has streamed (it may already have caused a side effect). The runtime's own retry logic (R3/R4) must keep that rule: a model call that produced tool calls is never silently replayed.
- Tools and a response schema cannot be combined in one request, so structured *final* output (R6) is a separate, tool-free model call, which matches the plan's wrap-up turn design.
- The `llmcli` message model now carries tool parts, so the runtime can store and replay full transcripts using `llmcli`'s own `Message` type.

## R1 notes (what was built, and what is deliberately temporary)
R1 builds the architecture's spine at its smallest, so R2 adds tools without rewriting the loop: `events` (the log), `state` (`reduce`, `replay`), `decide`, a thin `AgentRuntime` shell, `ModelClient`/`ModelTurn`/`TurnRequest` (reusing `llmcli`'s message and usage types), and the public test doubles `ScriptedModel` and `FakeClock`. `RunStarted` carries the whole configuration, so a log alone rebuilds a run.

`decide` currently contains four **temporary** rules that later milestones replace; each has a test that pins today's behaviour and will change on purpose:
- a reply with tool calls is a failed run (R2 dispatches them);
- an empty reply is a failed run, not a completed one with an empty answer (R4 nudges instead);
- a model error ends the run (R4 adds bounded retries and fallback);
- hitting the step budget is a failure (R6 turns it into a wrap-up and a `partial` result).

Properties already fuzzed: any model behaviour yields a consistent result whose log replays to the same state; folding events one at a time equals replaying each prefix; `decide` is total, pure, and returns `Stop` exactly when the run is finished.

## R2 notes (the ReAct loop)
Think, act, observe now works end to end over scripted models and real `toolrouter` tools; `Agent` became a frozen dataclass (it holds a registry and a policy, which are not data) with a `ToolRegistry`, a `Policy` (default `ReadOnlyPolicy`, so writes are opt-in) and `Budgets` (`max_steps`, `max_tool_calls`, `model_timeout_s`).

Design points worth knowing:
- **Two new events.** `ToolsDispatching` is logged before a batch runs; one `ToolResulted` follows per call, keyed by the call's **position in the batch** (a model can repeat a call id). `ToolOutcome.text` is exactly what the model saw, so a log replays to the same transcript.
- **The reducer assembles the tool message.** Once every outcome of a batch is in, `reduce` builds one `tool` message with results in **call order**, however they arrived. R8 can therefore append results as each call finishes (for per-call durability) without touching the reducer.
- **Budgets refuse, they do not truncate.** A turn asking for more tool calls than the budget leaves is refused whole, before anything runs (R6 turns refusal into a wrap-up). A model that calls tools forever is stopped by `max_steps`.
- **Tool failures are observations, not run failures.** Bad arguments, unknown tools, denials, timeouts and handler errors go back to the model as `Error (<code>): ...` text (with `toolrouter`'s repair hints) and the run continues; a bounded repair budget arrives in R4.
- **No tools registered is still consistent.** If a model calls a tool it was never offered, the router answers "No tools are available" and the run goes on.
- **The system prompt** is the caller's verbatim if given; otherwise the default, plus tool guidance only when tools exist. `RunStarted` records the final prompt and the tool names.
- **Call ids are the provider's, unique per model turn only** (Ollama numbers them `call_0`...). Events always carry `step`, so `(step, index)` identifies a call within a run.
- `localagent.testing.call` takes `name` and `call_id` positionally-only, so tools may have arguments with those names (found by a test).

Still temporary in `decide`: an empty reply fails the run (R4), a model error ends it (R4), budget exhaustion fails instead of wrapping up (R6), and an unresolved tool batch fails (R8 resumes it). Persisted logs will redact secret tool arguments (R8); the in-memory log returned today holds the raw arguments the model sent.
