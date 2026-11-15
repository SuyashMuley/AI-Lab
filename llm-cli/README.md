# llm-cli

A small, well-tested Python CLI for talking to LLMs: **streaming, token usage, structured output, and saved conversations.** No agent frameworks; the code is split so providers are easy to add.

```console
$ llm ask "Say hi in five words" --usage
Hello there, nice to meet you.
tokens: 30 in / 7 out | 0.17s | 40.8 tok/s | ~$0.0000 (estimate)
```

> **Providers:** Ollama (local, free) is the only provider implemented today. The provider interface is in place; see [Adding a provider](#adding-a-provider).

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd llm-cli
uv tool install .        # puts `llm` on your PATH
llm --version
```

Or run from the checkout without installing: `uv run llm ...`

## Quick start (Ollama)

```bash
brew install ollama && brew services start ollama
ollama pull llama3.2:1b                 # small and fast; `llama3.2` is the default model name
llm config set model llama3.2:1b        # make it the default
llm ask "Explain recursion in one sentence"
```

## Usage

```bash
llm ask "prompt"                         # one-shot answer on stdout
llm ask "prompt" -m fast                 # model or alias        (-m / --model)
llm ask "prompt" -s "Answer like a pirate"                       # system prompt
llm ask "Write a story" --stream         # print tokens as they arrive; Ctrl+C cancels cleanly
llm ask "prompt" --usage                 # tokens + latency (+ cost estimate) on stderr
llm ask "prompt" --timeout 30            # seconds of provider silence before failing
```

### Structured output

```bash
cat > person.json <<'JSON'
{"type":"object",
 "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
 "required":["name","age"]}
JSON
llm ask "Invent a fictional person" --schema person.json | jq .name
```

The schema is sent natively when the provider supports it (and always described in the prompt). The reply is parsed and validated; on a mismatch the validation error is fed back to the model and it retries (`schema_retries`, default 2). If it still fails you get exit code 8 and a clear message. `--schema` can't be combined with `--stream`.

### Conversations

Every `ask` is saved to a local SQLite database. Only completed exchanges are saved: errors and Ctrl+C save nothing.

```bash
llm ask "My name is Priya"
llm ask "What's my name?" --continue     # -c: continue the most recent session
llm ask "And my cat?" --session 3        # -S: continue a specific session
llm ask "private question" --no-save     # don't record this one

llm history list                         # recent sessions
llm history show [ID]                    # transcript (default: latest)
llm history delete ID
llm history clear --yes
```

A session remembers its system prompt (override it with `-s`). To stop saving entirely: `llm config set save_history false`. The database lives at `~/.local/share/llmcli/sessions.db` and is created with `0600` permissions.

### Output conventions

Response data goes to **stdout**; usage, warnings, and errors go to **stderr**, so pipes and redirects work: `llm ask "..." --usage > answer.txt`.

## Configuration

Precedence (highest first): **CLI flags > environment variables > `./.llmcli.toml` > `~/.config/llmcli/config.toml` > defaults.** An `.env` file in the working directory counts as environment.

```bash
llm config list                      # effective values (secrets masked)
llm config get model
llm config set model llama3.2:1b     # user config
llm config set timeout 30 --project  # ./.llmcli.toml
llm config set alias.fast llama3.2:1b
llm config unset model
llm config path                      # where the files live
```

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `provider` | `LLMCLI_PROVIDER` | `ollama` | Which provider to use |
| `model` | `LLMCLI_MODEL` | `llama3.2` | Model name or alias |
| `ollama_host` | `LLMCLI_OLLAMA_HOST` | Ollama default | Ollama server URL |
| `timeout` | `LLMCLI_TIMEOUT` | `60` | Seconds of provider silence before failing |
| `max_retries` | `LLMCLI_MAX_RETRIES` | `2` | Retries on rate-limit / 5xx / connection errors |
| `schema_retries` | `LLMCLI_SCHEMA_RETRIES` | `2` | Retries when output fails `--schema` |
| `save_history` | `LLMCLI_SAVE_HISTORY` | `true` | Save conversations |
| `alias.<name>` | n/a | none | Short name for a model |
| `api_key` | `LLMCLI_API_KEY` | none | **Env only.** For remote providers |

**Secrets are env-only.** `llm config set api_key ...` is refused, keys in config files are ignored, `config list` shows only `(set)`/`(not set)`, and tracebacks never print local variables.

`LLMCLI_CONFIG_HOME` and `LLMCLI_DATA_HOME` relocate the config and data directories.

## Reliability

- **Timeouts** fire on provider silence (not total time), so long streams aren't cut off.
- **Retries**: bounded exponential backoff with jitter (0.5s doubling to 8s) on 429 / 5xx / connection errors only. Invalid requests, auth errors, missing models, and timeouts are never retried, and **nothing is retried once output has started streaming.**
- Errors are normalized across providers; the original exception is kept as `__cause__`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Unexpected provider error |
| 2 | Invalid request, flag, schema file, or config |
| 3 | Authentication failed |
| 4 | Rate limited |
| 5 | Provider unavailable / unreachable |
| 6 | Model not found |
| 7 | Timeout |
| 8 | Output never matched `--schema` |
| 9 | History database error |
| 130 | Interrupted (Ctrl+C) |

## Architecture

```
src/llmcli/
├── cli.py              # Typer commands only
├── core/               # models, events, errors, provider protocol. No SDK imports
├── providers/          # one adapter per provider + registry (SDK <-> internal types)
├── services/           # LLMService (timeouts, retries, structured output), pricing
├── storage/sessions.py # SQLite conversation history
├── config/             # Settings (precedence chain) + config file store
├── data/prices.json    # cost-estimate table
└── ui/console.py       # stdout/stderr consoles, usage formatting
```

Design rules: the CLI and services never touch SDK types; streaming is a sequence of typed events (`TextDelta`, `UsageEvent`, `Done`, `ErrorEvent`) and `generate()` is a fold over `stream()`; retries live in the service, not the SDKs.

### Cost estimates

`--usage` adds `~$X (estimate)` only for models listed in `src/llmcli/data/prices.json` (USD per 1M tokens, keys like `provider/model`, wildcards allowed). It ships with local Ollama models at $0; add entries for paid providers and check them against the provider's pricing page.

### Adding a provider

1. Create `providers/<name>.py` with a class that has `name`, `supports_native_schema`, and `stream(request) -> AsyncGenerator[Event]`. Map the SDK's stream to `TextDelta` / `UsageEvent` / `Done`, close the SDK stream (`aclosing`), and convert SDK exceptions to `llmcli.core.errors` types.
2. Register it in `providers/registry.py`.
3. Add price entries if it's paid, and a smoke test under `tests/integration/`.

## Development

```bash
uv sync
uv run pytest -q                       # unit tests + smoke tests (smoke auto-skip without Ollama)
uv run pytest -m "not integration"     # unit tests only, no network
uv run pytest -m integration           # smoke tests only (needs Ollama + llama3.2:1b)
uv run ruff check . && uv run ruff format --check .
uv run mypy                            # strict
```

Tests never touch your real config, history, or environment (`tests/conftest.py` isolates them). Unit tests use `FakeProvider` (`tests/fakes/`) for deterministic, network-free runs. Set `LLMCLI_SMOKE_MODEL` to smoke-test a different model.

See [PLAN.md](PLAN.md) for the milestone plan and design decisions.
