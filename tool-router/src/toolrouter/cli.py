"""`toolrouter`: inspect tools, call them by hand, and let a local model use them.

Needs the optional extras: `pip install "toolrouter[cli]"` (typer, rich, ollama).
Data goes to stdout; traces, summaries and errors go to stderr, so pipes work.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from toolrouter import (
    AllowAll,
    CallEvent,
    ReadOnlyPolicy,
    ResultEvent,
    RouterHook,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolRouter,
)
from toolrouter.demo.chat import DEFAULT_SYSTEM, ChatFn, ModelTimeoutError, run_chat
from toolrouter.demo.tools import make_demo_tools
from toolrouter.errors import ToolDefinitionError
from toolrouter.feedback import model_text, signature
from toolrouter.formats import anthropic, openai
from toolrouter.formats import ollama as ollama_fmt
from toolrouter.schema import tool_schema

EXIT_USAGE = 2
EXIT_MODEL = 3  # the model server is unreachable, or the model is missing / cannot use tools
EXIT_GAVE_UP = 4
EXIT_INTERRUPTED = 130

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Inspect tools, call them by hand, and let a local model use them safely.",
    pretty_exceptions_show_locals=False,
)
out = Console()
err = Console(stderr=True)


def _show_version(value: bool) -> None:
    if value:
        try:
            typer.echo(f"toolrouter {version('toolrouter')}")
        except PackageNotFoundError:
            typer.echo("toolrouter (not installed)")
        raise typer.Exit()


@app.callback()
def _root(
    _version: Annotated[
        bool, typer.Option("--version", callback=_show_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Inspect tools, call them by hand, and let a local model use them safely."""


class SchemaFormat(StrEnum):
    ollama = "ollama"
    anthropic = "anthropic"
    openai = "openai"
    openai_responses = "openai-responses"
    json = "json"


def _fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    err.print(f"[red]error:[/red] {escape(message)}", soft_wrap=True)
    raise typer.Exit(code)


# ---- loading tools ----------------------------------------------------------------------------


def _tools_from_module(spec: str) -> list[Tool]:
    """Import a module (dotted name or a .py path) and collect its module-level Tool objects.

    This executes that module's code: only point it at code you trust.
    """
    try:
        if spec.endswith(".py"):
            path = Path(spec)
            module_spec = importlib.util.spec_from_file_location(
                f"_toolrouter_user_{path.stem}", path
            )
            if module_spec is None or module_spec.loader is None:
                _fail(f"Cannot load {spec}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(spec)
    except (ImportError, OSError, SyntaxError, ToolDefinitionError) as exc:
        _fail(f"Cannot load tools from {spec!r}: {type(exc).__name__}: {exc}")
    found = list({id(v): v for v in vars(module).values() if isinstance(v, Tool)}.values())
    if not found:
        _fail(f"No tools found in {spec!r} (expected module-level objects made with @tool)")
    return found


def _registry(tool_modules: list[str], files_dir: Path) -> ToolRegistry:
    try:
        if tool_modules:
            tools = [t for spec in tool_modules for t in _tools_from_module(spec)]
        else:
            tools = make_demo_tools(files_dir)
        return ToolRegistry(tools)
    except ToolDefinitionError as exc:
        _fail(str(exc))


ToolsOption = Annotated[
    list[str],
    typer.Option(
        "--tools",
        help="Module or .py file to load tools from (repeatable). Replaces the demo tools. "
        "Executes that code.",
    ),
]
FilesDirOption = Annotated[
    Path,
    typer.Option("--files-dir", help="Directory the demo file tools are confined to."),
]


# ---- list / schema / call -----------------------------------------------------------------------


@app.command("list")
def list_tools(tools: ToolsOption = [], files_dir: FilesDirOption = Path(".")) -> None:  # noqa: B006
    """List the available tools with their signatures."""
    for t in _registry(tools, files_dir):
        out.print(f"{t.name}  [{t.side_effect.value}]", markup=False, highlight=False)
        out.print(f"  {signature(t)}", markup=False, highlight=False, soft_wrap=True)
        out.print(f"  {t.description}", markup=False, highlight=False, soft_wrap=True)


@app.command()
def schema(
    tool: Annotated[str | None, typer.Argument(help="Tool name (default: all tools).")] = None,
    format: Annotated[SchemaFormat, typer.Option("--format", help="Payload shape.")] = (
        SchemaFormat.ollama
    ),
    tools: ToolsOption = [],  # noqa: B006
    files_dir: FilesDirOption = Path("."),
) -> None:
    """Print tool definitions in a provider's JSON format."""
    registry = _registry(tools, files_dir)
    chosen = list(registry)
    if tool is not None:
        found = registry.get(tool)
        if found is None:
            _fail(f"Unknown tool {tool!r}. Available: {', '.join(registry.names())}")
        chosen = [found]
    payload: Any
    match format:
        case SchemaFormat.ollama:
            payload = ollama_fmt.tools(chosen)
        case SchemaFormat.anthropic:
            payload = anthropic.tools(chosen)
        case SchemaFormat.openai:
            payload = openai.tools(chosen)
        case SchemaFormat.openai_responses:
            payload = openai.tools(chosen, api="responses")
        case SchemaFormat.json:
            payload = {t.name: tool_schema(t) for t in chosen}
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def call(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    args: Annotated[str, typer.Option("--args", "-a", help="Arguments as a JSON object.")] = "{}",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the full result as JSON.")
    ] = False,
    coerce: Annotated[bool, typer.Option("--coerce", help="Lenient argument validation.")] = False,
    read_only: Annotated[bool, typer.Option("--read-only", help="Deny WRITE tools.")] = False,
    tools: ToolsOption = [],  # noqa: B006
    files_dir: FilesDirOption = Path("."),
) -> None:
    """Call one tool through the router, exactly as a model's call would be handled."""
    router = ToolRouter(
        _registry(tools, files_dir),
        coerce=coerce,
        policy=ReadOnlyPolicy() if read_only else AllowAll(),
    )
    try:
        result = asyncio.run(router.dispatch(ToolCall(id="cli", name=tool, arguments=args)))
    except KeyboardInterrupt:
        err.print("\n[yellow]cancelled[/yellow]")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    elif result.ok:
        typer.echo(result.content)
    else:
        err.print(escape(model_text(result)), soft_wrap=True)
    raise typer.Exit(0 if result.ok else 1)


# ---- chat ---------------------------------------------------------------------------------------


class TraceHook(RouterHook):
    """Prints each tool call and its outcome to stderr (arguments are redacted by the router)."""

    def before(self, event: CallEvent) -> None:
        args = json.dumps(event.arguments, ensure_ascii=False, default=repr)
        err.print(f"[dim]→ {escape(event.name)}({escape(args)})[/dim]", soft_wrap=True)

    def after(self, event: ResultEvent) -> None:
        r = event.result
        if r.ok:
            preview = " ".join((r.content or "").split())
            preview = preview if len(preview) <= 100 else preview[:99] + "…"
            err.print(
                f"[green]✓[/green] {escape(r.name)} [dim]{r.duration_ms:.0f}ms[/dim] "
                f"{escape(preview)}",
                soft_wrap=True,
            )
        else:
            assert r.error is not None
            err.print(
                f"[red]✗[/red] {escape(r.name)} [dim]{r.error.code.value}[/dim] "
                f"{escape(r.error.message[:200])}",
                soft_wrap=True,
            )


def _ollama_chat_fn(model: str, host: str | None, max_tokens: int) -> ChatFn:
    import ollama

    client = ollama.AsyncClient(host=host)

    async def chat_fn(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        # num_predict caps one reply so a model stuck in a repetition loop cannot run forever
        response = await client.chat(
            model=model, messages=messages, tools=tools, options={"num_predict": max_tokens}
        )
        return response.message

    return chat_fn


def _model_error_message(exc: Exception, model: str) -> str | None:
    """A friendly message for the failures a user can fix, or None for anything else."""
    try:
        import httpx
        import ollama
    except ImportError:  # pragma: no cover
        return None
    if isinstance(exc, ModelTimeoutError):
        return f"{exc}. Try a smaller model, a shorter prompt, or --model-timeout."
    if isinstance(exc, ollama.ResponseError):
        text = str(exc)
        if exc.status_code == 404:
            return f"Model {model!r} not found. Run: ollama pull {model}"
        if "tool" in text.lower():
            return f"Model {model!r} cannot use tools ({text}). Try a tool-capable model."
        return f"Ollama error ({exc.status_code}): {text}"
    if isinstance(exc, ConnectionError | httpx.TransportError):
        return "Cannot reach Ollama. Is it running? (try: brew services start ollama)"
    return None


@app.command()
def chat(
    prompt: Annotated[str, typer.Argument(help="What to ask the model.")],
    model: Annotated[
        str, typer.Option("--model", "-m", envvar="TOOLROUTER_MODEL", help="Ollama model.")
    ] = "llama3.2",
    host: Annotated[str | None, typer.Option("--host", envvar="OLLAMA_HOST")] = None,
    max_turns: Annotated[int, typer.Option("--max-turns", min=1)] = 6,
    max_repairs: Annotated[
        int, typer.Option("--max-repairs", min=0, help="Consecutive failed-call repairs allowed.")
    ] = 2,
    allow_writes: Annotated[
        bool,
        typer.Option("--allow-writes", help="Let the model use WRITE tools (default: read-only)."),
    ] = False,
    coerce: Annotated[bool, typer.Option("--coerce", help="Lenient argument validation.")] = False,
    system: Annotated[str | None, typer.Option("--system", "-s")] = None,
    model_timeout: Annotated[
        float, typer.Option("--model-timeout", min=0.1, help="Seconds allowed per model reply.")
    ] = 120.0,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", min=1, help="Longest single model reply, in tokens.")
    ] = 1024,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No trace or summary.")] = False,
    tools: ToolsOption = [],  # noqa: B006
    files_dir: FilesDirOption = Path("."),
) -> None:
    """Ask a local tool-capable model a question; it may call the tools to answer.

    Read-only by default. The answer goes to stdout; the tool trace goes to stderr.
    """
    router = ToolRouter(
        _registry(tools, files_dir),
        coerce=coerce,
        policy=AllowAll() if allow_writes else ReadOnlyPolicy(),
        hooks=[] if quiet else [TraceHook()],
    )
    try:
        report = asyncio.run(
            run_chat(
                prompt,
                router=router,
                chat_fn=_ollama_chat_fn(model, host, max_tokens),
                system=system if system is not None else DEFAULT_SYSTEM,
                max_turns=max_turns,
                max_repairs=max_repairs,
                model_timeout_s=model_timeout,
            )
        )
    except KeyboardInterrupt:
        err.print("\n[yellow]cancelled[/yellow]")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:
        friendly = _model_error_message(exc, model)
        if friendly is None:
            raise
        _fail(friendly, EXIT_MODEL)

    if not quiet:
        failures = ", ".join(f"{k}={v}" for k, v in sorted(report.failures.items())) or "none"
        err.print(
            f"[dim]turns={report.turns} tool_calls={report.tool_calls} "
            f"repairs={report.repair_turns} failures: {escape(failures)}[/dim]",
            soft_wrap=True,
        )
    if report.gave_up is not None:
        _fail(f"The model {report.gave_up}.", EXIT_GAVE_UP)
    typer.echo(report.answer)
