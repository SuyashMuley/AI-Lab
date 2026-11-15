"""Typer commands only. Delegates everything to the service layer."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError
from rich.markup import escape

from llmcli.config import store
from llmcli.config.settings import (
    SECRET_FIELDS,
    Settings,
    data_path,
    project_config_path,
    user_config_path,
)
from llmcli.core.errors import InvalidRequestError, LLMError, StorageError
from llmcli.core.events import Done, TextDelta
from llmcli.core.models import LLMRequest, LLMResponse, Message
from llmcli.providers.registry import get_provider
from llmcli.services.llm import LLMService, RetryPolicy
from llmcli.services.pricing import estimate_cost
from llmcli.storage.sessions import SessionStore
from llmcli.ui.console import err, format_usage, out

# show_locals off: tracebacks must never dump variables (which could hold secrets).
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Talk to LLMs.",
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(no_args_is_help=True, help="View and edit configuration.")
app.add_typer(config_app, name="config")
history_app = typer.Typer(no_args_is_help=True, help="Browse and manage saved conversations.")
app.add_typer(history_app, name="history")

EXIT_INTERRUPTED = 130  # 128 + SIGINT, the shell convention


def _fail(exc: LLMError) -> NoReturn:
    err.print(f"[red]error:[/red] {escape(str(exc))}", soft_wrap=True)
    raise typer.Exit(exc.exit_code) from exc


def _load_settings(**flags: Any) -> Settings:
    """Build Settings; flags (when not None) override env and config files."""
    try:
        return Settings(**{k: v for k, v in flags.items() if v is not None})
    except ValidationError as exc:
        fields = ", ".join(str(e["loc"][0]) if e["loc"] else "?" for e in exc.errors())
        raise InvalidRequestError(
            f"Invalid configuration for: {fields}. Run `llm config list` to inspect."
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise InvalidRequestError(f"Malformed config file: {exc}") from exc


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text("utf-8"))
    except OSError as exc:
        raise InvalidRequestError(f"Cannot read schema file {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"Schema file {path} is not valid JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise InvalidRequestError("Schema must be a JSON object")
    return schema


async def _stream_to_stdout(service: LLMService, request: LLMRequest) -> LLMResponse | None:
    """Write deltas to stdout as they arrive; return the final response."""
    last = ""
    response: LLMResponse | None = None
    async for ev in service.stream(request):
        if isinstance(ev, TextDelta):
            sys.stdout.write(ev.text)
            sys.stdout.flush()
            last = ev.text
        elif isinstance(ev, Done):
            response = ev.response
    if last and not last.endswith("\n") and sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()
    return response


def _version(value: bool) -> None:
    if value:
        try:
            out.print(f"llm {version('llmcli')}", markup=False)
        except PackageNotFoundError:
            out.print("llm (uninstalled)", markup=False)
        raise typer.Exit()


@app.callback()
def _root(
    _version_flag: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Talk to LLMs."""


def _store() -> SessionStore:
    return SessionStore(data_path() / "sessions.db")


def _resolve_session(continue_: bool, session: int | None) -> int | None:
    """Which saved session this turn continues, if any."""
    if continue_ and session is not None:
        raise InvalidRequestError("Use either --continue or --session, not both")
    if session is not None:
        _store().get(session)  # validates it exists
        return session
    if continue_:
        latest = _store().latest_id()
        if latest is None:
            raise InvalidRequestError("No previous session to continue. Run `llm ask` first.")
        return latest
    return None


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="The prompt to send.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model or alias.")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    system: Annotated[str | None, typer.Option("--system", "-s")] = None,
    stream: Annotated[bool, typer.Option("--stream", help="Print output as it arrives.")] = False,
    usage: Annotated[bool, typer.Option("--usage", help="Show tokens/latency on stderr.")] = False,
    schema: Annotated[
        Path | None,
        typer.Option("--schema", help="JSON schema file; output is validated against it."),
    ] = None,
    timeout: Annotated[
        float | None, typer.Option("--timeout", help="Seconds of provider silence before failing.")
    ] = None,
    continue_: Annotated[
        bool, typer.Option("--continue", "-c", help="Continue the most recent session.")
    ] = False,
    session: Annotated[
        int | None, typer.Option("--session", "-S", help="Continue the session with this id.")
    ] = None,
    no_save: Annotated[
        bool, typer.Option("--no-save", help="Don't save this exchange to history.")
    ] = False,
) -> None:
    """Send a prompt and print the response. Saved to history unless --no-save."""
    try:
        settings = _load_settings(model=model, provider=provider, timeout=timeout)
        if schema is not None and stream:
            raise InvalidRequestError("--schema cannot be combined with --stream")
        schema_doc = _load_schema(schema) if schema is not None else None
        session_id = _resolve_session(continue_, session)
        history: list[Message] = []
        if session_id is not None:
            history = _store().messages(session_id)
            if system is None:
                system = _store().get(session_id).system  # keep the session's persona
        model_name = settings.aliases.get(settings.model, settings.model)
        request = LLMRequest(
            model=model_name, system=system, messages=[*history, Message.of("user", prompt)]
        )
        service = LLMService(
            get_provider(settings.provider, settings),
            retry=RetryPolicy(max_attempts=settings.max_retries + 1),
            timeout=settings.timeout,
            schema_retries=settings.schema_retries,
        )
        response: LLMResponse | None
        if stream:
            response = asyncio.run(_stream_to_stdout(service, request))
        elif schema_doc is not None:
            data, response = asyncio.run(service.generate_structured(request, schema_doc))
            out.print_json(data=data)
        else:
            response = asyncio.run(service.generate(request))
            out.print(response.text, markup=False, highlight=False)
    except KeyboardInterrupt:
        # Never report success on cancel; nothing is saved, partial output stays on screen.
        err.print("\n[yellow]cancelled[/yellow]")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except LLMError as exc:
        sys.stdout.flush()
        _fail(exc)
    if usage and response is not None:
        cost = estimate_cost(settings.provider, model_name, response.usage)
        err.print(format_usage(response.usage, cost), markup=False, highlight=False, soft_wrap=True)
    if response is not None and settings.save_history and not no_save:
        try:
            _store().save_turn(
                session_id,
                provider=settings.provider,
                model=model_name,
                system=system,
                user_text=prompt,
                assistant_text=response.text,
            )
        except StorageError as exc:
            # The answer was already delivered; don't turn a save failure into a failed run.
            err.print(f"[yellow]warning:[/yellow] not saved: {escape(str(exc))}", soft_wrap=True)


# ---- llm history -----------------------------------------------------------


def _one_line(text: str, width: int = 50) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


@history_app.command("list")
def history_list(limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 20) -> None:
    """List recent sessions, newest first."""
    try:
        sessions = _store().list_sessions(limit)
    except LLMError as exc:
        _fail(exc)
    if not sessions:
        err.print("No saved sessions.", markup=False)
        return
    for s in sessions:
        when = s.updated_at[:16].replace("T", " ")
        out.print(
            f"{s.id:>4}  {when}  {s.model:<16} {s.message_count:>3} msgs  {_one_line(s.title)}",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


@history_app.command("show")
def history_show(
    session_id: Annotated[int | None, typer.Argument(help="Session id (default: latest).")] = None,
) -> None:
    """Print a session's transcript."""
    try:
        sid = session_id if session_id is not None else _store().latest_id()
        if sid is None:
            raise InvalidRequestError("No saved sessions.")
        info = _store().get(sid)
        messages = _store().messages(sid)
    except LLMError as exc:
        _fail(exc)
    err.print(f"session {info.id} | {info.provider}/{info.model}", markup=False, soft_wrap=True)
    if info.system:
        out.print(f"system: {info.system}\n", markup=False, highlight=False)
    for m in messages:
        out.print(f"{m.role}: {m.text}\n", markup=False, highlight=False)


@history_app.command("delete")
def history_delete(session_id: int) -> None:
    """Delete one session."""
    try:
        _store().delete(session_id)
    except LLMError as exc:
        _fail(exc)
    err.print(f"deleted session {session_id}", markup=False)


@history_app.command("clear")
def history_clear(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete ALL saved sessions."""
    if not yes and not typer.confirm("Delete all saved conversations?"):
        raise typer.Exit(1)
    try:
        n = _store().clear()
    except LLMError as exc:
        _fail(exc)
    err.print(f"deleted {n} session(s)", markup=False)


# ---- llm config ------------------------------------------------------------


def _display(value: Any) -> str:
    return json.dumps(value) if isinstance(value, dict | list) else str(value)


def _effective(settings: Settings) -> dict[str, str]:
    """Effective config as display strings; secrets are masked, never printed."""
    rows: dict[str, str] = {}
    for name in Settings.model_fields:
        value = getattr(settings, name)
        if name in SECRET_FIELDS:
            rows[name] = "(set)" if value is not None else "(not set)"
        elif name == "aliases":
            for alias, target in sorted(value.items()):
                rows[f"alias.{alias}"] = target
        else:
            rows[name] = "(unset)" if value is None else _display(value)
    return rows


@config_app.command("list")
def config_list() -> None:
    """Show effective configuration (secrets masked)."""
    try:
        rows = _effective(_load_settings())
    except LLMError as exc:
        _fail(exc)
    for key, value in rows.items():
        out.print(f"{key} = {value}", markup=False, highlight=False)


@config_app.command("get")
def config_get(key: str) -> None:
    """Print one effective value."""
    try:
        rows = _effective(_load_settings())
    except LLMError as exc:
        _fail(exc)
    if key not in rows:
        _fail(InvalidRequestError(f"No such config key: {key!r}"))
    out.print(rows[key], markup=False, highlight=False)


@config_app.command("set")
def config_set(
    key: str,
    value: str,
    project: Annotated[
        bool, typer.Option("--project", help="Write ./.llmcli.toml instead of user config.")
    ] = False,
) -> None:
    """Persist a value, e.g. `llm config set model llama3.2:1b` or `alias.fast llama3.2:1b`."""
    path = project_config_path() if project else user_config_path()
    try:
        stored = store.set_key(path, key, value)
    except LLMError as exc:
        _fail(exc)
    err.print(f"set {key} = {stored} in {path}", markup=False, highlight=False, soft_wrap=True)


@config_app.command("unset")
def config_unset(
    key: str,
    project: Annotated[bool, typer.Option("--project")] = False,
) -> None:
    """Remove a persisted value."""
    path = project_config_path() if project else user_config_path()
    try:
        removed = store.unset_key(path, key)
    except LLMError as exc:
        _fail(exc)
    err.print("removed" if removed else "not set", markup=False)


@config_app.command("path")
def config_path() -> None:
    """Show config file locations."""
    out.print(f"user:    {user_config_path()}", markup=False, highlight=False, soft_wrap=True)
    out.print(f"project: {project_config_path()}", markup=False, highlight=False, soft_wrap=True)
