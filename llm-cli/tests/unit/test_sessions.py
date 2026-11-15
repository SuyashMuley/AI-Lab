from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmcli import cli
from llmcli.config.settings import data_path
from llmcli.core.errors import InvalidRequestError, ProviderUnavailableError, StorageError
from llmcli.storage.sessions import SessionStore
from tests.fakes.provider import FakeProvider

runner = CliRunner()


def turn(store: SessionStore, sid: int | None, user: str = "q", asst: str = "a") -> int:
    return store.save_turn(
        sid, provider="fake", model="m", system=None, user_text=user, assistant_text=asst
    )


# ---- store -----------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    sid = turn(store, None, "hello", "hi there")
    turn(store, sid, "again", "sure")
    msgs = store.messages(sid)
    assert [(m.role, m.text) for m in msgs] == [
        ("user", "hello"),
        ("assistant", "hi there"),
        ("user", "again"),
        ("assistant", "sure"),
    ]
    info = store.get(sid)
    assert info.message_count == 4
    assert info.title == "hello"


def test_latest_is_most_recently_updated(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    assert store.latest_id() is None
    a, b = turn(store, None), turn(store, None)
    assert store.latest_id() == b
    turn(store, a)  # touching `a` makes it the latest again
    assert store.latest_id() == a


def test_missing_session_errors(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    for call in (store.get, store.messages, store.delete):
        with pytest.raises(InvalidRequestError):
            call(99)
    with pytest.raises(InvalidRequestError):
        turn(store, 99)


def test_failed_save_leaves_no_partial_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    with pytest.raises(InvalidRequestError):
        turn(store, 99)  # unknown session: whole transaction rolls back
    assert store.list_sessions() == []


def test_delete_cascades_and_clear(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    a, b = turn(store, None), turn(store, None)
    store.delete(a)
    assert [s.id for s in store.list_sessions()] == [b]
    with sqlite3.connect(tmp_path / "s.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2  # only b's
    assert store.clear() == 1
    assert store.list_sessions() == []


def test_database_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "data" / "s.db"
    turn(SessionStore(path), None)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_corrupt_database_is_a_storage_error(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    path.write_text("this is not sqlite")
    with pytest.raises(StorageError):
        SessionStore(path).latest_id()


# ---- CLI -------------------------------------------------------------------


def ask(args: list[str], fake: FakeProvider, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "get_provider", lambda name, settings: fake)
    return runner.invoke(cli.app, ["ask", *args])


def db() -> SessionStore:
    return SessionStore(data_path() / "sessions.db")


def test_ask_saves_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ask(["hi"], FakeProvider(), monkeypatch).exit_code == 0
    sid = db().latest_id()
    assert sid is not None
    assert [m.text for m in db().messages(sid)] == ["hi", "Hello, world"]


def test_continue_sends_prior_history(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["first"], FakeProvider(), monkeypatch)
    fake = FakeProvider(chunks=["second answer"])
    r = ask(["second", "--continue"], fake, monkeypatch)
    assert r.exit_code == 0
    sent = [(m.role, m.text) for m in fake.requests[0].messages]
    assert sent == [("user", "first"), ("assistant", "Hello, world"), ("user", "second")]
    assert len(db().list_sessions()) == 1  # same session, not a new one
    assert db().get(db().latest_id() or 0).message_count == 4


def test_without_continue_starts_fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["one"], FakeProvider(), monkeypatch)
    fake = FakeProvider()
    ask(["two"], fake, monkeypatch)
    assert [m.text for m in fake.requests[0].messages] == ["two"]
    assert len(db().list_sessions()) == 2


def test_session_flag_targets_specific_session(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["old"], FakeProvider(), monkeypatch)
    old = db().latest_id()
    ask(["newer"], FakeProvider(), monkeypatch)
    fake = FakeProvider()
    ask(["more", "--session", str(old)], fake, monkeypatch)
    assert [m.text for m in fake.requests[0].messages][0] == "old"


def test_continue_with_no_history_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProvider()
    r = ask(["hi", "--continue"], fake, monkeypatch)
    assert r.exit_code == 2
    assert "No previous session" in r.stderr
    assert fake.requests == []


def test_unknown_session_and_conflicting_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ask(["hi", "--session", "42"], FakeProvider(), monkeypatch).exit_code == 2
    assert ask(["hi", "-c", "--session", "1"], FakeProvider(), monkeypatch).exit_code == 2


def test_system_prompt_is_remembered_across_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["hi", "-s", "Be a pirate"], FakeProvider(), monkeypatch)
    fake = FakeProvider()
    ask(["again", "-c"], fake, monkeypatch)
    assert fake.requests[0].system == "Be a pirate"
    fake2 = FakeProvider()
    ask(["again", "-c", "-s", "Be formal"], fake2, monkeypatch)
    assert fake2.requests[0].system == "Be formal"  # explicit flag wins


def test_no_save_flag_and_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["hi", "--no-save"], FakeProvider(), monkeypatch)
    assert db().latest_id() is None
    monkeypatch.setenv("LLMCLI_SAVE_HISTORY", "false")
    ask(["hi"], FakeProvider(), monkeypatch)
    assert db().latest_id() is None


def test_failed_or_cancelled_turns_are_not_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    r = ask(["hi"], FakeProvider(error=ProviderUnavailableError("down")), monkeypatch)
    assert r.exit_code != 0
    assert db().latest_id() is None

    class Interrupt(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt
            yield  # pragma: no cover

    r = ask(["hi", "--stream"], Interrupt(), monkeypatch)
    assert r.exit_code == 130
    assert db().latest_id() is None


def test_save_failure_warns_but_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    data_path().mkdir(parents=True)
    (data_path() / "sessions.db").write_text("corrupt")
    r = ask(["hi"], FakeProvider(), monkeypatch)
    assert r.exit_code == 0
    assert "Hello, world" in r.stdout
    assert "not saved" in r.stderr


def test_structured_turn_saves_validated_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schema = tmp_path / "s.json"
    schema.write_text('{"type":"object","required":["a"]}')
    ask(["hi", "--schema", str(schema)], FakeProvider(responses=[['{"a": 1}']]), monkeypatch)
    sid = db().latest_id() or 0
    assert db().messages(sid)[1].text == '{"a": 1}'


def test_history_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "No saved sessions" in runner.invoke(cli.app, ["history", "list"]).stderr
    ask(["what is gravity?", "-s", "Be brief"], FakeProvider(), monkeypatch)
    listing = runner.invoke(cli.app, ["history", "list"])
    assert "what is gravity?" in listing.stdout
    assert "2 msgs" in listing.stdout
    shown = runner.invoke(cli.app, ["history", "show"])
    assert "system: Be brief" in shown.stdout
    assert "user: what is gravity?" in shown.stdout
    assert "assistant: Hello, world" in shown.stdout
    assert runner.invoke(cli.app, ["history", "show", "99"]).exit_code == 2

    sid = str(db().latest_id())
    assert runner.invoke(cli.app, ["history", "delete", sid]).exit_code == 0
    assert db().latest_id() is None
    assert runner.invoke(cli.app, ["history", "delete", sid]).exit_code == 2


def test_history_clear_needs_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    ask(["a"], FakeProvider(), monkeypatch)
    assert runner.invoke(cli.app, ["history", "clear"], input="n\n").exit_code == 1
    assert db().latest_id() is not None
    assert runner.invoke(cli.app, ["history", "clear", "--yes"]).exit_code == 0
    assert db().latest_id() is None


def test_tests_never_touch_real_history() -> None:
    assert "userdata" in str(data_path())
