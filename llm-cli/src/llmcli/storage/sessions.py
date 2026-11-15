"""SQLite-backed conversation sessions (M9). Stdlib only, no ORM.

Only completed turns are saved: a session never contains a half-finished exchange.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from llmcli.core.errors import InvalidRequestError, StorageError
from llmcli.core.models import Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    system     TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


@dataclass(frozen=True)
class SessionInfo:
    id: int
    created_at: str
    updated_at: str
    provider: str
    model: str
    system: str | None
    message_count: int
    title: str  # first user prompt


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class SessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _connect(self) -> sqlite3.Connection:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            new = not self._path.exists()
            conn = sqlite3.connect(self._path)
            if new:
                os.chmod(self._path, 0o600)  # conversations are private
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)
            return conn
        except (sqlite3.Error, OSError) as exc:
            raise StorageError(f"Cannot open history database {self._path}: {exc}") from exc

    @contextmanager
    def _db(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Connection that always closes; `write=True` wraps one transaction."""
        conn = self._connect()
        try:
            if write:
                with conn:  # commit on success, rollback on any exception
                    yield conn
            else:
                yield conn
        except sqlite3.Error as exc:
            raise StorageError(f"History database error: {exc}") from exc
        finally:
            conn.close()

    # -- reads ---------------------------------------------------------------

    def latest_id(self) -> int | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return int(row["id"]) if row else None

    def get(self, session_id: int) -> SessionInfo:
        with self._db() as conn:
            rows = self._info_rows(conn, "WHERE s.id = ?", (session_id,))
        if not rows:
            raise InvalidRequestError(f"No session with id {session_id}")
        return rows[0]

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        with self._db() as conn:
            return self._info_rows(conn, "ORDER BY s.updated_at DESC, s.id DESC LIMIT ?", (limit,))

    def messages(self, session_id: int) -> list[Message]:
        with self._db() as conn:
            self._require(conn, session_id)
            rows = conn.execute(
                "SELECT role, text FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [Message.of(_role(r["role"]), r["text"]) for r in rows]

    # -- writes --------------------------------------------------------------

    def save_turn(
        self,
        session_id: int | None,
        *,
        provider: str,
        model: str,
        system: str | None,
        user_text: str,
        assistant_text: str,
    ) -> int:
        """Atomically append one completed exchange; creates the session if needed."""
        now = _now()
        with self._db(write=True) as conn:  # inner `conn` = one transaction
            if session_id is None:
                cur = conn.execute(
                    "INSERT INTO sessions (created_at, updated_at, provider, model, system)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (now, now, provider, model, system),
                )
                session_id = int(cur.lastrowid or 0)
            else:
                self._require(conn, session_id)
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, provider = ?, model = ? WHERE id = ?",
                    (now, provider, model, session_id),
                )
            conn.executemany(
                "INSERT INTO messages (session_id, role, text, created_at) VALUES (?, ?, ?, ?)",
                [
                    (session_id, "user", user_text, now),
                    (session_id, "assistant", assistant_text, now),
                ],
            )
        return session_id

    def delete(self, session_id: int) -> None:
        with self._db(write=True) as conn:
            self._require(conn, session_id)
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def clear(self) -> int:
        with self._db(write=True) as conn:
            n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            conn.execute("DELETE FROM sessions")
        return int(n)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _require(conn: sqlite3.Connection, session_id: int) -> None:
        if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
            raise InvalidRequestError(f"No session with id {session_id}")

    @staticmethod
    def _info_rows(
        conn: sqlite3.Connection, tail: str, params: tuple[object, ...]
    ) -> list[SessionInfo]:
        rows = conn.execute(
            "SELECT s.*,"
            " (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n,"
            " (SELECT text FROM messages m WHERE m.session_id = s.id AND m.role = 'user'"
            "  ORDER BY m.id LIMIT 1) AS title"
            f" FROM sessions s {tail}",
            params,
        ).fetchall()
        return [
            SessionInfo(
                id=r["id"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                provider=r["provider"],
                model=r["model"],
                system=r["system"],
                message_count=r["n"],
                title=r["title"] or "",
            )
            for r in rows
        ]


def _role(value: str) -> Literal["user", "assistant"]:
    return "assistant" if value == "assistant" else "user"
