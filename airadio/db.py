"""SQLite access layer.

Several processes touch this database (brain loop, Discord bot, CLI), so every
connection runs in WAL mode with a generous busy timeout.  Writes are short.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL UNIQUE,
    dedupe_key    TEXT    NOT NULL,
    content_hash  TEXT,
    title         TEXT    NOT NULL,
    artist        TEXT    NOT NULL,
    album         TEXT,
    genre         TEXT,
    tags          TEXT,
    mood          TEXT,
    energy        INTEGER,
    duration      REAL,
    source_url    TEXT,
    added_at      REAL    NOT NULL,
    play_count    INTEGER NOT NULL DEFAULT 0,
    last_played_at REAL,
    missing       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tracks_dedupe   ON tracks(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_tracks_content  ON tracks(content_hash);
CREATE INDEX IF NOT EXISTS idx_tracks_played   ON tracks(last_played_at);
CREATE INDEX IF NOT EXISTS idx_tracks_energy   ON tracks(energy);

CREATE TABLE IF NOT EXISTS queue_items (
    id         INTEGER PRIMARY KEY,
    block_id   INTEGER,
    seq        INTEGER NOT NULL,
    kind       TEXT    NOT NULL,           -- 'song' | 'patter'
    tier       TEXT    NOT NULL DEFAULT 'ai',  -- 'ai' | 'request'
    path       TEXT,
    track_id   INTEGER,
    text       TEXT,
    title      TEXT,
    artist     TEXT,
    duration   REAL,
    status     TEXT    NOT NULL DEFAULT 'pending',  -- pending|ready|pushed|done|failed
    created_at REAL    NOT NULL,
    pushed_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(status, seq);

CREATE TABLE IF NOT EXISTS blocks (
    id         INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    hour       INTEGER,
    mood_name  TEXT,
    mood_note  TEXT,
    source     TEXT,                        -- 'llm' | 'fallback'
    item_count INTEGER
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS chat_requests (
    id         INTEGER PRIMARY KEY,
    user       TEXT,
    text       TEXT NOT NULL,
    kind       TEXT,                        -- track|vibe|question|unknown
    payload    TEXT,
    status     TEXT NOT NULL DEFAULT 'new', -- new|working|done|failed
    note       TEXT,
    created_at REAL NOT NULL,
    handled_at REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_status ON chat_requests(status, id);

CREATE TABLE IF NOT EXISTS candidates (
    id         INTEGER PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    artist     TEXT NOT NULL,
    title      TEXT NOT NULL,
    source     TEXT,
    status     TEXT NOT NULL DEFAULT 'new', -- new|downloaded|rejected|failed
    attempts   INTEGER NOT NULL DEFAULT 0,
    note       TEXT,
    created_at REAL NOT NULL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);

CREATE TABLE IF NOT EXISTS download_log (
    id          INTEGER PRIMARY KEY,
    artist      TEXT,
    title       TEXT,
    video_id    TEXT,
    video_title TEXT,
    channel     TEXT,
    duration    REAL,
    decision    TEXT,                       -- accepted|rejected|error
    reason      TEXT,
    created_at  REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def init(self) -> None:
        conn = self.connect()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    # -- convenience --------------------------------------------------------

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, tuple(params)).fetchall()
        finally:
            conn.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.lastrowid or cur.rowcount

    # -- shared key/value state --------------------------------------------

    def get_state(self, key: str, default: str = "") -> str:
        row = self.one("SELECT value FROM state WHERE key = ?", (key,))
        return row["value"] if row and row["value"] is not None else default

    def set_state(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO state(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, time.time()),
        )
