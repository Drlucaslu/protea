"""Memory store backed by SQLite.

Records and queries experiential memories (reflections, observations,
directives) across generations.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER  NOT NULL,
    entry_type  TEXT     NOT NULL,
    content     TEXT     NOT NULL,
    metadata    TEXT     DEFAULT '{}',
    timestamp   TEXT     DEFAULT CURRENT_TIMESTAMP
)
"""


class MemoryStore:
    """Store and retrieve experiential memories in a local SQLite database."""

    def __init__(self, db_path: pathlib.Path) -> None:
        self.db_path = db_path
        with self._connect() as con:
            con.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse metadata JSON back to dict.
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d

    def add(
        self,
        generation: int,
        entry_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> int:
        """Insert a memory entry and return its *rowid*."""
        meta_json = json.dumps(metadata or {})
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO memory "
                "(generation, entry_type, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (generation, entry_type, content, meta_json),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Return the most recent entries ordered by *id* descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, entry_type: str, limit: int = 10) -> list[dict]:
        """Return entries of a specific type, most recent first."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE entry_type = ? "
                "ORDER BY id DESC LIMIT ?",
                (entry_type, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Return total number of memory entries."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) AS cnt FROM memory").fetchone()
            return row["cnt"]

    def clear(self) -> None:
        """Delete all memory entries."""
        with self._connect() as con:
            con.execute("DELETE FROM memory")
