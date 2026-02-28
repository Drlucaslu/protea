"""Output Queue — SQLite-backed store for evolution outputs awaiting user review.

Captures new capabilities after evolution survives, delivers them to the user
via Telegram, and records feedback (accepted/rejected/scheduled) to guide
future evolution.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import logging
import pathlib
import time

from ring0.sqlite_store import SQLiteStore

log = logging.getLogger("protea.output_queue")


class OutputQueue(SQLiteStore):
    """SQLite store for evolution output items pending user review."""

    _TABLE_NAME = "output_queue"
    _CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS output_queue (
        id            INTEGER PRIMARY KEY,
        gene_id       INTEGER,
        generation    INTEGER NOT NULL,
        capability    TEXT NOT NULL,
        summary       TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'pending',
        created_at    REAL NOT NULL,
        delivered_at  REAL,
        feedback_at   REAL,
        feedback_text TEXT,
        telegram_msg_id INTEGER
    )"""

    def add(self, gene_id: int | None, generation: int, capability: str, summary: str) -> int:
        """Enqueue a new evolution output. Returns the row id."""
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO output_queue "
                "(gene_id, generation, capability, summary, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (gene_id, generation, capability, summary, now),
            )
            return cur.lastrowid

    def get_pending(self, limit: int = 5) -> list[dict]:
        """Get items waiting for delivery."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM output_queue WHERE status = 'pending' "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_by_id(self, item_id: int) -> dict | None:
        """Look up an item by its id."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM output_queue WHERE id = ?",
                (item_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_by_telegram_msg_id(self, msg_id: int) -> dict | None:
        """Look up an item by the Telegram message id."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM output_queue WHERE telegram_msg_id = ?",
                (msg_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def mark_delivered(self, item_id: int, telegram_msg_id: int | None = None) -> None:
        """Mark an item as delivered to the user."""
        now = time.time()
        with self._connect() as con:
            con.execute(
                "UPDATE output_queue SET status = 'delivered', delivered_at = ?, "
                "telegram_msg_id = ? WHERE id = ?",
                (now, telegram_msg_id, item_id),
            )

    def mark_feedback(self, item_id: int, status: str, feedback_text: str = "") -> None:
        """Record user feedback on an item (accepted/rejected/scheduled)."""
        now = time.time()
        with self._connect() as con:
            con.execute(
                "UPDATE output_queue SET status = ?, feedback_at = ?, "
                "feedback_text = ? WHERE id = ?",
                (status, now, feedback_text, item_id),
            )

    def expire_old(self, max_age_hours: int = 24) -> int:
        """Expire unresponded items older than max_age_hours. Returns count."""
        cutoff = time.time() - max_age_hours * 3600
        with self._connect() as con:
            cur = con.execute(
                "UPDATE output_queue SET status = 'expired' "
                "WHERE status IN ('pending', 'delivered') AND created_at < ?",
                (cutoff,),
            )
            return cur.rowcount

    def get_accepted(self) -> list[dict]:
        """Return all accepted/scheduled capabilities (for evolution constraints)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM output_queue WHERE status IN ('accepted', 'scheduled') "
                "ORDER BY feedback_at DESC",
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_rejected(self) -> list[dict]:
        """Return all rejected capabilities (for evolution constraints)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM output_queue WHERE status = 'rejected' "
                "ORDER BY feedback_at DESC",
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def daily_push_count(self) -> int:
        """Count items created in the last 24 hours (for rate limiting)."""
        cutoff = time.time() - 86400
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS cnt FROM output_queue WHERE created_at > ?",
                (cutoff,),
            ).fetchone()
            return row["cnt"]
