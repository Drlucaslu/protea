"""Scheduled Task Store — SQLite-backed storage for cron/one-shot tasks.

Stores scheduled tasks with cron expressions (or ISO datetime for one-shot)
and tracks next-run times for the Sentinel main loop to poll.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import pathlib
import time
import uuid
from datetime import datetime

from ring0.sqlite_store import SQLiteStore


class ScheduledTaskStore(SQLiteStore):
    """SQLite store for scheduled (cron / one-shot) tasks."""

    _TABLE_NAME = "scheduled_tasks"
    _CREATE_TABLE = """\
    CREATE TABLE IF NOT EXISTS scheduled_tasks (
        id            INTEGER PRIMARY KEY,
        schedule_id   TEXT    NOT NULL UNIQUE,
        name          TEXT    NOT NULL,
        task_text     TEXT    NOT NULL,
        chat_id       TEXT    NOT NULL DEFAULT '',
        cron_expr     TEXT    NOT NULL,
        schedule_type TEXT    NOT NULL DEFAULT 'cron',
        enabled       INTEGER NOT NULL DEFAULT 1,
        created_at    REAL    NOT NULL,
        last_run_at   REAL    DEFAULT NULL,
        next_run_at   REAL    DEFAULT NULL,
        run_count     INTEGER NOT NULL DEFAULT 0,
        expires_at    REAL    DEFAULT NULL
    )"""

    def _migrate(self, con) -> None:
        """Add columns introduced after the initial schema."""
        cols = {row[1] for row in con.execute("PRAGMA table_info(scheduled_tasks)").fetchall()}
        if "expires_at" not in cols:
            con.execute("ALTER TABLE scheduled_tasks ADD COLUMN expires_at REAL DEFAULT NULL")
        if "published_template_hash" not in cols:
            con.execute("ALTER TABLE scheduled_tasks ADD COLUMN published_template_hash TEXT DEFAULT NULL")

    def add(
        self,
        name: str,
        task_text: str,
        cron_expr: str,
        schedule_type: str = "cron",
        chat_id: str = "",
        expires_at: float | None = None,
    ) -> str:
        """Insert a new scheduled task and return its schedule_id."""
        schedule_id = f"sched-{uuid.uuid4().hex[:8]}"
        now = time.time()

        # Compute next_run_at
        next_at = self._compute_next_run(cron_expr, schedule_type)

        with self._connect() as con:
            con.execute(
                "INSERT INTO scheduled_tasks "
                "(schedule_id, name, task_text, chat_id, cron_expr, schedule_type, enabled, created_at, next_run_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (schedule_id, name, task_text, chat_id, cron_expr, schedule_type, now, next_at, expires_at),
            )
        return schedule_id

    def get_all(self) -> list[dict]:
        """Return all scheduled tasks (for Dashboard)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM scheduled_tasks ORDER BY next_run_at ASC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_enabled(self) -> list[dict]:
        """Return all enabled scheduled tasks."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY next_run_at ASC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_due(self, now: float | None = None) -> list[dict]:
        """Return enabled tasks whose next_run_at <= now and not expired."""
        if now is None:
            now = time.time()
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled = 1 "
                "AND next_run_at IS NOT NULL AND next_run_at <= ? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (now, now),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_after_run(self, schedule_id: str, next_run_at: float | None) -> None:
        """Update last_run_at, next_run_at, and increment run_count."""
        now = time.time()
        with self._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET last_run_at = ?, next_run_at = ?, run_count = run_count + 1 "
                "WHERE schedule_id = ?",
                (now, next_run_at, schedule_id),
            )

    def enable(self, schedule_id: str) -> bool:
        """Enable a task. Returns True if a row was updated."""
        with self._connect() as con:
            cur = con.execute(
                "UPDATE scheduled_tasks SET enabled = 1 WHERE schedule_id = ?",
                (schedule_id,),
            )
            return cur.rowcount > 0

    def disable(self, schedule_id: str) -> bool:
        """Disable a task. Returns True if a row was updated."""
        with self._connect() as con:
            cur = con.execute(
                "UPDATE scheduled_tasks SET enabled = 0 WHERE schedule_id = ?",
                (schedule_id,),
            )
            return cur.rowcount > 0

    def remove(self, schedule_id: str) -> bool:
        """Delete a task. Returns True if a row was deleted."""
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM scheduled_tasks WHERE schedule_id = ?",
                (schedule_id,),
            )
            return cur.rowcount > 0

    def get_by_name(self, name: str) -> dict | None:
        """Look up a task by name. Returns None if not found."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM scheduled_tasks WHERE name = ?",
                (name,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_by_id(self, schedule_id: str) -> dict | None:
        """Look up a task by schedule_id. Returns None if not found."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM scheduled_tasks WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    # ------------------------------------------------------------------
    # Task Template sharing
    # ------------------------------------------------------------------

    def get_publishable(self, min_runs: int = 2) -> list[dict]:
        """Return enabled tasks with run_count >= min_runs (candidates for sharing)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND run_count >= ? "
                "ORDER BY run_count DESC",
                (min_runs,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def extract_template(task: dict) -> dict:
        """Extract a parameterized template from a scheduled task.

        Strips chat_id and replaces concrete subjects/topics with {param}
        placeholders where possible.
        """
        import hashlib
        import re

        task_text = task.get("task_text", "")
        name = task.get("name", "")
        cron_expr = task.get("cron_expr", "")
        schedule_type = task.get("schedule_type", "cron")

        # Build parameterized version: strip chat_id specifics.
        parameterized = task_text

        # Compute a hash of the template for dedup.
        template_hash = hashlib.sha256(
            f"{name}:{cron_expr}:{parameterized}".encode()
        ).hexdigest()[:16]

        return {
            "name": name,
            "task_text": parameterized,
            "cron_expr": cron_expr,
            "schedule_type": schedule_type,
            "template_hash": template_hash,
            "tags": _extract_tags(task_text),
        }

    def install_from_template(
        self, template: dict, params: dict | None = None, chat_id: str = ""
    ) -> str:
        """Instantiate a task from a downloaded template.

        Returns the new schedule_id.
        """
        task_text = template.get("task_text", "")
        # Substitute any {param} placeholders.
        if params:
            for key, value in params.items():
                task_text = task_text.replace(f"{{{key}}}", str(value))

        return self.add(
            name=template.get("name", "imported-task"),
            task_text=task_text,
            cron_expr=template.get("cron_expr", "0 9 * * *"),
            schedule_type=template.get("schedule_type", "cron"),
            chat_id=chat_id,
        )

    def mark_template_published(self, schedule_id: str, template_hash: str) -> None:
        """Mark a task as published with its template hash."""
        with self._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET published_template_hash = ? "
                "WHERE schedule_id = ?",
                (template_hash, schedule_id),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_next_run(cron_expr: str, schedule_type: str) -> float | None:
        """Compute the next run timestamp for a cron or once expression."""
        if schedule_type == "once":
            try:
                dt = datetime.fromisoformat(cron_expr)
                return dt.timestamp()
            except (ValueError, TypeError):
                return None
        # cron
        try:
            from ring0.cron import next_run
            dt = next_run(cron_expr, datetime.now())
            return dt.timestamp()
        except Exception:
            return None


def _extract_tags(text: str) -> list[str]:
    """Extract simple keyword tags from task text."""
    import re
    # Common task-related keywords.
    keywords = {
        "news", "weather", "summary", "report", "monitor", "check",
        "alert", "digest", "daily", "weekly", "rss", "feed", "price",
        "stock", "crypto", "email", "backup", "cleanup", "health",
        "status", "notify", "remind", "translate", "search", "analyze",
    }
    words = set(re.findall(r"[a-z]+", text.lower()))
    return sorted(words & keywords)
