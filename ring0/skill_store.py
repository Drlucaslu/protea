"""Skill store backed by SQLite.

Stores prompt templates and structured descriptions for reusable skills.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS skills (
    id              INTEGER PRIMARY KEY,
    name            TEXT     NOT NULL UNIQUE,
    description     TEXT     NOT NULL,
    prompt_template TEXT     NOT NULL,
    parameters      TEXT     DEFAULT '{}',
    tags            TEXT     DEFAULT '[]',
    source          TEXT     NOT NULL DEFAULT 'user',
    source_code     TEXT     DEFAULT '',
    usage_count     INTEGER  DEFAULT 0,
    active          BOOLEAN  DEFAULT 1,
    created_at      TEXT     DEFAULT CURRENT_TIMESTAMP
)
"""


class SkillStore:
    """Store and retrieve skills in a local SQLite database."""

    def __init__(self, db_path: pathlib.Path) -> None:
        self.db_path = db_path
        with self._connect() as con:
            con.execute(_CREATE_TABLE)
            self._migrate(con)

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema."""
        cols = {row[1] for row in con.execute("PRAGMA table_info(skills)")}
        if "source_code" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN source_code TEXT DEFAULT ''")
        if "last_used_at" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN last_used_at TEXT DEFAULT NULL")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("parameters", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = {} if key == "parameters" else []
        d["active"] = bool(d.get("active", 1))
        return d

    def add(
        self,
        name: str,
        description: str,
        prompt_template: str,
        parameters: dict | None = None,
        tags: list[str] | None = None,
        source: str = "user",
        source_code: str = "",
    ) -> int:
        """Insert a skill and return its rowid."""
        params_json = json.dumps(parameters or {})
        tags_json = json.dumps(tags or [])
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO skills "
                "(name, description, prompt_template, parameters, tags, source, source_code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, description, prompt_template, params_json, tags_json, source, source_code),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_by_name(self, name: str) -> dict | None:
        """Return a skill by name, or None if not found."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM skills WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def get_active(self, limit: int = 50) -> list[dict]:
        """Return active skills ordered by usage count descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM skills WHERE active = 1 "
                "ORDER BY usage_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_usage(self, name: str) -> None:
        """Increment the usage count and update last_used_at for a skill."""
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET usage_count = usage_count + 1, "
                "last_used_at = CURRENT_TIMESTAMP WHERE name = ?",
                (name,),
            )

    def deactivate(self, name: str) -> None:
        """Deactivate a skill by name."""
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET active = 0 WHERE name = ?", (name,),
            )

    def count(self) -> int:
        """Return total number of skills."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) AS cnt FROM skills").fetchone()
            return row["cnt"]

    def count_active(self) -> int:
        """Return number of active skills."""
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS cnt FROM skills WHERE active = 1"
            ).fetchone()
            return row["cnt"]

    def get_least_used(self, limit: int = 1) -> list[dict]:
        """Return least-used active skills, ordered by usage_count ASC, id ASC."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM skills WHERE active = 1 "
                "ORDER BY usage_count ASC, id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update(
        self,
        name: str,
        description: str | None = None,
        prompt_template: str | None = None,
        tags: list[str] | None = None,
        source_code: str | None = None,
    ) -> bool:
        """Update fields of an existing skill. Returns True if a row was updated."""
        sets: list[str] = []
        vals: list = []
        if description is not None:
            sets.append("description = ?")
            vals.append(description)
        if prompt_template is not None:
            sets.append("prompt_template = ?")
            vals.append(prompt_template)
        if tags is not None:
            sets.append("tags = ?")
            vals.append(json.dumps(tags))
        if source_code is not None:
            sets.append("source_code = ?")
            vals.append(source_code)
        if not sets:
            return False
        vals.append(name)
        with self._connect() as con:
            cur = con.execute(
                f"UPDATE skills SET {', '.join(sets)} WHERE name = ?",
                vals,
            )
            return cur.rowcount > 0

    def install_from_hub(self, skill_data: dict) -> int:
        """Install a skill downloaded from the hub into the local store.

        If a skill with the same name already exists, update it.
        Returns the rowid.
        """
        name = skill_data["name"]
        existing = self.get_by_name(name)
        if existing:
            self.update(
                name,
                description=skill_data.get("description"),
                prompt_template=skill_data.get("prompt_template"),
                tags=skill_data.get("tags"),
                source_code=skill_data.get("source_code"),
            )
            with self._connect() as con:
                con.execute(
                    "UPDATE skills SET source = 'hub', active = 1 WHERE name = ?",
                    (name,),
                )
            return existing["id"]
        return self.add(
            name=name,
            description=skill_data.get("description", ""),
            prompt_template=skill_data.get("prompt_template", ""),
            parameters=skill_data.get("parameters"),
            tags=skill_data.get("tags"),
            source="hub",
            source_code=skill_data.get("source_code", ""),
        )

    def evict_stale(self, days: int = 30) -> int:
        """Remove hub-sourced skills unused for more than *days* days.

        Locally crystallized skills (source != 'hub') are never evicted.
        Returns the number of skills removed.
        """
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM skills WHERE source = 'hub' AND ("
                "  last_used_at IS NULL AND created_at < datetime('now', ?)"
                "  OR last_used_at < datetime('now', ?)"
                ")",
                (f"-{days} days", f"-{days} days"),
            )
            return cur.rowcount

    def clear(self) -> None:
        """Delete all skills."""
        with self._connect() as con:
            con.execute("DELETE FROM skills")
