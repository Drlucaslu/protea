"""Skill store backed by SQLite.

Stores prompt templates and structured descriptions for reusable skills.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import sqlite3

from ring0.sqlite_store import SQLiteStore

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


class SkillStore(SQLiteStore):
    """Store and retrieve skills in a local SQLite database."""

    _TABLE_NAME = "skills"
    _CREATE_TABLE = _CREATE_TABLE

    def _migrate(self, con: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema."""
        cols = {row[1] for row in con.execute("PRAGMA table_info(skills)")}
        if "source_code" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN source_code TEXT DEFAULT ''")
        if "last_used_at" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN last_used_at TEXT DEFAULT NULL")
        if "published" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN published BOOLEAN DEFAULT 0")
        if "dependencies" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN dependencies TEXT DEFAULT '[]'")
        if "permanent" not in cols:
            con.execute("ALTER TABLE skills ADD COLUMN permanent BOOLEAN DEFAULT 0")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("parameters", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = {} if key == "parameters" else []
        if "dependencies" in d and isinstance(d["dependencies"], str):
            try:
                d["dependencies"] = json.loads(d["dependencies"])
            except (json.JSONDecodeError, TypeError):
                d["dependencies"] = []
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
        dependencies: list[str] | None = None,
    ) -> int:
        """Insert a skill and return its rowid."""
        params_json = json.dumps(parameters or {})
        tags_json = json.dumps(tags or [])
        deps_json = json.dumps(dependencies or [])
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO skills "
                "(name, description, prompt_template, parameters, tags, source, source_code, dependencies) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, description, prompt_template, params_json, tags_json, source, source_code, deps_json),
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
        """Increment the usage count and update last_used_at for a skill.

        Evolved capability skills are auto-promoted to permanent on first use.
        """
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET usage_count = usage_count + 1, "
                "last_used_at = CURRENT_TIMESTAMP WHERE name = ?",
                (name,),
            )
            # Auto-promote evolved capabilities to permanent after first use.
            con.execute(
                "UPDATE skills SET permanent = 1 "
                "WHERE name = ? AND source = 'evolved' AND permanent = 0",
                (name,),
            )

    def deactivate(self, name: str) -> None:
        """Deactivate a skill by name."""
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET active = 0 WHERE name = ?", (name,),
            )

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
        dependencies = skill_data.get("dependencies")
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
                if dependencies is not None:
                    con.execute(
                        "UPDATE skills SET dependencies = ? WHERE name = ?",
                        (json.dumps(dependencies), name),
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
            dependencies=dependencies,
        )

    def get_unpublished(self, min_usage: int = 2) -> list[dict]:
        """Return active, locally-created skills not yet published to the Hub.

        Only returns skills with usage_count >= *min_usage* to ensure quality.
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM skills WHERE active = 1 "
                "AND source IN ('crystallized', 'user') "
                "AND published = 0 "
                "AND usage_count >= ? "
                "ORDER BY usage_count DESC",
                (min_usage,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def mark_published(self, name: str) -> None:
        """Mark a skill as published to the Hub."""
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET published = 1 WHERE name = ?", (name,),
            )

    def get_local_names(self) -> set[str]:
        """Return names of all active skills (for dedup during discovery)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT name FROM skills WHERE active = 1",
            ).fetchall()
            return {r["name"] for r in rows}

    def evict_stale(self, days: int = 30) -> int:
        """Remove hub-sourced skills unused for more than *days* days.

        Hub skills that have *never* been used (usage_count == 0) are evicted
        after 7 days instead of *days* to save space.
        Locally crystallized skills (source != 'hub') are never evicted.
        Permanent skills are never evicted regardless of source.
        Returns the number of skills removed.
        """
        with self._connect() as con:
            # Never-used hub skills: 7-day expiry.
            cur1 = con.execute(
                "DELETE FROM skills WHERE source = 'hub' "
                "AND permanent = 0 "
                "AND usage_count = 0 AND ("
                "  last_used_at IS NULL AND created_at < datetime('now', '-7 days')"
                "  OR last_used_at < datetime('now', '-7 days')"
                ")",
            )
            count = cur1.rowcount
            # Used hub skills: normal expiry.
            cur2 = con.execute(
                "DELETE FROM skills WHERE source = 'hub' "
                "AND permanent = 0 "
                "AND usage_count > 0 AND ("
                "  last_used_at IS NULL AND created_at < datetime('now', ?)"
                "  OR last_used_at < datetime('now', ?)"
                ")",
                (f"-{days} days", f"-{days} days"),
            )
            return count + cur2.rowcount

    def mark_permanent(self, name: str) -> None:
        """Mark a skill as permanent (will not be evicted)."""
        with self._connect() as con:
            con.execute(
                "UPDATE skills SET permanent = 1 WHERE name = ?", (name,),
            )

    def get_permanent(self) -> list[dict]:
        """Return all permanent capability skills."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM skills WHERE permanent = 1 AND active = 1 "
                "ORDER BY usage_count DESC",
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def cleanup_unused(self, min_age_days: int = 14) -> int:
        """Deactivate evolved skills that have never been used.

        Targets skills from evolution/crystallization that have zero usage
        after existing for at least min_age_days. Permanent and user-created
        skills are preserved.

        Returns the number of skills deactivated.
        """
        with self._connect() as con:
            cur = con.execute(
                "UPDATE skills SET active = 0 "
                "WHERE active = 1 "
                "AND permanent = 0 "
                "AND source IN ('crystallized', 'evolved') "
                "AND usage_count = 0 "
                "AND created_at < datetime('now', ?)",
                (f"-{min_age_days} days",),
            )
            return cur.rowcount

