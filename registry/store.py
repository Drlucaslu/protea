"""Registry store backed by SQLite.

Multi-node skill registry — stores skills published by different Protea
instances for discovery and sharing.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS registry_skills (
    id              INTEGER PRIMARY KEY,
    node_id         TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    description     TEXT    NOT NULL,
    prompt_template TEXT    NOT NULL,
    parameters      TEXT    DEFAULT '{}',
    tags            TEXT    DEFAULT '[]',
    source_code     TEXT    DEFAULT '',
    downloads       INTEGER DEFAULT 0,
    rating_up       INTEGER DEFAULT 0,
    rating_down     INTEGER DEFAULT 0,
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, name)
)
"""


class RegistryStore:
    """Store and retrieve skills in a multi-node registry database."""

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
        for key in ("parameters", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key] = {} if key == "parameters" else []
        return d

    def publish(
        self,
        node_id: str,
        name: str,
        description: str,
        prompt_template: str,
        parameters: dict | None = None,
        tags: list[str] | None = None,
        source_code: str = "",
    ) -> int:
        """Publish or update a skill.  Auto-bumps version on update."""
        params_json = json.dumps(parameters or {})
        tags_json = json.dumps(tags or [])
        with self._connect() as con:
            existing = con.execute(
                "SELECT version FROM registry_skills WHERE node_id = ? AND name = ?",
                (node_id, name),
            ).fetchone()
            if existing:
                new_version = existing["version"] + 1
                con.execute(
                    "UPDATE registry_skills SET "
                    "version = ?, description = ?, prompt_template = ?, "
                    "parameters = ?, tags = ?, source_code = ?, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE node_id = ? AND name = ?",
                    (new_version, description, prompt_template,
                     params_json, tags_json, source_code,
                     node_id, name),
                )
                row = con.execute(
                    "SELECT id FROM registry_skills WHERE node_id = ? AND name = ?",
                    (node_id, name),
                ).fetchone()
                return row["id"]
            else:
                cur = con.execute(
                    "INSERT INTO registry_skills "
                    "(node_id, name, description, prompt_template, "
                    "parameters, tags, source_code) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (node_id, name, description, prompt_template,
                     params_json, tags_json, source_code),
                )
                return cur.lastrowid  # type: ignore[return-value]

    def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search skills by name/description keyword and/or tags."""
        with self._connect() as con:
            if query and tags:
                pattern = f"%{query}%"
                rows = con.execute(
                    "SELECT * FROM registry_skills "
                    "WHERE (name LIKE ? OR description LIKE ?) "
                    "ORDER BY downloads DESC LIMIT ?",
                    (pattern, pattern, limit),
                ).fetchall()
                # Filter by tags in Python (JSON array in TEXT column).
                result = []
                for r in rows:
                    d = self._row_to_dict(r)
                    if any(t in d.get("tags", []) for t in tags):
                        result.append(d)
                return result
            elif query:
                pattern = f"%{query}%"
                rows = con.execute(
                    "SELECT * FROM registry_skills "
                    "WHERE name LIKE ? OR description LIKE ? "
                    "ORDER BY downloads DESC LIMIT ?",
                    (pattern, pattern, limit),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
            elif tags:
                rows = con.execute(
                    "SELECT * FROM registry_skills "
                    "ORDER BY downloads DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                result = []
                for r in rows:
                    d = self._row_to_dict(r)
                    if any(t in d.get("tags", []) for t in tags):
                        result.append(d)
                return result
            else:
                rows = con.execute(
                    "SELECT * FROM registry_skills "
                    "ORDER BY downloads DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]

    def get(self, node_id: str, name: str) -> dict | None:
        """Return a skill by (node_id, name) or None."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM registry_skills WHERE node_id = ? AND name = ?",
                (node_id, name),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def get_all(self, limit: int = 100) -> list[dict]:
        """Return all skills ordered by downloads descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM registry_skills ORDER BY downloads DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def increment_downloads(self, node_id: str, name: str) -> None:
        """Increment the download counter for a skill."""
        with self._connect() as con:
            con.execute(
                "UPDATE registry_skills SET downloads = downloads + 1 "
                "WHERE node_id = ? AND name = ?",
                (node_id, name),
            )

    def rate(self, node_id: str, name: str, up: bool) -> None:
        """Increment rating_up or rating_down for a skill."""
        col = "rating_up" if up else "rating_down"
        with self._connect() as con:
            con.execute(
                f"UPDATE registry_skills SET {col} = {col} + 1 "
                "WHERE node_id = ? AND name = ?",
                (node_id, name),
            )

    def delete(self, node_id: str, name: str) -> bool:
        """Delete a skill.  Returns True if a row was deleted."""
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM registry_skills WHERE node_id = ? AND name = ?",
                (node_id, name),
            )
            return cur.rowcount > 0

    def stats(self) -> dict:
        """Return registry statistics."""
        with self._connect() as con:
            total = con.execute(
                "SELECT COUNT(*) AS cnt FROM registry_skills"
            ).fetchone()["cnt"]
            nodes = con.execute(
                "SELECT COUNT(DISTINCT node_id) AS cnt FROM registry_skills"
            ).fetchone()["cnt"]
            dl = con.execute(
                "SELECT COALESCE(SUM(downloads), 0) AS cnt FROM registry_skills"
            ).fetchone()["cnt"]
            return {
                "total_skills": total,
                "total_nodes": nodes,
                "total_downloads": dl,
            }
