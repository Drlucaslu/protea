"""Gene pool — top-N gene storage for evolutionary inheritance.

Extracts compact code summaries from Ring 2 source and stores the best
ones in SQLite.  During evolution, the top genes are injected into the
prompt so the LLM can build upon proven patterns.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import re
import sqlite3

log = logging.getLogger("protea.gene_pool")

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS gene_pool (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER NOT NULL,
    score       REAL    NOT NULL,
    source_hash TEXT    NOT NULL UNIQUE,
    gene_summary TEXT   NOT NULL,
    created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""

# Heartbeat boilerplate patterns to skip during summary extraction.
_SKIP_NAMES = frozenset({
    "heartbeat_loop", "write_heartbeat", "main", "_heartbeat_loop",
    "heartbeat_thread", "start_heartbeat", "_write_heartbeat",
})


class GenePool:
    """Top-N gene storage for evolutionary inheritance."""

    def __init__(self, db_path: pathlib.Path, max_size: int = 10) -> None:
        self.db_path = db_path
        self.max_size = max_size
        with self._connect() as con:
            con.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def add(self, generation: int, score: float, source_code: str, detail: str | None = None) -> bool:
        """Extract gene summary from source_code and store if score qualifies.

        Returns True if added (score high enough or pool not full).
        """
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()
        gene_summary = self.extract_summary(source_code)
        if not gene_summary.strip():
            return False

        with self._connect() as con:
            # Check for duplicate source.
            existing = con.execute(
                "SELECT id FROM gene_pool WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            if existing:
                return False

            count = con.execute("SELECT COUNT(*) AS cnt FROM gene_pool").fetchone()["cnt"]

            if count < self.max_size:
                con.execute(
                    "INSERT INTO gene_pool (generation, score, source_hash, gene_summary) "
                    "VALUES (?, ?, ?, ?)",
                    (generation, score, source_hash, gene_summary),
                )
                log.info("Gene pool: added gen-%d (score=%.2f, pool=%d/%d)",
                         generation, score, count + 1, self.max_size)
                return True

            # Pool is full — evict lowest if new score is higher.
            min_row = con.execute(
                "SELECT id, score FROM gene_pool ORDER BY score ASC LIMIT 1"
            ).fetchone()
            if min_row and score > min_row["score"]:
                con.execute("DELETE FROM gene_pool WHERE id = ?", (min_row["id"],))
                con.execute(
                    "INSERT INTO gene_pool (generation, score, source_hash, gene_summary) "
                    "VALUES (?, ?, ?, ?)",
                    (generation, score, source_hash, gene_summary),
                )
                log.info("Gene pool: replaced lowest (%.2f) with gen-%d (score=%.2f)",
                         min_row["score"], generation, score)
                return True

            return False

    def get_top(self, n: int = 3) -> list[dict]:
        """Return top N genes by score, each with gene_summary field."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT generation, score, gene_summary FROM gene_pool "
                "ORDER BY score DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        """Return total number of genes stored."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) AS cnt FROM gene_pool").fetchone()
            return row["cnt"]

    def backfill(self, skill_store) -> int:
        """One-time backfill from existing crystallized skills.

        Reads source_code from the skill store and populates the gene
        pool.  Only runs if the pool is currently empty.
        Returns the number of genes added.
        """
        if self.count() > 0:
            return 0

        try:
            skills = skill_store.get_active(50)
        except Exception:
            return 0

        added = 0
        for skill in skills:
            source = skill.get("source_code", "")
            if not source or len(source) < 50:
                continue
            # Use usage_count as a proxy score (normalised to 0-1).
            usage = skill.get("usage_count", 0)
            score = min(0.50 + usage * 0.05, 0.95)
            gen = skill.get("id", 0)
            if self.add(gen, score, source):
                added += 1

        if added:
            log.info("Gene pool: backfilled %d genes from skill store", added)
        return added

    @staticmethod
    def extract_summary(source_code: str) -> str:
        """Extract compact gene summary from Ring 2 source code.

        - Parse def/class signatures with regex (not AST, to handle broken code)
        - Skip heartbeat boilerplate (heartbeat_loop, write_heartbeat, main)
        - Include first-line docstrings
        - Cap at 500 chars
        """
        lines: list[str] = []

        # Match class and def definitions.
        for match in re.finditer(
            r'^(class\s+(\w+)[^\n]*|def\s+(\w+)[^\n]*)', source_code, re.MULTILINE
        ):
            full_line = match.group(0)
            name = match.group(2) or match.group(3)

            # Skip heartbeat boilerplate.
            if name and name.lower() in _SKIP_NAMES:
                continue

            lines.append(full_line)

            # Try to grab first-line docstring right after the def/class.
            end_pos = match.end()
            rest = source_code[end_pos:]
            doc_match = re.match(r'\s*\n\s*("""([^"]*?)"""|\'\'\'([^\']*?)\'\'\')', rest)
            if doc_match:
                docstring = (doc_match.group(2) or doc_match.group(3) or "").strip()
                first_line = docstring.split("\n")[0].strip()
                if first_line:
                    lines.append(f'    """{first_line}"""')

        summary = "\n".join(lines)
        if len(summary) > 500:
            summary = summary[:497] + "..."
        return summary
