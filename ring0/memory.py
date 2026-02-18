"""Memory store backed by SQLite.

Records and queries experiential memories (reflections, observations,
directives) across generations.  Supports tiered storage (hot/warm/cold),
importance scoring, keyword search, and optional embedding-based semantic
search.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3

from ring0.sqlite_store import SQLiteStore


# New columns added via migration (idempotent ALTER).
_MIGRATIONS = [
    ("importance", "REAL DEFAULT 0.5"),
    ("tier", "TEXT DEFAULT 'hot'"),
    ("keywords", "TEXT DEFAULT ''"),
    ("embedding", "TEXT DEFAULT ''"),
]

# Importance base scores by entry type.
_IMPORTANCE_BASE: dict[str, float] = {
    "directive": 0.9,
    "crash_log": 0.8,
    "task": 0.7,
    "reflection": 0.6,
    "p1_task": 0.5,
    "observation": 0.5,
    "evolution_intent": 0.4,
}

_KEYWORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# Patterns for detecting low-value follow-up messages (operational commands).
# Chinese + English operational verbs / short commands.
_FOLLOWUP_PATTERNS: list[re.Pattern[str]] = [
    # Chinese operational phrases
    re.compile(r"^.{0,5}(发给我|发过来|给我看|看一下|看看|调一下|改一下|删一下|传给我)", re.IGNORECASE),
    re.compile(r"^.{0,5}(前面一段|后面一段|上面的|下面的)", re.IGNORECASE),
    re.compile(r"^.{0,5}(记到|记一下|也记|把.{0,10}记)", re.IGNORECASE),
    # English operational phrases
    re.compile(r"^.{0,10}(commit|push|pull|send|show me|let me see)", re.IGNORECASE),
    re.compile(r"^(yes|no|ok|好的|是的|可以|对|不用|不要|算了)\s*$", re.IGNORECASE),
]

# Patterns for detecting high-value substantive requests.
_SUBSTANTIVE_PATTERNS: list[re.Pattern[str]] = [
    # Chinese substantive phrases
    re.compile(r"(帮我|请|实现|分析|研究|设计|创建|开发|构建|优化|调查|对比|解释一下)", re.IGNORECASE),
    re.compile(r"(为什么|怎么|如何|能不能|是不是|有没有).{5,}", re.IGNORECASE),
    # English substantive phrases
    re.compile(r"(implement|build|create|design|analyze|investigate|explain|compare|develop|add .+ feature)", re.IGNORECASE),
    re.compile(r"(why|how|what if|can you|could you).{10,}", re.IGNORECASE),
]


def _compute_importance(entry_type: str, content: str) -> float:
    """Compute importance score for a memory entry.

    For 'task' entries, applies content-based analysis:
    - Short operational follow-ups (e.g. "发给我", "commit this") → 0.3
    - Yes/no confirmations → 0.2
    - Substantive requests with detail → 0.7-0.8
    """
    base = _IMPORTANCE_BASE.get(entry_type, 0.5)

    if entry_type == "task":
        text = content.strip()
        text_len = len(text)

        # Very short confirmations (< 10 chars): minimal value.
        if text_len < 10:
            return 0.2

        # Check for follow-up / operational patterns.
        for pat in _FOLLOWUP_PATTERNS:
            if pat.search(text):
                return 0.35 if text_len > 30 else 0.25

        # Short messages (< 25 chars) without substantive keywords.
        if text_len < 25:
            is_substantive = any(p.search(text) for p in _SUBSTANTIVE_PATTERNS)
            if not is_substantive:
                return 0.4

        # Substantive requests get a boost.
        if any(p.search(text) for p in _SUBSTANTIVE_PATTERNS):
            base = 0.75 if text_len > 100 else 0.7

        # Long detailed messages get extra credit.
        if text_len > 500:
            base = min(base + 0.05, 1.0)

        return round(base, 2)

    # Non-task types: original logic.
    if len(content) > 500:
        base = min(base + 0.05, 1.0)
    return round(base, 2)


def _extract_keywords(content: str) -> str:
    """Extract searchable keywords from content."""
    tokens = _KEYWORD_RE.findall(content.lower())
    # Keep unique tokens of length >= 3, up to 50 keywords.
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            keywords.append(t)
            if len(keywords) >= 50:
                break
    return " ".join(keywords)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure stdlib)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryStore(SQLiteStore):
    """Store and retrieve experiential memories in a local SQLite database."""

    _TABLE_NAME = "memory"
    _CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER  NOT NULL,
    entry_type  TEXT     NOT NULL,
    content     TEXT     NOT NULL,
    metadata    TEXT     DEFAULT '{}',
    importance  REAL     DEFAULT 0.5,
    tier        TEXT     DEFAULT 'hot',
    keywords    TEXT     DEFAULT '',
    embedding   TEXT     DEFAULT '',
    timestamp   TEXT     DEFAULT CURRENT_TIMESTAMP
)
"""

    def _migrate(self, con: sqlite3.Connection) -> None:
        """Add new columns if they don't exist (idempotent)."""
        existing = {row[1] for row in con.execute("PRAGMA table_info(memory)").fetchall()}
        for col_name, col_def in _MIGRATIONS:
            if col_name not in existing:
                con.execute(f"ALTER TABLE memory ADD COLUMN {col_name} {col_def}")

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        # Parse metadata JSON back to dict.
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d

    # Session gap threshold: 30 minutes of silence = new session.
    _SESSION_GAP_SEC = 30 * 60

    def _is_new_session(self, con: sqlite3.Connection) -> bool:
        """Check if this task starts a new session (> 30 min since last task)."""
        row = con.execute(
            "SELECT timestamp FROM memory WHERE entry_type = 'task' "
            "ORDER BY id DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return True
        import datetime
        try:
            last_ts = datetime.datetime.fromisoformat(row["timestamp"])
            now = datetime.datetime.now(datetime.timezone.utc)
            # Handle naive timestamps (SQLite CURRENT_TIMESTAMP is UTC but naive).
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=datetime.timezone.utc)
            return (now - last_ts).total_seconds() > self._SESSION_GAP_SEC
        except (ValueError, TypeError):
            return False

    def _apply_session_boost(
        self, con: sqlite3.Connection, entry_type: str, importance: float,
    ) -> float:
        """Boost importance for the first task in a new session."""
        if entry_type == "task" and self._is_new_session(con):
            importance = min(importance + 0.1, 1.0)
        return round(importance, 2)

    def add(
        self,
        generation: int,
        entry_type: str,
        content: str,
        metadata: dict | None = None,
        importance: float | None = None,
    ) -> int:
        """Insert a memory entry and return its *rowid*.

        Automatically computes importance and extracts keywords if not provided.
        For task entries: content-based scoring + session-start boost.
        """
        if importance is None:
            importance = _compute_importance(entry_type, content)
        keywords = _extract_keywords(content)
        meta_json = json.dumps(metadata or {})
        with self._connect() as con:
            importance = self._apply_session_boost(con, entry_type, importance)
            cur = con.execute(
                "INSERT INTO memory "
                "(generation, entry_type, content, metadata, importance, tier, keywords) "
                "VALUES (?, ?, ?, ?, ?, 'hot', ?)",
                (generation, entry_type, content, meta_json, importance, keywords),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def add_with_embedding(
        self,
        generation: int,
        entry_type: str,
        content: str,
        metadata: dict | None = None,
        importance: float | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert a memory entry with an optional embedding vector."""
        if importance is None:
            importance = _compute_importance(entry_type, content)
        keywords = _extract_keywords(content)
        meta_json = json.dumps(metadata or {})
        emb_json = json.dumps(embedding) if embedding else ""
        with self._connect() as con:
            importance = self._apply_session_boost(con, entry_type, importance)
            cur = con.execute(
                "INSERT INTO memory "
                "(generation, entry_type, content, metadata, importance, tier, keywords, embedding) "
                "VALUES (?, ?, ?, ?, ?, 'hot', ?, ?)",
                (generation, entry_type, content, meta_json, importance, keywords, emb_json),
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

    def get_by_tier(self, tier: str, limit: int = 50) -> list[dict]:
        """Return entries in a specific tier, most recent first."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE tier = ? "
                "ORDER BY id DESC LIMIT ?",
                (tier, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_relevant(self, keywords: list[str], limit: int = 5) -> list[dict]:
        """Keyword-based relevance search using SQL LIKE."""
        if not keywords:
            return []
        # Build OR clause for each keyword.
        clauses = []
        params: list[str] = []
        for kw in keywords:
            clauses.append("keywords LIKE ?")
            params.append(f"%{kw.lower()}%")
        where = " OR ".join(clauses)
        params.append(str(limit))
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM memory WHERE {where} "
                f"ORDER BY importance DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """Vector similarity search across all entries with embeddings."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE embedding != ''",
            ).fetchall()

        results: list[tuple[float, dict]] = []
        for row in rows:
            d = self._row_to_dict(row)
            try:
                emb = json.loads(row["embedding"])
            except (json.JSONDecodeError, TypeError):
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                d["similarity"] = round(sim, 4)
                results.append((sim, d))

        results.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in results[:limit]]

    def hybrid_search(
        self,
        keywords: list[str],
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Mixed search: keywords + vector similarity.

        Score = 0.4 * keyword_score + 0.6 * vector_score (when embedding available).
        Falls back to pure keyword search when no embedding is provided.
        """
        if not keywords and query_embedding is None:
            return []

        with self._connect() as con:
            rows = con.execute("SELECT * FROM memory ORDER BY id DESC").fetchall()

        kw_set = {kw.lower() for kw in keywords} if keywords else set()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            d = self._row_to_dict(row)
            entry_keywords = (row["keywords"] or "").lower().split()
            entry_kw_set = set(entry_keywords)

            # Keyword score: fraction of query keywords found in entry.
            kw_score = 0.0
            if kw_set:
                matches = len(kw_set & entry_kw_set)
                kw_score = matches / len(kw_set) if kw_set else 0.0

            # Vector score.
            vec_score = 0.0
            has_embedding = False
            if query_embedding and row["embedding"]:
                try:
                    emb = json.loads(row["embedding"])
                    vec_score = _cosine_similarity(query_embedding, emb)
                    has_embedding = True
                except (json.JSONDecodeError, TypeError):
                    pass

            # Combined score.
            if has_embedding:
                score = 0.4 * kw_score + 0.6 * vec_score
            else:
                score = kw_score

            if score > 0:
                d["search_score"] = round(score, 4)
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def get_stats(self) -> dict:
        """Return memory statistics by tier and type."""
        with self._connect() as con:
            # Count by tier.
            tier_rows = con.execute(
                "SELECT tier, COUNT(*) as cnt FROM memory GROUP BY tier",
            ).fetchall()
            by_tier = {r["tier"]: r["cnt"] for r in tier_rows}

            # Count by type.
            type_rows = con.execute(
                "SELECT entry_type, COUNT(*) as cnt FROM memory GROUP BY entry_type",
            ).fetchall()
            by_type = {r["entry_type"]: r["cnt"] for r in type_rows}

            # Total.
            total_row = con.execute("SELECT COUNT(*) as cnt FROM memory").fetchone()

            return {
                "total": total_row["cnt"],
                "by_tier": by_tier,
                "by_type": by_type,
            }

    def compact(self, current_generation: int, curator=None) -> dict:
        """Run tiered compaction: hot→warm, warm→cold (LLM or rules), cold cleanup.

        Args:
            current_generation: The current generation number.
            curator: Optional MemoryCurator for LLM-assisted warm→cold transition.

        Returns dict with counts: hot_to_warm, warm_to_cold, deleted, llm_curated.
        """
        result = {"hot_to_warm": 0, "warm_to_cold": 0, "deleted": 0, "llm_curated": 0}

        # --- Phase 1: Hot → Warm (rule-driven) ---
        result["hot_to_warm"] = self._compact_hot_to_warm(current_generation)

        # --- Phase 2: Warm → Cold (LLM-assisted or rule-driven) ---
        warm_candidates = self._get_warm_candidates(current_generation)
        if curator and warm_candidates:
            try:
                decisions = curator.curate(warm_candidates)
                if decisions:
                    self._apply_curation(decisions)
                    result["llm_curated"] = len(decisions)
                    result["warm_to_cold"] = len(decisions)
                else:
                    result["warm_to_cold"] = self._rule_based_warm_to_cold(warm_candidates)
            except Exception:
                result["warm_to_cold"] = self._rule_based_warm_to_cold(warm_candidates)
        elif warm_candidates:
            result["warm_to_cold"] = self._rule_based_warm_to_cold(warm_candidates)

        # --- Phase 3: Cold cleanup (rule-driven, selective forgetting) ---
        result["deleted"] = self._cleanup_cold(current_generation)

        return result

    def _compact_hot_to_warm(self, current_generation: int) -> int:
        """Demote old, low-importance hot entries to warm tier."""
        threshold_gen = current_generation - 10
        with self._connect() as con:
            # Get hot entries older than 10 generations with importance < 0.7
            rows = con.execute(
                "SELECT * FROM memory WHERE tier = 'hot' "
                "AND generation <= ? AND importance < 0.7 "
                "ORDER BY entry_type, importance DESC",
                (threshold_gen,),
            ).fetchall()

            if not rows:
                return 0

            # Group by entry_type, keep top 3 per group, merge rest.
            groups: dict[str, list[sqlite3.Row]] = {}
            for r in rows:
                groups.setdefault(r["entry_type"], []).append(r)

            demoted = 0
            for entry_type, entries in groups.items():
                # Keep top 3 by importance (just demote them).
                keep = entries[:3]
                merge = entries[3:]

                for row in keep:
                    con.execute(
                        "UPDATE memory SET tier = 'warm' WHERE id = ?",
                        (row["id"],),
                    )
                    demoted += 1

                if merge:
                    # Create a compacted summary entry.
                    ids = [r["id"] for r in merge]
                    gen_min = min(r["generation"] for r in merge)
                    gen_max = max(r["generation"] for r in merge)
                    first_content = merge[0]["content"][:50]
                    summary = (
                        f"Compacted {len(merge)} {entry_type} entries "
                        f"from gen {gen_min}-{gen_max}: {first_content}..."
                    )
                    con.execute(
                        "INSERT INTO memory "
                        "(generation, entry_type, content, metadata, importance, tier, keywords) "
                        "VALUES (?, ?, ?, '{}', 0.3, 'warm', ?)",
                        (current_generation, entry_type, summary, _extract_keywords(summary)),
                    )
                    # Delete the merged originals.
                    placeholders = ",".join("?" * len(ids))
                    con.execute(
                        f"DELETE FROM memory WHERE id IN ({placeholders})",
                        ids,
                    )
                    demoted += len(merge)

            return demoted

    def _get_warm_candidates(self, current_generation: int, limit: int = 20) -> list[dict]:
        """Get warm-tier entries eligible for cold transition."""
        threshold_gen = current_generation - 30
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, entry_type, content, importance FROM memory "
                "WHERE tier = 'warm' AND generation <= ? "
                "ORDER BY importance ASC LIMIT ?",
                (threshold_gen, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def _apply_curation(self, decisions: list[dict]) -> None:
        """Apply LLM curation decisions to memory entries."""
        with self._connect() as con:
            for d in decisions:
                entry_id = d.get("id")
                action = d.get("action", "keep")
                if action == "discard":
                    con.execute("DELETE FROM memory WHERE id = ?", (entry_id,))
                elif action == "summarize":
                    summary = d.get("summary", "")
                    if summary:
                        con.execute(
                            "UPDATE memory SET content = ?, tier = 'cold' WHERE id = ?",
                            (summary, entry_id),
                        )
                    else:
                        con.execute(
                            "UPDATE memory SET tier = 'cold' WHERE id = ?",
                            (entry_id,),
                        )
                else:  # keep
                    con.execute(
                        "UPDATE memory SET tier = 'cold' WHERE id = ?",
                        (entry_id,),
                    )

    def _rule_based_warm_to_cold(self, candidates: list[dict]) -> int:
        """Fallback rule-based warm→cold transition."""
        with self._connect() as con:
            count = 0
            for c in candidates:
                if c["importance"] >= 0.6:
                    con.execute(
                        "UPDATE memory SET tier = 'cold' WHERE id = ?",
                        (c["id"],),
                    )
                else:
                    # Low importance warm entries get demoted with reduced importance.
                    con.execute(
                        "UPDATE memory SET tier = 'cold', importance = importance * 0.5 WHERE id = ?",
                        (c["id"],),
                    )
                count += 1
            return count

    def _cleanup_cold(self, current_generation: int) -> int:
        """Delete old, low-importance cold entries (selective forgetting)."""
        threshold_gen = current_generation - 200
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM memory WHERE tier = 'cold' "
                "AND generation <= ? AND importance < 0.3",
                (threshold_gen,),
            )
            return cur.rowcount


