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
    "task": 0.7,          # User input — primary evolution signal
    "p1_task": 0.5,
    "crash_log": 0.4,     # Machine-generated — only useful for immediate next evolution
    "reflection": 0.4,    # Machine-generated — repetitive, compacts aggressively
    "observation": 0.3,
    "evolution_intent": 0.3,
}

_KEYWORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# ============= SYSTEM NOISE DETECTION =============
# Patterns for detecting system-generated noise that should be filtered out.
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    # Process termination signals (low value)
    re.compile(r"killed by signal (SIGTERM|SIGKILL|SIGINT)", re.IGNORECASE),
    re.compile(r"Reason: killed by signal", re.IGNORECASE),
    re.compile(r"Gen \d+ died after \d+s", re.IGNORECASE),
    
    # ASCII art / game output
    re.compile(r"^[·●\s]{20,}$", re.MULTILINE),  # Lines with only dots/circles
    re.compile(r"(█{3,}|▓{3,}|░{3,})", re.IGNORECASE),  # Block characters
    
    # Empty or near-empty output logs
    re.compile(r"^--- Last output ---\s*$", re.MULTILINE),
    re.compile(r"^Step \d+:\s*$", re.MULTILINE),
    
    # Heartbeat/health check spam
    re.compile(r"heartbeat.*ok|health check.*passed", re.IGNORECASE),
    
    # Generic process lifecycle events (without context)
    re.compile(r"^(started|stopped|running|idle)\s*$", re.IGNORECASE),
]

# Minimum quality threshold: entries scoring below this are rejected.
_MIN_QUALITY_SCORE = 0.15

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

    # Content quality filtering for evolution_intent entries.
    if entry_type == "evolution_intent":
        content_lower = content.lower()
        
        # Low-value patterns → reduce importance
        low_value_patterns = [
            ("optimize: survived", 0.25),  # Too generic
            ("repair: error", 0.30),        # Failed repairs
            ("repair: ⚠", 0.30),
            ("indexerror", 0.30),
            ("list index out of range", 0.30),
            ("tasks.jsonl not found", 0.25),  # Known persistent error
        ]
        
        for pattern, penalty_score in low_value_patterns:
            if pattern in content_lower:
                return penalty_score
        
        # High-value patterns → boost importance
        high_value_patterns = [
            "new feature",
            "performance improvement",
            "bug fix:",
            "optimization:",
            "refactor:",
            "implement",
        ]
        
        if any(pattern in content_lower for pattern in high_value_patterns):
            base = min(base + 0.15, 0.65)
        
        return round(base, 2)

    # Content quality filtering for crash_log entries.
    if entry_type == "crash_log":
        content_lower = content.lower()
        
        # Generic/repetitive failures → lower importance
        if "clean exit but heartbeat lost" in content_lower:
            return 0.30  # Very common, not actionable

        if "killed by signal sigterm" in content_lower:
            return 0.30  # Normal shutdown

        if "timeout" in content_lower or "timed out" in content_lower:
            return 0.35

        # Novel crashes with tracebacks → slightly higher
        novel_indicators = [
            "traceback",
            "exception:",
            "segmentation fault",
            "memory error",
            "assertion failed",
        ]

        if any(indicator in content_lower for indicator in novel_indicators):
            base = min(base + 0.10, 0.55)
        
        return round(base, 2)

    # Non-task/special types: original logic.
    if len(content) > 500:
        base = min(base + 0.05, 1.0)
    return round(base, 2)


def _is_system_noise(entry_type: str, content: str) -> bool:
    """Detect if content is system-generated noise that should be filtered out.
    
    Returns True if content matches noise patterns and should be rejected.
    
    Protection rules:
    - Never filter 'task' entries (user input is sacred)
    - Never filter 'directive' entries (high-value instructions)
    """
    # CRITICAL: Never filter user tasks or directives
    if entry_type in ("task", "directive"):
        return False
    
    # Additional heuristics for crash_log (special handling before general pattern matching)
    if entry_type == "crash_log":
        content_lower = content.lower()
        
        # Pure termination signals without context → noise
        if "killed by signal" in content_lower:
            # Check if there's ANY contextual information beyond the signal
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Count lines that provide actual context (not just noise patterns or metadata)
            contextual_lines = [
                line for line in lines 
                if not any(p.search(line) for p in _NOISE_PATTERNS)
                and not line.lower().startswith("gen ")
                and not line.lower().startswith("reason:")
                and len(line) > 10  # Exclude very short lines
            ]
            
            # If we have contextual lines OR context keywords, keep it
            has_context_keywords = any(
                keyword in content_lower 
                for keyword in ["context:", "user", "during", "while", "because", "requested"]
            )
            
            # Reject only if: short content + no context lines + no context keywords
            if len(content) < 80 and len(contextual_lines) == 0 and not has_context_keywords:
                return True
    
    if entry_type == "evolution_intent":
        content_lower = content.lower()
        
        # Generic "survived" or "error" with no detail → noise
        noise_phrases = [
            "optimize: survived",
            "repair: error",
            "repair: ⚠",
        ]
        
        if any(phrase in content_lower for phrase in noise_phrases):
            # Must have additional context to be valuable
            if len(content) < 50:  # Too short to have useful detail
                return True
    
    # General noise pattern check for other entry types
    # (crash_log and evolution_intent were handled above with special logic)
    if entry_type not in ("crash_log", "evolution_intent"):
        for pattern in _NOISE_PATTERNS:
            if pattern.search(content):
                return True
    
    return False


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
    _CREATE_TABLE = _CREATE_TABLE

    def _migrate(self, con: sqlite3.Connection) -> None:
        """Add new columns if they don't exist (idempotent)."""
        existing = {row[1] for row in con.execute("PRAGMA table_info(memory)").fetchall()}
        for col_name, col_def in _MIGRATIONS:
            if col_name not in existing:
                con.execute(f"ALTER TABLE memory ADD COLUMN {col_name} {col_def}")

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

    # Machine-generated types use a lower similarity threshold because they
    # repeat the same patterns with minor generation-number variations.
    _MACHINE_TYPES = frozenset({"reflection", "crash_log", "evolution_intent", "p1_task"})

    def _is_recent_duplicate(
        self,
        entry_type: str,
        content: str,
        lookback: int = 10,
        similarity_threshold: float = 0.85,
    ) -> bool:
        """Check if content duplicates a recent entry of the same type.

        For ``task`` entries only exact (case-insensitive) matching is used,
        because users may phrase legitimately different requests in similar
        ways.  Machine-generated types (reflection, crash_log, etc.) use a
        lower fuzzy threshold (0.70) to aggressively suppress repetition.
        """
        from difflib import SequenceMatcher

        # Machine-generated content is highly repetitive — lower threshold.
        threshold = 0.70 if entry_type in self._MACHINE_TYPES else similarity_threshold

        recent = self.get_by_type(entry_type, limit=lookback)
        content_clean = content.strip().lower()

        for entry in recent:
            entry_content = entry.get("content", "").strip().lower()

            # Exact match applies to all types.
            if content_clean == entry_content:
                return True

            # Fuzzy match only for non-task types with sufficient length.
            # Short strings (< 50 chars) produce misleadingly high similarity
            # for legitimately different entries (e.g. "entry-0" vs "entry-1").
            if entry_type != "task" and len(content_clean) >= 50:
                similarity = SequenceMatcher(
                    None, content_clean, entry_content,
                ).ratio()
                if similarity >= threshold:
                    return True

        return False

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

        Pipeline: noise filter → quality gate → dedup → insert.
        Rejected entries return -1.
        """
        # PHASE 1: Noise detection (reject early)
        if _is_system_noise(entry_type, content):
            return -1  # Rejected: system noise

        # PHASE 2: Quality scoring
        if importance is None:
            importance = _compute_importance(entry_type, content)

        # PHASE 3: Minimum quality threshold (except protected types)
        if entry_type not in ("task", "directive") and importance < _MIN_QUALITY_SCORE:
            return -1  # Rejected: below quality threshold

        # PHASE 4: Deduplication — reject near-duplicate content.
        # Tasks use exact-match only (user may phrase similar requests differently).
        # Other types use fuzzy matching.
        if self._is_recent_duplicate(entry_type, content):
            return -1  # Rejected: duplicate

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
        """Insert a memory entry with an optional embedding vector.

        Pipeline: noise filter → quality gate → dedup → insert.
        Rejected entries return -1.
        """
        # PHASE 1: Noise detection (reject early)
        if _is_system_noise(entry_type, content):
            return -1  # Rejected: system noise

        # PHASE 2: Quality scoring
        if importance is None:
            importance = _compute_importance(entry_type, content)

        # PHASE 3: Minimum quality threshold (except protected types)
        if entry_type not in ("task", "directive") and importance < _MIN_QUALITY_SCORE:
            return -1  # Rejected: below quality threshold

        # PHASE 4: Deduplication
        if self._is_recent_duplicate(entry_type, content):
            return -1  # Rejected: duplicate
        
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
        """Return the most recent non-archived entries ordered by *id* descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE tier != 'archive' "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, entry_type: str, limit: int = 10) -> list[dict]:
        """Return non-archived entries of a specific type, most recent first."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE entry_type = ? AND tier != 'archive' "
                "ORDER BY id DESC LIMIT ?",
                (entry_type, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def query(
        self,
        query: str = "",
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:
        """Query memories with optional type filtering.
        
        Args:
            query: Unused (kept for API compatibility with evolver/sentinel)
            limit: Maximum number of results
            memory_type: Filter by entry_type if provided
            
        Returns:
            List of memory entries, most recent first
        """
        if memory_type:
            return self.get_by_type(memory_type, limit=limit)
        else:
            return self.get_recent(limit=limit)

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
        """Keyword-based relevance search using SQL LIKE (excludes archive)."""
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
                f"SELECT * FROM memory WHERE tier != 'archive' AND ({where}) "
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

    def recall(
        self,
        keywords: list[str],
        query_embedding: list[float] | None = None,
        limit: int = 2,
    ) -> list[dict]:
        """Search only the archive tier for related old memories.

        Uses hybrid search logic (keyword + vector) restricted to archive.
        Returns entries with ``recalled = True`` marker, or empty list.
        """
        if not keywords and query_embedding is None:
            return []

        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory WHERE tier = 'archive' ORDER BY id DESC",
            ).fetchall()

        if not rows:
            return []

        kw_set = {kw.lower() for kw in keywords} if keywords else set()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            d = self._row_to_dict(row)
            entry_keywords = (row["keywords"] or "").lower().split()
            entry_kw_set = set(entry_keywords)

            # Keyword score.
            kw_score = 0.0
            if kw_set:
                matches = len(kw_set & entry_kw_set)
                kw_score = matches / len(kw_set)

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

            # Combined score (same weights as hybrid_search).
            if has_embedding:
                score = 0.4 * kw_score + 0.6 * vec_score
            else:
                score = kw_score

            if score > 0:
                d["search_score"] = round(score, 4)
                d["recalled"] = True
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

    def is_duplicate_content(
        self,
        entry_type: str,
        content: str,
        lookback: int = 5,
        similarity_threshold: float = 0.85,
    ) -> bool:
        """Check if content is a duplicate of recent entries (exact or fuzzy match).

        Uses SequenceMatcher for fuzzy comparison to catch near-duplicates.

        Args:
            entry_type: Type of entry to check
            content: Content to check for duplication
            lookback: Number of recent entries to check
            similarity_threshold: Minimum similarity (0-1) to consider as duplicate

        Returns:
            True if duplicate found, False otherwise
        """
        from difflib import SequenceMatcher

        recent = self.get_by_type(entry_type, limit=lookback)
        content_clean = content.strip().lower()

        for entry in recent:
            entry_content = entry.get("content", "").strip().lower()

            # Exact match
            if content_clean == entry_content:
                return True

            # Fuzzy match
            similarity = SequenceMatcher(None, content_clean, entry_content).ratio()
            if similarity >= similarity_threshold:
                return True

        return False

    def deduplicate(self) -> int:
        """Archive exact-duplicate entries (keep the newest of each group).

        Scans non-archived entries for identical (entry_type, content) pairs
        and archives all but the most recent (highest id) in each group.

        Returns the number of entries archived.
        """
        with self._connect() as con:
            # Find duplicates — keep the MAX(id) in each group.
            dupes = con.execute(
                "SELECT id FROM memory "
                "WHERE tier != 'archive' "
                "AND id NOT IN ("
                "  SELECT MAX(id) FROM memory "
                "  WHERE tier != 'archive' "
                "  GROUP BY entry_type, content"
                ")",
            ).fetchall()

            if not dupes:
                return 0

            ids = [r["id"] for r in dupes]
            placeholders = ",".join("?" * len(ids))
            con.execute(
                f"UPDATE memory SET tier = 'archive' WHERE id IN ({placeholders})",
                ids,
            )
            return len(ids)

    def compact(self, current_generation: int, curator=None) -> dict:
        """Run tiered compaction: hot→warm, warm→cold (LLM or rules), cold cleanup.

        Args:
            current_generation: The current generation number.
            curator: Optional MemoryCurator for LLM-assisted warm→cold transition.

        Returns dict with counts: hot_to_warm, warm_to_cold, deleted, llm_curated.
        """
        result = {"hot_to_warm": 0, "warm_to_cold": 0, "deleted": 0, "llm_curated": 0, "deduped": 0}

        # --- Phase 0: Deduplication ---
        result["deduped"] = self.deduplicate()

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
        threshold_gen = current_generation - 5
        with self._connect() as con:
            # Get hot entries older than 5 generations with importance < 0.6
            rows = con.execute(
                "SELECT * FROM memory WHERE tier = 'hot' "
                "AND generation <= ? AND importance < 0.6 "
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
                    # Create a compacted summary entry with archived_ids.
                    ids = [r["id"] for r in merge]
                    gen_min = min(r["generation"] for r in merge)
                    gen_max = max(r["generation"] for r in merge)
                    first_content = merge[0]["content"][:50]
                    summary = (
                        f"Compacted {len(merge)} {entry_type} entries "
                        f"from gen {gen_min}-{gen_max}: {first_content}..."
                    )
                    meta = json.dumps({"archived_ids": ids})
                    con.execute(
                        "INSERT INTO memory "
                        "(generation, entry_type, content, metadata, importance, tier, keywords) "
                        "VALUES (?, ?, ?, ?, 0.3, 'warm', ?)",
                        (current_generation, entry_type, summary, meta, _extract_keywords(summary)),
                    )
                    # Archive the merged originals instead of deleting.
                    placeholders = ",".join("?" * len(ids))
                    con.execute(
                        f"UPDATE memory SET tier = 'archive' "
                        f"WHERE id IN ({placeholders})",
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
        """Archive old, low-importance cold entries (demote to archive tier)."""
        threshold_gen = current_generation - 200
        with self._connect() as con:
            cur = con.execute(
                "UPDATE memory SET tier = 'archive', importance = importance * 0.3 "
                "WHERE tier = 'cold' AND generation <= ? AND importance < 0.3",
                (threshold_gen,),
            )
            return cur.rowcount

