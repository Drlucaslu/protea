"""User profiler backed by SQLite.

Extracts topics from task history via keyword matching against predefined
categories.  Tracks topic weights with decay, producing a compact profile
summary that can be injected into evolution prompts.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3

_CREATE_TOPICS_TABLE = """\
CREATE TABLE IF NOT EXISTS user_profile_topics (
    id         INTEGER PRIMARY KEY,
    topic      TEXT UNIQUE,
    weight     REAL    DEFAULT 1.0,
    category   TEXT    DEFAULT 'general',
    first_seen TEXT    DEFAULT CURRENT_TIMESTAMP,
    last_seen  TEXT    DEFAULT CURRENT_TIMESTAMP,
    hit_count  INTEGER DEFAULT 1
)
"""

_CREATE_STATS_TABLE = """\
CREATE TABLE IF NOT EXISTS user_profile_stats (
    id         INTEGER PRIMARY KEY,
    stat_key   TEXT UNIQUE,
    stat_value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

# ---------------------------------------------------------------------------
# Category keyword mapping
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "coding": {
        "python", "javascript", "code", "debug", "function", "class", "api",
        "git", "bug", "refactor", "compile", "syntax", "variable", "loop",
        "algorithm",
    },
    "math": {
        "equation", "calculate", "matrix", "statistics", "probability",
        "algebra", "geometry", "integral", "derivative", "theorem",
        "formula", "vector", "linear", "optimization", "numerical",
    },
    "data": {
        "csv", "json", "database", "sql", "pandas", "analysis",
        "visualization", "dataset", "dataframe", "query", "table",
        "schema", "etl", "pipeline", "warehouse",
    },
    "web": {
        "http", "html", "css", "scrape", "fetch", "url", "browser",
        "request", "endpoint", "websocket", "server", "client",
        "rest", "graphql", "cors",
    },
    "ai": {
        "model", "llm", "neural", "machine", "learning", "transformer",
        "embedding", "training", "inference", "prompt", "gpt", "claude",
        "chatbot", "nlp", "classification",
    },
    "system": {
        "file", "process", "shell", "command", "docker", "deploy",
        "server", "linux", "terminal", "daemon", "cron", "systemd",
        "container", "kubernetes", "nginx",
    },
    "creative": {
        "write", "story", "poem", "generate", "image", "music", "design",
        "art", "creative", "narrative", "compose", "sketch", "animation",
        "illustration", "game",
    },
    "finance": {
        "stock", "market", "trade", "price", "portfolio", "crypto",
        "investment", "dividend", "forex", "bond", "revenue", "profit",
        "accounting", "budget", "tax",
    },
    "research": {
        "paper", "study", "survey", "literature", "academic", "journal",
        "citation", "hypothesis", "experiment", "methodology", "thesis",
        "review", "abstract", "conclusion", "reference",
    },
}

# Inverted index: keyword → category
_KEYWORD_TO_CATEGORY: dict[str, str] = {}
for _cat, _kws in CATEGORY_KEYWORDS.items():
    for _kw in _kws:
        _KEYWORD_TO_CATEGORY[_kw] = _cat

# Simple stop words (English)
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "and", "or",
    "but", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "just", "also", "now", "then", "here",
    "there", "when", "where", "why", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "my", "your", "his",
    "her", "our", "their", "me", "him", "her", "us", "them", "i", "you",
    "he", "she", "we", "they", "if", "up", "out", "get", "got", "let",
    "make", "like", "use", "help", "want", "need", "please", "thanks",
}

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, filtering stop words and short tokens."""
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS]


def _extract_bigrams(tokens: list[str]) -> list[str]:
    """Extract underscore-joined bigrams from adjacent tokens."""
    bigrams = []
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i + 1]}"
        # Only keep if both parts are meaningful
        if len(tokens[i]) >= 3 and len(tokens[i + 1]) >= 3:
            bigrams.append(bigram)
    return bigrams


class UserProfiler:
    """Extract and maintain a user interest profile from task history."""

    def __init__(self, db_path: pathlib.Path) -> None:
        self.db_path = db_path
        with self._connect() as con:
            con.execute(_CREATE_TOPICS_TABLE)
            con.execute(_CREATE_STATS_TABLE)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def update_from_task(self, task_text: str, response_summary: str = "") -> None:
        """Extract topics from a completed task and update the profile."""
        combined = f"{task_text} {response_summary}"
        tokens = _tokenize(combined)
        bigrams = _extract_bigrams(tokens)

        # Match tokens to categories
        matched_topics: list[tuple[str, str]] = []  # (topic, category)
        for token in tokens:
            if token in _KEYWORD_TO_CATEGORY:
                matched_topics.append((token, _KEYWORD_TO_CATEGORY[token]))

        # Check bigrams against known patterns
        for bigram in bigrams:
            parts = bigram.split("_")
            for part in parts:
                if part in _KEYWORD_TO_CATEGORY:
                    matched_topics.append((bigram, _KEYWORD_TO_CATEGORY[part]))
                    break

        # Unmatched tokens with length >= 4 go to 'general'
        matched_words = {t for t, _ in matched_topics}
        for token in tokens:
            if token not in matched_words and len(token) >= 4:
                matched_topics.append((token, "general"))

        if not matched_topics:
            return

        with self._connect() as con:
            for topic, category in matched_topics:
                con.execute(
                    "INSERT INTO user_profile_topics (topic, weight, category, first_seen, last_seen, hit_count) "
                    "VALUES (?, 1.0, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1) "
                    "ON CONFLICT(topic) DO UPDATE SET "
                    "weight = weight + 1.0, "
                    "last_seen = CURRENT_TIMESTAMP, "
                    "hit_count = hit_count + 1",
                    (topic, category),
                )

            # Update interaction count
            con.execute(
                "INSERT INTO user_profile_stats (stat_key, stat_value, updated_at) "
                "VALUES ('interaction_count', '1', CURRENT_TIMESTAMP) "
                "ON CONFLICT(stat_key) DO UPDATE SET "
                "stat_value = CAST(CAST(stat_value AS INTEGER) + 1 AS TEXT), "
                "updated_at = CURRENT_TIMESTAMP",
            )

    def apply_decay(self, decay_factor: float = 0.95) -> int:
        """Decay all topic weights and remove topics below threshold.

        Returns the number of topics removed.
        """
        with self._connect() as con:
            con.execute(
                "UPDATE user_profile_topics SET weight = weight * ?",
                (decay_factor,),
            )
            cur = con.execute(
                "DELETE FROM user_profile_topics WHERE weight < 0.1",
            )
            return cur.rowcount

    def get_top_topics(self, limit: int = 20) -> list[dict]:
        """Return topics ordered by weight descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT topic, weight, category, hit_count, first_seen, last_seen "
                "FROM user_profile_topics ORDER BY weight DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_category_distribution(self) -> dict[str, float]:
        """Return total weight per category, sorted descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT category, SUM(weight) as total_weight "
                "FROM user_profile_topics "
                "GROUP BY category ORDER BY total_weight DESC",
            ).fetchall()
            return {r["category"]: r["total_weight"] for r in rows}

    def get_stats(self) -> dict:
        """Return interaction statistics."""
        with self._connect() as con:
            # Interaction count
            row = con.execute(
                "SELECT stat_value FROM user_profile_stats "
                "WHERE stat_key = 'interaction_count'",
            ).fetchone()
            interaction_count = int(row["stat_value"]) if row else 0

            # Topic count
            row = con.execute(
                "SELECT COUNT(*) as cnt FROM user_profile_topics",
            ).fetchone()
            topic_count = row["cnt"]

            # Earliest and latest
            row = con.execute(
                "SELECT MIN(first_seen) as earliest, MAX(last_seen) as latest "
                "FROM user_profile_topics",
            ).fetchone()
            earliest = row["earliest"] if row else None
            latest = row["latest"] if row else None

            return {
                "interaction_count": interaction_count,
                "topic_count": topic_count,
                "earliest_interaction": earliest,
                "latest_interaction": latest,
            }

    def get_profile_summary(self) -> str:
        """Generate a compact text summary for injection into evolution prompts."""
        categories = self.get_category_distribution()
        if not categories:
            return ""

        total = sum(categories.values())
        if total == 0:
            return ""

        # Format category percentages
        cat_parts = []
        for cat, weight in categories.items():
            pct = weight / total * 100
            if pct >= 1:
                cat_parts.append(f"{cat} ({pct:.0f}%)")

        # Top topics
        top = self.get_top_topics(5)
        topic_names = [t["topic"] for t in top]

        stats = self.get_stats()
        interaction_count = stats["interaction_count"]

        parts = []
        if cat_parts:
            parts.append(f"User interests: {', '.join(cat_parts)}")
        if topic_names:
            parts.append(f"Top topics: {', '.join(topic_names)}")
        if interaction_count:
            parts.append(f"Total interactions: {interaction_count}")

        return "\n".join(parts)
