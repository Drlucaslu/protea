"""User profiler backed by SQLite.

Extracts topics from task history via keyword matching against predefined
categories.  Tracks topic weights with decay, producing a compact profile
summary that can be injected into evolution prompts.

Also provides PreferenceStore — a structured preference layer that captures
implicit signals (moments) from user interactions and aggregates them into
stable preferences with confidence scores.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import time

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
        "github", "typescript", "markdown", "async", "merge", "template",
        "interface", "login", "directory", "pattern", "logic", "program",
        "functions", "components", "subroutine", "input", "execution",
        "dedupe", "implementation", "files", "instructions", "looping",
        "duplicate", "record", "standalone", "comment", "module", "library",
        "runtime", "callback", "parameter", "config", "regex", "parsing",
        "代码", "编程", "调试", "函数", "变量", "算法", "重构",
    },
    "math": {
        "equation", "calculate", "matrix", "statistics", "probability",
        "algebra", "geometry", "integral", "derivative", "theorem",
        "formula", "vector", "linear", "optimization", "numerical",
        "计算", "方程", "矩阵", "统计", "概率", "公式",
    },
    "data": {
        "csv", "json", "database", "sql", "pandas", "analysis",
        "visualization", "dataset", "dataframe", "query", "table",
        "schema", "etl", "pipeline", "warehouse",
        "数据", "数据库", "分析", "可视化", "查询", "报表",
    },
    "web": {
        "http", "html", "css", "scrape", "fetch", "url", "browser",
        "request", "endpoint", "websocket", "server", "client",
        "rest", "graphql", "cors",
        "website", "domain", "email", "wechat", "instagram", "twitter",
        "online", "links", "article", "articles",
        "网页", "爬虫", "浏览器", "请求", "链接", "网站",
    },
    "ai": {
        "model", "llm", "neural", "machine", "learning", "transformer",
        "embedding", "training", "inference", "prompt", "gpt", "claude",
        "chatbot", "nlp", "classification",
        "intent", "intents", "prompts", "agents", "detection", "models",
        "recognition", "anomalies", "robotics", "automation", "robot",
        "personalized", "recommendations", "computational", "algorithmic",
        "accuracy", "conversation", "awareness", "traits", "behavior",
        "模型", "训练", "推理", "神经", "智能", "机器",
    },
    "system": {
        "file", "process", "shell", "command", "docker", "deploy",
        "server", "linux", "terminal", "daemon", "cron", "systemd",
        "container", "kubernetes", "nginx",
        "restart", "restarts", "reboot", "crash", "malfunction", "freeze",
        "monitoring", "hardware", "device", "devices", "network",
        "performance", "cloud", "computer", "timer", "keyboard", "mouse",
        "screen", "screens", "snapshot", "privileges", "operational",
        "power", "processes",
        "文件", "进程", "命令", "部署", "服务器", "终端",
    },
    "creative": {
        "write", "story", "poem", "generate", "image", "music", "design",
        "art", "creative", "narrative", "compose", "sketch", "animation",
        "illustration", "game",
        "写作", "故事", "设计", "创意", "生成", "图片",
    },
    "finance": {
        "stock", "market", "trade", "price", "portfolio", "crypto",
        "investment", "dividend", "forex", "bond", "revenue", "profit",
        "accounting", "budget", "tax",
        "blockchain", "valuation", "funds", "financing", "funding",
        "investments", "invested", "sales", "payment", "contract",
        "purchase", "premium", "rates", "company", "raising", "decline",
        "股票", "市场", "交易", "价格", "投资", "收益", "充值", "金额",
    },
    "research": {
        "paper", "study", "survey", "literature", "academic", "journal",
        "citation", "hypothesis", "experiment", "methodology", "thesis",
        "review", "abstract", "conclusion", "reference",
        "arxiv", "papers", "diagnosis", "trends", "comparison", "analyzing",
        "论文", "研究", "学术", "文献", "摘要", "综述",
    },
    "lifestyle": {
        "fitness", "exercise", "workout", "running", "health", "diet",
        "sleep", "schedule", "daily", "routine", "habit", "calendar",
        "travel", "recipe", "cooking", "weather", "reminder",
        "meeting", "family", "office", "notification", "schedules",
        "健身", "运动", "跑步", "健康", "饮食", "睡眠", "日程", "习惯",
    },
}

# Inverted index: keyword → category
_KEYWORD_TO_CATEGORY: dict[str, str] = {}
for _cat, _kws in CATEGORY_KEYWORDS.items():
    for _kw in _kws:
        _KEYWORD_TO_CATEGORY[_kw] = _cat

# Simple stop words (English)
_STOP_WORDS = {
    # Basic function words
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
    "make", "like", "use", "help", "want", "wants", "need", "needs",
    "please", "thanks", "thank",
    # Common verbs in LLM-translated intent sentences
    "find", "show", "send", "look", "check", "start", "stop", "set",
    "add", "remove", "delete", "create", "run", "see", "tell", "ask",
    "know", "give", "take", "try", "keep", "put", "call", "come", "go",
    "execute", "improve", "solve", "appear", "indicate", "represent",
    "suggest", "confirm", "establish", "prevent", "clarify", "perform",
    "deliver", "interact", "proceed", "affect", "receive", "contain",
    "respond", "refer", "develop", "explore", "access",
    # Verb forms (-ing / -ed / -s)
    "causing", "asking", "pushing", "entering", "reaching", "launching",
    "containing", "combining", "mentioning", "demonstrating", "doing",
    "going", "sending", "sends", "working",
    "changed", "included", "returned", "selected", "described",
    "requested", "formed", "involved", "accumulated", "discarded",
    "suspected", "positioned", "borrowed", "learned", "received",
    "scheduled", "needed", "automated",
    "appears", "indicates", "contains", "fails", "keeps", "covers",
    "causes", "remains", "replies",
    # LLM instruction / description words
    "currently", "recently", "available", "specific", "related",
    "whether", "based", "using", "regarding", "including",
    "information", "details", "status", "report", "result", "results",
    "summary", "recent", "current", "new", "first", "last", "next",
    "user", "system", "feature", "task", "message", "messages",
    "explain", "understand", "identify", "determine", "implement",
    "analyze", "summarize", "compare", "evaluate", "provide",
    "ready", "able", "still", "done", "made", "sent", "given",
    "search", "display", "fix", "cancel", "disable", "enable",
    "modify", "update", "change",
    "evolving", "evolved", "evolution", "developing", "developed",
    "content", "features", "capabilities", "functionality",
    "hourly", "minutes", "every", "time", "times",
    "version", "control", "value", "chinese", "english",
    "correct", "incorrect", "complete", "completed", "previous",
    "items", "level", "levels",
    "actual", "actually", "issue", "issues", "error", "errors",
    "operation", "operations", "response", "responses",
    # Generic adjectives / adverbs
    "unnecessary", "similar", "comprehensive", "reasonable", "permanent",
    "popular", "detailed", "unrelated", "certain", "external", "original",
    "unable", "unclear", "dangerous", "important", "single", "latest",
    "multiple", "recurring", "concise",
    "immediately", "specifically", "directly", "continuously",
    "completely", "successfully", "automatically", "frequently",
    # Generic connectors / fillers
    "instead", "without", "already", "versus", "further", "forward",
    "because", "since", "except", "cannot", "shouldn", "haven",
    "better", "brief", "short", "medium", "three", "means",
    # Generic nouns (no topical signal)
    "something", "direction", "directions", "approach", "solution",
    "instances", "examples", "requirements", "suggestions", "improvements",
    "advantages", "options", "actions", "abilities", "applications",
    "plans", "cycles", "progress", "changes", "questions", "cases",
    "maximum", "minimum", "limit", "order", "scale", "format",
    # Temporal words
    "today", "tomorrow", "yesterday", "morning", "month", "months",
    "hours", "january", "timing",
}

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")

# Known system/noise tokens to exclude from profiling
_NOISE_TOKENS = frozenset({
    # Protea internal terms
    "protea", "sentinel", "ring0", "ring1", "ring2", "ring",
    "telegram", "message", "generation", "evolution", "evolver",
    "openclaw", "openviking", "bytefuture",
    "dashboard", "skill", "skills", "memory", "memories", "agent", "task", "tasks",
    "commit", "report", "output", "error", "system", "token",
    "phase", "summary", "notes", "research",
    # Evolution / genetics terms (Protea internals)
    "genes", "mutations", "evolutionary", "evolve", "evomap", "directive",
    "crystallization", "compacting", "pollution", "contaminated",
    "stimulate", "environmental", "typeless", "liang",
    # Context / response residuals
    "context", "previous", "sent", "says", "reply", "sorry", "couldn",
    "user", "users", "last",
    # System / ops terms
    "timeout", "operation", "shutdown", "cooldown", "repair", "archive",
    "compacted", "entries", "recalled", "timed",
    # Timestamps / IDs
    "2024", "2025", "2026",
})

# Chinese noise bigrams (system terms)
_NOISE_TOKENS_ZH = frozenset({
    "消息", "用户", "回复", "发送", "系统", "进程",
})


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, filtering stop words and noise.

    English tokens: len >= 3, not in stop words or noise list.
    Chinese tokens: extracted as bigrams (2-char sliding window).
    """
    raw = _WORD_RE.findall(text.lower())
    tokens: list[str] = []
    for t in raw:
        if t[0] >= "\u4e00":  # CJK range
            for i in range(len(t) - 1):
                bigram = t[i : i + 2]
                if bigram not in _NOISE_TOKENS_ZH:
                    tokens.append(bigram)
        elif len(t) >= 3 and t not in _STOP_WORDS and t not in _NOISE_TOKENS:
            tokens.append(t)
    return tokens


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

        # Check bigrams against known category patterns (only categorized
        # bigrams are kept — unmatched bigrams are dropped, not sent to general)
        for bigram in bigrams:
            parts = bigram.split("_")
            for part in parts:
                if part in _KEYWORD_TO_CATEGORY:
                    matched_topics.append((bigram, _KEYWORD_TO_CATEGORY[part]))
                    break

        # Unmatched English tokens with length >= 5 go to 'general'
        # Chinese bigrams and underscore bigrams excluded — too noisy
        matched_words = {t for t, _ in matched_topics}
        for token in tokens:
            if token not in matched_words and token not in _NOISE_TOKENS:
                if token[0] < "\u4e00" and len(token) >= 5:
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


# ---------------------------------------------------------------------------
# Structured Preference Store
# ---------------------------------------------------------------------------

_CREATE_PREFERENCES_TABLE = """\
CREATE TABLE IF NOT EXISTS user_preferences (
    id              INTEGER PRIMARY KEY,
    preference_key  TEXT    UNIQUE NOT NULL,
    category        TEXT    DEFAULT 'general',
    value           TEXT    NOT NULL,
    confidence      REAL    DEFAULT 0.5,
    source          TEXT    DEFAULT 'implicit',
    moment_count    INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_MOMENTS_TABLE = """\
CREATE TABLE IF NOT EXISTS preference_moments (
    id              INTEGER PRIMARY KEY,
    moment_type     TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    category        TEXT    DEFAULT 'general',
    extracted_signal TEXT   DEFAULT '',
    aggregated      BOOLEAN DEFAULT 0,
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_DRIFT_LOG_TABLE = """\
CREATE TABLE IF NOT EXISTS preference_drift_log (
    id              INTEGER PRIMARY KEY,
    preference_key  TEXT    NOT NULL,
    old_confidence  REAL,
    new_confidence  REAL,
    drift_direction TEXT    NOT NULL,
    trigger         TEXT    DEFAULT '',
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
)
"""


class PreferenceStore:
    """Structured preference storage with moment → preference → routine model.

    Moments are raw signals extracted from user interactions. When enough
    moments accumulate for a topic, they are aggregated into a stable
    preference with a confidence score.
    """

    def __init__(self, db_path: pathlib.Path, config: dict | None = None) -> None:
        self.db_path = db_path
        cfg = config or {}
        self.moment_aggregation_threshold = cfg.get("moment_aggregation_threshold", 3)
        self.confidence_decay_rate = cfg.get("confidence_decay_rate", 0.98)
        self.max_preferences = cfg.get("max_preferences", 50)
        with self._connect() as con:
            con.execute(_CREATE_PREFERENCES_TABLE)
            con.execute(_CREATE_MOMENTS_TABLE)
            con.execute(_CREATE_DRIFT_LOG_TABLE)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    # ---- Moment Layer ----

    def store_moment(
        self,
        moment_type: str,
        content: str,
        category: str = "general",
        extracted_signal: str = "",
    ) -> int:
        """Store a raw preference signal (moment).

        Args:
            moment_type: Type of moment (task_preference, language_choice,
                         time_pattern, feedback, behavior).
            content: The raw observation text.
            category: Category from UserProfiler taxonomy.
            extracted_signal: LLM-extracted preference signal (if available).

        Returns:
            The rowid of the inserted moment.
        """
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO preference_moments "
                "(moment_type, content, category, extracted_signal) "
                "VALUES (?, ?, ?, ?)",
                (moment_type, content, category, extracted_signal),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_pending_moments(self, limit: int = 50) -> list[dict]:
        """Return unaggregated moments, oldest first."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM preference_moments "
                "WHERE aggregated = 0 ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- Aggregation Layer ----

    def aggregate_moments(self) -> int:
        """Aggregate pending moments into stable preferences.

        Groups moments by category, and for each group with enough moments
        (>= threshold), creates or updates a preference entry.

        Returns the number of preferences created or updated.
        """
        pending = self.get_pending_moments(100)
        if not pending:
            return 0

        # Group by category.
        groups: dict[str, list[dict]] = {}
        for m in pending:
            groups.setdefault(m["category"], []).append(m)

        updated = 0
        with self._connect() as con:
            for category, moments in groups.items():
                if len(moments) < self.moment_aggregation_threshold:
                    continue

                # Build preference value from moment signals.
                signals = [
                    m["extracted_signal"] or m["content"]
                    for m in moments
                ]
                # Use the most common signal as the preference value.
                signal_counts: dict[str, int] = {}
                for s in signals:
                    s_lower = s.strip().lower()[:100]
                    if s_lower:
                        signal_counts[s_lower] = signal_counts.get(s_lower, 0) + 1
                if not signal_counts:
                    continue

                top_signal = max(signal_counts, key=signal_counts.get)  # type: ignore[arg-type]
                pref_key = f"{category}:{top_signal[:50]}"
                moment_count = len(moments)

                # Confidence: more moments = higher confidence, capped at 0.95.
                confidence = min(0.3 + moment_count * 0.1, 0.95)

                # Upsert preference.
                existing = con.execute(
                    "SELECT id, confidence, moment_count FROM user_preferences "
                    "WHERE preference_key = ?",
                    (pref_key,),
                ).fetchone()

                if existing:
                    new_confidence = min(
                        existing["confidence"] + moment_count * 0.05, 0.95,
                    )
                    new_count = existing["moment_count"] + moment_count
                    con.execute(
                        "UPDATE user_preferences SET "
                        "confidence = ?, moment_count = ?, "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE id = ?",
                        (new_confidence, new_count, existing["id"]),
                    )
                else:
                    con.execute(
                        "INSERT INTO user_preferences "
                        "(preference_key, category, value, confidence, "
                        "source, moment_count) "
                        "VALUES (?, ?, ?, ?, 'implicit', ?)",
                        (pref_key, category, top_signal, confidence, moment_count),
                    )
                updated += 1

                # Mark moments as aggregated.
                ids = [m["id"] for m in moments]
                placeholders = ",".join("?" * len(ids))
                con.execute(
                    f"UPDATE preference_moments SET aggregated = 1 "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )

        return updated

    # ---- Preference Query Layer ----

    def get_preferences(self, category: str | None = None, limit: int = 20) -> list[dict]:
        """Return preferences ordered by confidence descending."""
        with self._connect() as con:
            if category:
                rows = con.execute(
                    "SELECT * FROM user_preferences "
                    "WHERE category = ? ORDER BY confidence DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM user_preferences "
                    "ORDER BY confidence DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_structured_profile(self) -> dict:
        """Return a structured profile dict suitable for LLM injection.

        Returns:
            {
                "preferences": [{"key": ..., "category": ..., "value": ..., "confidence": ...}],
                "categories": {"category": avg_confidence},
                "total_moments": int,
                "total_preferences": int,
            }
        """
        with self._connect() as con:
            prefs = con.execute(
                "SELECT preference_key, category, value, confidence "
                "FROM user_preferences "
                "ORDER BY confidence DESC LIMIT ?",
                (self.max_preferences,),
            ).fetchall()

            moment_count = con.execute(
                "SELECT COUNT(*) as cnt FROM preference_moments",
            ).fetchone()["cnt"]

            pref_count = con.execute(
                "SELECT COUNT(*) as cnt FROM user_preferences",
            ).fetchone()["cnt"]

        pref_list = [
            {
                "key": r["preference_key"],
                "category": r["category"],
                "value": r["value"],
                "confidence": round(r["confidence"], 2),
            }
            for r in prefs
        ]

        # Category summary.
        cat_scores: dict[str, list[float]] = {}
        for p in pref_list:
            cat_scores.setdefault(p["category"], []).append(p["confidence"])
        categories = {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in cat_scores.items()
        }

        return {
            "preferences": pref_list,
            "categories": categories,
            "total_moments": moment_count,
            "total_preferences": pref_count,
        }

    # ---- Confidence Decay ----

    def apply_confidence_decay(self) -> int:
        """Decay all preference confidences and remove below threshold.

        Returns the number of preferences removed.
        """
        with self._connect() as con:
            con.execute(
                "UPDATE user_preferences SET confidence = confidence * ?",
                (self.confidence_decay_rate,),
            )
            cur = con.execute(
                "DELETE FROM user_preferences WHERE confidence < 0.1",
            )
            return cur.rowcount

    # ---- Explicit Preference Recording ----

    def record_explicit(
        self, preference_key: str, value: str, category: str = "general",
    ) -> None:
        """Record an explicit user preference (from feedback or statement)."""
        with self._connect() as con:
            existing = con.execute(
                "SELECT id FROM user_preferences WHERE preference_key = ?",
                (preference_key,),
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE user_preferences SET "
                    "value = ?, confidence = MIN(confidence + 0.15, 0.95), "
                    "source = 'explicit', updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (value, existing["id"]),
                )
            else:
                con.execute(
                    "INSERT INTO user_preferences "
                    "(preference_key, category, value, confidence, source) "
                    "VALUES (?, ?, ?, 0.8, 'explicit')",
                    (preference_key, category, value),
                )

    # ---- Drift Detection ----

    def detect_drift(self, window_days: int = 7) -> list[dict]:
        """Detect preference drift by comparing recent vs older moments.

        Looks for categories where recent activity significantly differs
        from the stored preference confidence levels.

        Returns a list of drift events:
            [{"preference_key": ..., "old_confidence": ..., "new_confidence": ...,
              "direction": "rising"|"falling", "magnitude": float}]
        """
        drift_events: list[dict] = []
        cutoff = time.time() - window_days * 86400
        cutoff_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(cutoff))

        with self._connect() as con:
            # Count recent moments per category.
            recent = con.execute(
                "SELECT category, COUNT(*) as cnt "
                "FROM preference_moments "
                "WHERE created_at >= ? "
                "GROUP BY category",
                (cutoff_iso,),
            ).fetchall()
            recent_counts = {r["category"]: r["cnt"] for r in recent}

            # Compare with current preferences.
            prefs = con.execute(
                "SELECT preference_key, category, confidence "
                "FROM user_preferences",
            ).fetchall()

            # Aggregate by category.
            cat_confidence: dict[str, float] = {}
            cat_prefs: dict[str, list[dict]] = {}
            for p in prefs:
                cat = p["category"]
                cat_confidence.setdefault(cat, 0.0)
                cat_confidence[cat] = max(cat_confidence[cat], p["confidence"])
                cat_prefs.setdefault(cat, []).append(dict(p))

            # Detect rising categories (lots of recent activity, low confidence).
            for cat, count in recent_counts.items():
                existing_conf = cat_confidence.get(cat, 0.0)
                if count >= 5 and existing_conf < 0.5:
                    new_conf = min(0.3 + count * 0.08, 0.9)
                    event = {
                        "preference_key": f"{cat}:rising",
                        "old_confidence": round(existing_conf, 2),
                        "new_confidence": round(new_conf, 2),
                        "direction": "rising",
                        "magnitude": round(new_conf - existing_conf, 2),
                        "category": cat,
                        "recent_count": count,
                    }
                    drift_events.append(event)
                    # Log the drift.
                    con.execute(
                        "INSERT INTO preference_drift_log "
                        "(preference_key, old_confidence, new_confidence, "
                        "drift_direction, trigger) "
                        "VALUES (?, ?, ?, 'rising', ?)",
                        (event["preference_key"], existing_conf, new_conf,
                         f"{count} moments in {window_days}d"),
                    )

            # Detect falling categories (high confidence, no recent activity).
            for cat, conf in cat_confidence.items():
                if conf >= 0.6 and recent_counts.get(cat, 0) == 0:
                    new_conf = conf * 0.7
                    event = {
                        "preference_key": f"{cat}:falling",
                        "old_confidence": round(conf, 2),
                        "new_confidence": round(new_conf, 2),
                        "direction": "falling",
                        "magnitude": round(conf - new_conf, 2),
                        "category": cat,
                        "recent_count": 0,
                    }
                    drift_events.append(event)
                    con.execute(
                        "INSERT INTO preference_drift_log "
                        "(preference_key, old_confidence, new_confidence, "
                        "drift_direction, trigger) "
                        "VALUES (?, ?, ?, 'falling', ?)",
                        (event["preference_key"], conf, new_conf,
                         f"no activity in {window_days}d"),
                    )

        return drift_events

    def get_drift_log(self, limit: int = 20) -> list[dict]:
        """Return recent drift log entries."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM preference_drift_log "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_preference_summary_text(self) -> str:
        """Generate a text summary of structured preferences for prompt injection."""
        profile = self.get_structured_profile()
        if not profile["preferences"]:
            return ""

        parts = []
        if profile["categories"]:
            cat_parts = [
                f"{cat} ({conf:.0%})"
                for cat, conf in sorted(
                    profile["categories"].items(),
                    key=lambda x: -x[1],
                )
                if conf >= 0.2
            ]
            if cat_parts:
                parts.append(f"Interest areas: {', '.join(cat_parts)}")

        high_conf = [
            p for p in profile["preferences"] if p["confidence"] >= 0.5
        ]
        if high_conf:
            pref_parts = [f"{p['value']} ({p['confidence']:.0%})" for p in high_conf[:5]]
            parts.append(f"Strong preferences: {', '.join(pref_parts)}")

        parts.append(
            f"Data: {profile['total_moments']} observations, "
            f"{profile['total_preferences']} preferences"
        )
        return "\n".join(parts)
