"""Structured preference store backed by SQLite.

Captures implicit signals (moments) from user interactions and aggregates
them into stable preferences with confidence scores.  Includes drift
detection and reclassification helpers.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import pathlib
import sqlite3
import time

from ring0.user_profile import _KEYWORD_TO_CATEGORY

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
    """Structured preference storage with moment -> preference -> routine model.

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

                # Upsert preference — but never overwrite soul-sourced entries.
                existing = con.execute(
                    "SELECT id, confidence, moment_count, source "
                    "FROM user_preferences "
                    "WHERE preference_key = ?",
                    (pref_key,),
                ).fetchone()

                if existing and existing["source"] == "soul":
                    # Soul entries are constitutional — skip aggregation.
                    ids = [m["id"] for m in moments]
                    placeholders = ",".join("?" * len(ids))
                    con.execute(
                        f"UPDATE preference_moments SET aggregated = 1 "
                        f"WHERE id IN ({placeholders})",
                        ids,
                    )
                    continue

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
                "UPDATE user_preferences SET confidence = confidence * ? "
                "WHERE source != 'soul'",
                (self.confidence_decay_rate,),
            )
            cur = con.execute(
                "DELETE FROM user_preferences WHERE confidence < 0.1",
            )
            return cur.rowcount

    # ---- Explicit Preference Recording ----

    @staticmethod
    def _infer_category(text: str, fallback: str = "lifestyle") -> str:
        """Infer a category from text using keyword matching.

        Checks if any keyword from _KEYWORD_TO_CATEGORY appears in the text.
        Returns the matched category or the fallback.
        """
        text_lower = text.lower()
        # Exact token match first.
        for word in text_lower.split():
            if word in _KEYWORD_TO_CATEGORY:
                return _KEYWORD_TO_CATEGORY[word]
        # Substring match.
        for kw, cat in _KEYWORD_TO_CATEGORY.items():
            if len(kw) >= 3 and kw in text_lower:
                return cat
        return fallback

    def reclassify_general(self, fallback_category: str = "lifestyle") -> dict:
        """Reclassify all 'general' rows in preference_moments and user_preferences.

        Returns {"moments": int, "preferences": int}.
        """
        result = {"moments": 0, "preferences": 0}

        with self._connect() as con:
            # Reclassify moments.
            rows = con.execute(
                "SELECT id, content, extracted_signal FROM preference_moments "
                "WHERE category = 'general'",
            ).fetchall()
            for row in rows:
                text = row["extracted_signal"] or row["content"]
                new_cat = self._infer_category(text, fallback_category)
                con.execute(
                    "UPDATE preference_moments SET category = ? WHERE id = ?",
                    (new_cat, row["id"]),
                )
                result["moments"] += 1

            # Reclassify preferences.
            rows = con.execute(
                "SELECT id, preference_key, value FROM user_preferences "
                "WHERE category = 'general'",
            ).fetchall()
            for row in rows:
                text = f"{row['preference_key']} {row['value']}"
                new_cat = self._infer_category(text, fallback_category)
                con.execute(
                    "UPDATE user_preferences SET category = ? WHERE id = ?",
                    (new_cat, row["id"]),
                )
                result["preferences"] += 1

        return result

    def record_explicit(
        self, preference_key: str, value: str, category: str = "lifestyle",
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

    def sync_soul_rules(self, rules: list[str]) -> int:
        """Sync soul rules into user_preferences (source='soul', confidence=1.0).

        Upserts each rule and removes stale soul entries no longer in the list.
        Returns the number of rules synced.
        """
        current_keys = set()
        with self._connect() as con:
            for rule in rules:
                key = f"soul:{rule[:50]}"
                current_keys.add(key)
                existing = con.execute(
                    "SELECT id FROM user_preferences WHERE preference_key = ?",
                    (key,),
                ).fetchone()
                if existing:
                    con.execute(
                        "UPDATE user_preferences SET value = ?, confidence = 1.0, "
                        "source = 'soul', updated_at = CURRENT_TIMESTAMP "
                        "WHERE id = ?",
                        (rule, existing["id"]),
                    )
                else:
                    con.execute(
                        "INSERT INTO user_preferences "
                        "(preference_key, category, value, confidence, source) "
                        "VALUES (?, 'soul', ?, 1.0, 'soul')",
                        (key, rule),
                    )
            # Remove stale soul entries not in current rules.
            soul_rows = con.execute(
                "SELECT id, preference_key FROM user_preferences WHERE source = 'soul'",
            ).fetchall()
            for row in soul_rows:
                if row["preference_key"] not in current_keys:
                    con.execute("DELETE FROM user_preferences WHERE id = ?", (row["id"],))
        return len(rules)

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
