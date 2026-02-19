"""Habit Detector — detects repeated task patterns and proposes automation.

Scans task history from MemoryStore, identifies recurring patterns (skill reuse,
keyword clusters, time regularity), and generates proposals for scheduled tasks
or skill consolidation.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass

log = logging.getLogger("protea.habit_detector")

# Cooldown: do not re-propose the same pattern within this window.
_PROPOSE_COOLDOWN_SEC = 24 * 3600  # 24 hours

# Minimum number of occurrences to qualify as a habit.
_MIN_OCCURRENCES = 3

# Jaccard similarity threshold for keyword clustering.
_JACCARD_THRESHOLD = 0.4

# Fraction of tasks in same 2h window to suggest a cron time.
_TIME_CONCENTRATION_THRESHOLD = 0.6


@dataclass
class HabitPattern:
    """A detected recurring pattern in user tasks."""

    pattern_type: str    # "skill_reuse" | "content_cluster"
    pattern_key: str     # unique id, e.g. "skill:news_summary"
    count: int           # number of occurrences
    task_summary: str    # human-readable description
    suggested_cron: str  # suggested cron expression, "" if no time pattern
    sample_task: str     # representative task content for display


class HabitDetector:
    """Analyse task history to find recurring patterns."""

    def __init__(self, memory_store, scheduled_store=None):
        self._memory = memory_store
        self._scheduled = scheduled_store
        self._proposed: dict[str, float] = {}   # pattern_key -> timestamp
        self._dismissed: set[str] = set()       # user-rejected patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, limit: int = 50) -> list[HabitPattern]:
        """Scan recent task history and return detected habit patterns."""
        tasks = self._get_tasks(limit)
        if not tasks:
            return []

        # Collect existing scheduled task names for exclusion.
        existing_scheduled = self._get_existing_scheduled_names()

        patterns: list[HabitPattern] = []

        # Pattern 1: Skill reuse
        patterns.extend(self._detect_skill_reuse(tasks, existing_scheduled))

        # Pattern 2: Content keyword clusters
        patterns.extend(self._detect_content_clusters(tasks, existing_scheduled))

        # Filter out proposed/dismissed
        patterns = [p for p in patterns if self._should_propose(p.pattern_key)]

        return patterns

    def mark_proposed(self, pattern_key: str) -> None:
        """Record that a pattern has been proposed; suppress for 24h."""
        self._proposed[pattern_key] = time.time()

    def mark_dismissed(self, pattern_key: str) -> None:
        """User rejected this pattern; never propose again."""
        self._dismissed.add(pattern_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tasks(self, limit: int) -> list[dict]:
        """Fetch recent task entries from memory.

        Only keeps tasks with importance >= 0.5, filtering out operational
        commands (e.g. "/calendar 取消") and session follow-up corrections
        that should not count as independent repeated patterns.
        """
        try:
            tasks = self._memory.get_by_type("task", limit=limit)
            return [t for t in tasks if t.get("importance", 0.5) >= 0.5]
        except Exception:
            log.debug("Failed to fetch tasks from memory", exc_info=True)
            return []

    def _get_existing_scheduled_names(self) -> set[str]:
        """Return set of task_text values from enabled scheduled tasks."""
        if not self._scheduled:
            return set()
        try:
            enabled = self._scheduled.get_enabled()
            names: set[str] = set()
            for s in enabled:
                names.add(s.get("name", ""))
                names.add(s.get("task_text", ""))
            return names
        except Exception:
            return set()

    def _should_propose(self, pattern_key: str) -> bool:
        """Check cooldown and dismissal filters."""
        if pattern_key in self._dismissed:
            return False
        proposed_at = self._proposed.get(pattern_key)
        if proposed_at and (time.time() - proposed_at) < _PROPOSE_COOLDOWN_SEC:
            return False
        return True

    # ------------------------------------------------------------------
    # Pattern 1: Skill reuse
    # ------------------------------------------------------------------

    def _detect_skill_reuse(
        self, tasks: list[dict], existing_scheduled: set[str],
    ) -> list[HabitPattern]:
        """Find skills used >= _MIN_OCCURRENCES times."""
        skill_counter: Counter[str] = Counter()
        skill_tasks: dict[str, list[dict]] = {}

        for task in tasks:
            meta = task.get("metadata", {})
            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            skills_used = meta.get("skills_used", [])
            for skill in skills_used:
                skill_counter[skill] += 1
                skill_tasks.setdefault(skill, []).append(task)

        patterns: list[HabitPattern] = []
        for skill_name, count in skill_counter.most_common():
            if count < _MIN_OCCURRENCES:
                break

            pattern_key = f"skill:{skill_name}"

            # Skip if already covered by a scheduled task.
            if any(skill_name in s for s in existing_scheduled):
                continue

            related = skill_tasks[skill_name]
            sample = related[0].get("content", skill_name)[:100]
            cron = self._detect_time_pattern(related)

            patterns.append(HabitPattern(
                pattern_type="skill_reuse",
                pattern_key=pattern_key,
                count=count,
                task_summary=f"run_skill {skill_name}",
                suggested_cron=cron,
                sample_task=sample,
            ))

        return patterns

    # ------------------------------------------------------------------
    # Pattern 2: Content keyword clusters
    # ------------------------------------------------------------------

    def _detect_content_clusters(
        self, tasks: list[dict], existing_scheduled: set[str],
    ) -> list[HabitPattern]:
        """Find groups of tasks with high keyword overlap (Jaccard > threshold)."""
        # Extract keyword sets per task.
        task_kw: list[tuple[dict, set[str]]] = []
        for task in tasks:
            kw_str = task.get("keywords", "")
            if not kw_str:
                # Fallback: extract from content.
                content = task.get("content", "")
                kw_set = self._simple_keywords(content)
            else:
                kw_set = set(kw_str.lower().split())
            if len(kw_set) >= 2:  # Need at least 2 keywords to cluster.
                task_kw.append((task, kw_set))

        if len(task_kw) < _MIN_OCCURRENCES:
            return []

        # Simple single-pass clustering: assign each task to the first
        # cluster it matches (Jaccard >= threshold), or create a new one.
        clusters: list[list[tuple[dict, set[str]]]] = []
        for item in task_kw:
            _, kw = item
            placed = False
            for cluster in clusters:
                # Compare against the cluster's "representative" (first entry).
                _, rep_kw = cluster[0]
                if self._jaccard(kw, rep_kw) >= _JACCARD_THRESHOLD:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        patterns: list[HabitPattern] = []
        for cluster in clusters:
            if len(cluster) < _MIN_OCCURRENCES:
                continue

            # Build cluster key from shared keywords.
            shared_kw = set.intersection(*(kw for _, kw in cluster))
            if not shared_kw:
                # Use the most common keywords across the cluster.
                all_kw: Counter[str] = Counter()
                for _, kw in cluster:
                    all_kw.update(kw)
                shared_kw = {w for w, _ in all_kw.most_common(3)}

            key_label = "+".join(sorted(shared_kw)[:3])
            pattern_key = f"cluster:{key_label}"

            # Skip if already scheduled.
            if any(key_label in s for s in existing_scheduled):
                continue

            sample = cluster[0][0].get("content", "")[:100]
            related_tasks = [t for t, _ in cluster]
            cron = self._detect_time_pattern(related_tasks)

            patterns.append(HabitPattern(
                pattern_type="content_cluster",
                pattern_key=pattern_key,
                count=len(cluster),
                task_summary=f"执行{key_label}相关任务",
                suggested_cron=cron,
                sample_task=sample,
            ))

        return patterns

    # ------------------------------------------------------------------
    # Time pattern detection
    # ------------------------------------------------------------------

    def _detect_time_pattern(self, tasks: list[dict]) -> str:
        """If >= 60% of tasks fall in the same 2h window, suggest a cron."""
        if len(tasks) < _MIN_OCCURRENCES:
            return ""

        hours: list[int] = []
        for task in tasks:
            ts = task.get("timestamp", "")
            h = self._extract_hour(ts)
            if h is not None:
                hours.append(h)

        if len(hours) < _MIN_OCCURRENCES:
            return ""

        # Check each 2h window.
        best_count = 0
        best_start = 0
        for start in range(24):
            end = (start + 2) % 24
            if start < end:
                count = sum(1 for h in hours if start <= h < end)
            else:
                count = sum(1 for h in hours if h >= start or h < end)
            if count > best_count:
                best_count = count
                best_start = start

        if best_count / len(hours) >= _TIME_CONCENTRATION_THRESHOLD:
            return f"0 {best_start} * * *"

        return ""

    @staticmethod
    def _extract_hour(timestamp_str: str) -> int | None:
        """Extract hour from an ISO timestamp string."""
        if not timestamp_str:
            return None
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp_str)
            return dt.hour
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _simple_keywords(text: str) -> set[str]:
        """Extract simple keyword tokens from text.

        English: words of 3+ chars.  Chinese: bigrams (2-char segments).
        """
        import re
        raw = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower())
        tokens: set[str] = set()
        for t in raw:
            if t[0] >= "\u4e00":
                # Chinese: generate bigrams.
                for i in range(len(t) - 1):
                    tokens.add(t[i : i + 2])
            elif len(t) >= 3:
                tokens.add(t)
        return tokens

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0
