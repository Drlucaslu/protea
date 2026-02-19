"""Habit Detector — two-layer system for detecting task patterns.

Layer 1: Template matching — match tasks against predefined templates with
keywords and regex patterns.  Low threshold, high precision.

Layer 2: High-threshold repetitive detection — cluster tasks by keyword overlap
(Jaccard), require more occurrences and time spread, plus optional LLM
assessment.

Pure stdlib — no external dependencies (LLM client is optional).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("protea.habit_detector")

# Regex to strip conversation context prefix injected by telegram_bot.
# Matches: "[Context: ...]\nYour (previous )?message: \"...\"\nUser('s reply|now says): "
# Uses .*? (lazy) with DOTALL to handle internal quotes in bot messages.
_CONTEXT_PREFIX_RE = re.compile(
    r"^\[Context:[^\]]*\]\n"
    r"Your (?:previous )?message: \".*?\"[\n]+"
    r"User(?:'s reply|\s+now says): ",
    re.DOTALL,
)


def _strip_context_prefix(text: str) -> str:
    """Remove conversation context prefix, returning only the user's text."""
    return _CONTEXT_PREFIX_RE.sub("", text)


# Cooldown: do not re-propose the same pattern within this window.
_PROPOSE_COOLDOWN_SEC = 24 * 3600  # 24 hours

# Layer 2 defaults
_L2_MIN_OCCURRENCES = 5
_L2_MIN_DAYS = 3
_L2_WINDOW_HOURS = 72
_JACCARD_THRESHOLD = 0.5

# Fraction of tasks in same 2h window to suggest a cron time.
_TIME_CONCENTRATION_THRESHOLD = 0.6

_LLM_ASSESS_PROMPT = """\
The user has performed the following task multiple times:
{sample_tasks}

Question: Is this a pattern worth automating as a scheduled/recurring task?
Consider: Is it genuinely repetitive? Would automation save the user effort?
Answer YES or NO (one word first, then brief reasoning)."""


def load_templates(path) -> list[dict]:
    """Load task templates from a JSON file.  Returns [] on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("templates", [])
    except Exception:
        log.debug("Failed to load templates from %s", path, exc_info=True)
        return []


@dataclass
class HabitPattern:
    """A detected recurring pattern in user tasks."""

    pattern_type: str    # "template" | "repetitive"
    pattern_key: str     # unique id, e.g. "template:flight_price_tracker"
    count: int           # number of occurrences
    task_summary: str    # human-readable description
    suggested_cron: str  # suggested cron expression, "" if no time pattern
    sample_task: str     # representative task content for display
    template_name: str = ""     # template id (Layer 1)
    auto_stop_hours: int = 0    # realtime template auto-stop time
    all_samples: list[str] = field(default_factory=list)  # hit task contents


class HabitDetector:
    """Analyse task history to find recurring patterns (two-layer)."""

    def __init__(
        self,
        memory_store,
        scheduled_store=None,
        templates: list[dict] | None = None,
        llm_client=None,
        layer2_config: dict | None = None,
    ):
        self._memory = memory_store
        self._scheduled = scheduled_store
        self._templates = templates or []
        self._llm_client = llm_client
        self._proposed: dict[str, float] = {}   # pattern_key -> timestamp
        self._dismissed: set[str] = set()       # user-rejected patterns

        l2 = layer2_config or {}
        self._l2_min_occurrences = l2.get("min_occurrences", _L2_MIN_OCCURRENCES)
        self._l2_min_days = l2.get("min_days", _L2_MIN_DAYS)
        self._l2_window_hours = l2.get("window_hours", _L2_WINDOW_HOURS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, limit: int = 50) -> list[HabitPattern]:
        """Scan recent task history and return detected habit patterns."""
        tasks = self._get_tasks(limit)
        if not tasks:
            return []

        existing_scheduled = self._get_existing_scheduled_names()

        # Layer 1: Template matching
        patterns = self._detect_template_matches(tasks, existing_scheduled)

        # Layer 2: High-threshold repetitive detection (only if L1 found nothing)
        if not patterns:
            patterns = self._detect_repetitive_patterns(tasks, existing_scheduled)

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
        commands that should not count as independent repeated patterns.
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
    # Layer 1: Template matching
    # ------------------------------------------------------------------

    def _detect_template_matches(
        self, tasks: list[dict], existing_scheduled: set[str],
    ) -> list[HabitPattern]:
        """Match tasks against predefined templates (keyword + regex)."""
        if not self._templates:
            return []

        now = time.time()
        patterns: list[HabitPattern] = []

        for template in self._templates:
            tmpl_id = template.get("id", "")
            keywords = template.get("keywords", [])
            regex_patterns = template.get("regex_patterns", [])
            window_hours = template.get("window_hours", 72)
            min_hits = template.get("min_hits", 2)
            task_type = template.get("task_type", "periodic")
            default_cron = template.get("default_cron", "")
            auto_stop_hours = template.get("auto_stop_hours", 0)
            tmpl_name = template.get("name", tmpl_id)

            # Skip if already scheduled with this template name
            auto_name = f"auto_template_{tmpl_id}"
            if any(auto_name in s or tmpl_name in s for s in existing_scheduled):
                continue

            # Count matching tasks within time window
            hits: list[dict] = []
            for task in tasks:
                raw_content = task.get("content", "")
                if not raw_content:
                    continue

                # Strip conversation context prefix so we only match
                # against the user's actual text, not the bot's prior reply.
                content = _strip_context_prefix(raw_content)

                # Check time window
                ts_str = task.get("timestamp", "")
                if ts_str:
                    try:
                        task_time = datetime.fromisoformat(ts_str).timestamp()
                        if (now - task_time) > window_hours * 3600:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Keyword match (count mode: require min_keyword_hits)
                content_lower = content.lower()
                min_kw_hits = template.get("min_keyword_hits", 1)
                kw_hit_count = sum(1 for kw in keywords if kw.lower() in content_lower)
                kw_match = kw_hit_count >= min_kw_hits

                # Regex match
                rx_match = False
                for pat in regex_patterns:
                    try:
                        if re.search(pat, content):
                            rx_match = True
                            break
                    except re.error:
                        pass

                if kw_match or rx_match:
                    hits.append(task)

            if len(hits) >= min_hits and task_type != "on_demand":
                # Prefer template's default_task_text; fall back to first hit.
                sample = template.get("default_task_text", "") or hits[0].get("content", "")[:100]
                cron = default_cron or self._detect_time_pattern(hits)

                patterns.append(HabitPattern(
                    pattern_type="template",
                    pattern_key=f"template:{tmpl_id}",
                    count=len(hits),
                    task_summary=tmpl_name,
                    suggested_cron=cron,
                    sample_task=sample,
                    template_name=tmpl_id,
                    auto_stop_hours=auto_stop_hours,
                    all_samples=[_strip_context_prefix(h.get("content", ""))[:80] for h in hits[:5]],
                ))

        return patterns

    # ------------------------------------------------------------------
    # Layer 2: High-threshold repetitive detection
    # ------------------------------------------------------------------

    def _detect_repetitive_patterns(
        self, tasks: list[dict], existing_scheduled: set[str],
    ) -> list[HabitPattern]:
        """Find groups of tasks with high keyword overlap, strict thresholds."""
        clusters = self._cluster_by_keywords(tasks)
        patterns: list[HabitPattern] = []

        for cluster in clusters:
            if len(cluster) < self._l2_min_occurrences:
                continue

            # Check time spread: need tasks across multiple days.
            if not self._check_time_spread(cluster, self._l2_min_days):
                continue

            # Build cluster key from shared keywords.
            kw_sets = [self._simple_keywords(t.get("content", "")) for t in cluster]
            shared_kw = set.intersection(*kw_sets) if kw_sets else set()
            if not shared_kw:
                all_kw: Counter[str] = Counter()
                for kw_set in kw_sets:
                    all_kw.update(kw_set)
                shared_kw = {w for w, _ in all_kw.most_common(3)}

            key_label = "+".join(sorted(shared_kw)[:3])
            pattern_key = f"repetitive:{key_label}"

            # Skip if already scheduled.
            if any(key_label in s for s in existing_scheduled):
                continue

            # LLM assessment (optional)
            if not self._llm_assess_pattern(cluster):
                continue

            sample = cluster[0].get("content", "")[:100]
            cron = self._detect_time_pattern(cluster)

            patterns.append(HabitPattern(
                pattern_type="repetitive",
                pattern_key=pattern_key,
                count=len(cluster),
                task_summary=f"执行{key_label}相关任务",
                suggested_cron=cron,
                sample_task=sample,
            ))

        return patterns

    def _cluster_by_keywords(self, tasks: list[dict]) -> list[list[dict]]:
        """Single-pass Jaccard clustering of tasks by keyword sets."""
        task_kw: list[tuple[dict, set[str]]] = []
        for task in tasks:
            kw_str = task.get("keywords", "")
            if kw_str:
                kw_set = set(kw_str.lower().split())
            else:
                kw_set = self._simple_keywords(task.get("content", ""))
            if len(kw_set) >= 2:
                task_kw.append((task, kw_set))

        clusters: list[list[tuple[dict, set[str]]]] = []
        for item in task_kw:
            _, kw = item
            placed = False
            for cluster in clusters:
                _, rep_kw = cluster[0]
                if self._jaccard(kw, rep_kw) >= _JACCARD_THRESHOLD:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        return [[t for t, _ in cluster] for cluster in clusters]

    def _check_time_spread(self, tasks: list[dict], min_days: int) -> bool:
        """Check if tasks span at least *min_days* distinct calendar days."""
        days: set[str] = set()
        for task in tasks:
            ts_str = task.get("timestamp", "")
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(ts_str)
                days.add(dt.strftime("%Y-%m-%d"))
            except (ValueError, TypeError):
                pass
        return len(days) >= min_days

    def _llm_assess_pattern(self, tasks: list[dict]) -> bool:
        """Use LLM to judge if a cluster is worth automating.

        Returns False (conservative) if no LLM client or on error.
        """
        if not self._llm_client:
            return False

        sample_texts = "\n".join(
            f"- {t.get('content', '')[:100]}" for t in tasks[:5]
        )
        prompt = _LLM_ASSESS_PROMPT.format(sample_tasks=sample_texts)

        try:
            response = self._llm_client.send_message(
                "You are a task pattern analyst.", prompt,
            )
            return response.strip().upper().startswith("YES")
        except Exception:
            log.debug("LLM assess pattern failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Time pattern detection
    # ------------------------------------------------------------------

    def _detect_time_pattern(self, tasks: list[dict]) -> str:
        """If >= 60% of tasks fall in the same 2h window, suggest a cron."""
        if len(tasks) < 2:
            return ""

        hours: list[int] = []
        for task in tasks:
            ts = task.get("timestamp", "")
            h = self._extract_hour(ts)
            if h is not None:
                hours.append(h)

        if len(hours) < 2:
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
            dt = datetime.fromisoformat(timestamp_str)
            return dt.hour
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _simple_keywords(text: str) -> set[str]:
        """Extract simple keyword tokens from text.

        English: words of 3+ chars.  Chinese: bigrams (2-char segments).
        """
        raw = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower())
        tokens: set[str] = set()
        for t in raw:
            if t[0] >= "\u4e00":
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
