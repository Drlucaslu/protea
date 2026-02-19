"""Habit Detector — two-layer system for detecting task patterns.

Layer 1: Template matching — match tasks against predefined templates with
keywords and regex patterns.  Low threshold, high precision.

Layer 2: Tool-sequence + intent detection — cluster tasks by tool-call profile
similarity, filtered by intent repeatability.  No LLM required.

Pure stdlib — no external dependencies (LLM client is optional).
"""

from __future__ import annotations

import json
import logging
import math
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
_L2_MIN_OCCURRENCES = 3
_L2_MIN_DAYS = 2
_L2_WINDOW_HOURS = 72
_JACCARD_THRESHOLD = 0.5  # kept for backward compat (tests may reference)

# Fraction of tasks in same 2h window to suggest a cron time.
_TIME_CONCENTRATION_THRESHOLD = 0.6

# Tool → (intent_category, repeatability_weight)
_TOOL_INTENT_MAP: dict[str, tuple[str, float]] = {
    "web_search":      ("search",  1.0),
    "web_fetch":       ("search",  0.8),
    "run_skill":       ("skill",   1.0),
    "exec":            ("process", 0.5),
    "read_file":       ("process", 0.3),
    "list_dir":        ("process", 0.2),
    "spawn":           ("process", 0.3),
    "view_skill":      ("skill",   0.3),
    "write_file":      ("create", -0.5),
    "edit_file":       ("create", -0.5),
    "edit_skill":      ("create", -0.3),
    "generate_pdf":    ("create", -0.3),
    "message":         ("chat",    0.0),
    "send_file":       ("create",  0.0),
    "manage_schedule": ("admin",  -1.0),
}


def _get_tool_sequence(task: dict) -> list[str]:
    """Extract tool_sequence from task metadata, handling JSON strings."""
    meta = task.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return []
    seq = meta.get("tool_sequence", [])
    if isinstance(seq, str):
        try:
            seq = json.loads(seq)
        except (json.JSONDecodeError, TypeError):
            return []
    return seq if isinstance(seq, list) else []


def classify_task_intent(task: dict) -> tuple[str, bool]:
    """Classify task intent and repeatability from tool_sequence metadata.

    Returns (intent_category, is_repeatable).
    """
    seq = _get_tool_sequence(task)
    if not seq:
        return ("unknown", False)

    counts: Counter[str] = Counter(seq)
    total = sum(counts.values())

    # Weighted repeatability score
    score = 0.0
    for tool, count in counts.items():
        _, weight = _TOOL_INTENT_MAP.get(tool, ("unknown", 0.0))
        score += weight * count

    is_repeatable = (score / total) > 0.2

    # Intent category by majority vote
    intent_votes: Counter[str] = Counter()
    for tool, count in counts.items():
        cat, _ = _TOOL_INTENT_MAP.get(tool, ("unknown", 0.0))
        intent_votes[cat] += count

    intent = intent_votes.most_common(1)[0][0] if intent_votes else "unknown"
    return (intent, is_repeatable)


def _tool_profile(task: dict) -> dict[str, float]:
    """Normalized tool frequency profile from tool_sequence."""
    seq = _get_tool_sequence(task)
    if not seq:
        return {}
    counts = Counter(seq)
    total = sum(counts.values())
    return {tool: cnt / total for tool, cnt in counts.items()}


def _tool_profile_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse tool-profile dicts."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


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
    # Layer 2: Tool-sequence + intent detection
    # ------------------------------------------------------------------

    def _detect_repetitive_patterns(
        self, tasks: list[dict], existing_scheduled: set[str],
    ) -> list[HabitPattern]:
        """Find groups of tasks with similar tool-call profiles.

        Only considers tasks that have a tool_sequence and whose intent
        is classified as repeatable.
        """
        # Step 1: Filter — repeatable tasks with tool_sequence
        candidates: list[tuple[dict, dict[str, float], str]] = []
        for task in tasks:
            seq = _get_tool_sequence(task)
            if not seq:
                continue
            intent, repeatable = classify_task_intent(task)
            if not repeatable:
                continue
            profile = _tool_profile(task)
            candidates.append((task, profile, intent))

        if not candidates:
            return []

        # Step 2: Single-pass clustering by tool profile cosine >= 0.7
        clusters: list[list[tuple[dict, dict[str, float], str]]] = []
        for item in candidates:
            task, profile, intent = item
            placed = False
            for cluster in clusters:
                rep_task, rep_profile, rep_intent = cluster[0]
                tool_sim = _tool_profile_cosine(profile, rep_profile)
                if tool_sim >= 0.7:
                    content_sim = self._content_similarity(task, rep_task)
                    combined = tool_sim * 0.6 + content_sim * 0.4
                    if combined >= 0.6:
                        cluster.append(item)
                        placed = True
                        break
            if not placed:
                clusters.append([item])

        # Step 3 & 4: Filter and generate patterns
        patterns: list[HabitPattern] = []
        for cluster in clusters:
            if len(cluster) < self._l2_min_occurrences:
                continue

            cluster_tasks = [t for t, _, _ in cluster]
            if not self._check_time_spread(cluster_tasks, self._l2_min_days):
                continue

            # Build pattern key from intent + top tools
            intent_votes: Counter[str] = Counter()
            tool_counts: Counter[str] = Counter()
            for _, profile, intent in cluster:
                intent_votes[intent] += 1
                for tool in profile:
                    tool_counts[tool] += 1
            main_intent = intent_votes.most_common(1)[0][0]
            top_tools = "+".join(t for t, _ in tool_counts.most_common(2))
            pattern_key = f"tool_pattern:{main_intent}:{top_tools}"

            # Skip if already scheduled
            if any(top_tools in s for s in existing_scheduled):
                continue

            sample = cluster_tasks[0].get("content", "")[:100]
            cron = self._detect_time_pattern(cluster_tasks)

            patterns.append(HabitPattern(
                pattern_type="repetitive",
                pattern_key=pattern_key,
                count=len(cluster),
                task_summary=f"执行{main_intent}类任务({top_tools})",
                suggested_cron=cron,
                sample_task=sample,
                all_samples=[
                    _strip_context_prefix(t.get("content", ""))[:80]
                    for t in cluster_tasks[:5]
                ],
            ))

        return patterns

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

    def _content_similarity(self, task_a: dict, task_b: dict) -> float:
        """Content similarity: prefer embedding cosine, fallback to Jaccard."""
        emb_a = task_a.get("embedding")
        emb_b = task_b.get("embedding")
        if emb_a and emb_b and isinstance(emb_a, list) and isinstance(emb_b, list):
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            mag_a = math.sqrt(sum(a * a for a in emb_a))
            mag_b = math.sqrt(sum(b * b for b in emb_b))
            if mag_a > 0 and mag_b > 0:
                return dot / (mag_a * mag_b)
        # Fallback: keyword Jaccard
        kw_a = self._simple_keywords(task_a.get("content", ""))
        kw_b = self._simple_keywords(task_b.get("content", ""))
        return self._jaccard(kw_a, kw_b)

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
