"""Tests for ring1.habit_detector."""

from __future__ import annotations

import json
import queue
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ring1.habit_detector import (
    HabitDetector,
    HabitPattern,
    _JACCARD_THRESHOLD,
    _MIN_OCCURRENCES,
    _PROPOSE_COOLDOWN_SEC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    content: str,
    skills_used: list[str] | None = None,
    keywords: str = "",
    timestamp: str | None = None,
) -> dict:
    """Create a mock task memory entry."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    meta = {}
    if skills_used is not None:
        meta["skills_used"] = skills_used
    return {
        "content": content,
        "metadata": meta,
        "keywords": keywords,
        "timestamp": timestamp,
        "generation": 1,
        "entry_type": "task",
    }


def _make_memory_store(tasks: list[dict]) -> MagicMock:
    """Create a mock MemoryStore that returns *tasks* for get_by_type('task')."""
    store = MagicMock()
    store.get_by_type.return_value = tasks
    return store


def _make_scheduled_store(enabled: list[dict] | None = None) -> MagicMock:
    store = MagicMock()
    store.get_enabled.return_value = enabled or []
    store.add.return_value = "sched-test123"
    return store


# ---------------------------------------------------------------------------
# Skill reuse detection
# ---------------------------------------------------------------------------

class TestSkillReuse:
    def test_skill_reuse_detected(self):
        """5 tasks using news_summary -> at least 1 skill_reuse pattern."""
        tasks = [
            _make_task(f"获取新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
        assert skill_patterns[0].pattern_key == "skill:news_summary"
        assert skill_patterns[0].count == 5

    def test_below_threshold_ignored(self):
        """2 tasks using same skill -> no pattern (need >= 3)."""
        tasks = [
            _make_task(f"任务 #{i}", skills_used=["weather"])
            for i in range(2)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_exactly_threshold(self):
        """3 tasks = exactly the minimum -> 1 skill_reuse pattern."""
        tasks = [
            _make_task(f"分析 #{i}", skills_used=["analyzer"])
            for i in range(3)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
        assert skill_patterns[0].count == 3

    def test_existing_schedule_excluded(self):
        """Skill already has a scheduled task -> skip."""
        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        sched = _make_scheduled_store([
            {"name": "daily_news", "task_text": "run_skill news_summary"},
        ])
        detector = HabitDetector(mem, sched)
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_dismissed_excluded(self):
        """Pattern was dismissed -> skip."""
        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        detector.mark_dismissed("skill:news_summary")
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_proposed_cooldown(self):
        """Pattern was proposed < 24h ago -> skip."""
        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        detector.mark_proposed("skill:news_summary")
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_proposed_cooldown_expired(self):
        """Pattern was proposed > 24h ago -> allow again."""
        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        detector._proposed["skill:news_summary"] = time.time() - _PROPOSE_COOLDOWN_SEC - 1
        patterns = detector.detect()
        assert len(patterns) == 1

    def test_multiple_skills(self):
        """Multiple skills each used enough -> multiple patterns."""
        tasks = []
        for i in range(4):
            tasks.append(_make_task(f"新闻 #{i}", skills_used=["news"]))
        for i in range(3):
            tasks.append(_make_task(f"天气 #{i}", skills_used=["weather"]))
        for i in range(2):
            tasks.append(_make_task(f"备忘 #{i}", skills_used=["memo"]))
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        # news (4x) and weather (3x) qualify; memo (2x) does not.
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 2
        keys = {p.pattern_key for p in skill_patterns}
        assert "skill:news" in keys
        assert "skill:weather" in keys

    def test_metadata_as_string(self):
        """metadata stored as JSON string should still work."""
        tasks = [
            {
                "content": f"任务 #{i}",
                "metadata": json.dumps({"skills_used": ["parser"]}),
                "keywords": "",
                "timestamp": datetime.now().isoformat(),
                "generation": 1,
            }
            for i in range(4)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
        assert skill_patterns[0].pattern_key == "skill:parser"


# ---------------------------------------------------------------------------
# Content cluster detection
# ---------------------------------------------------------------------------

class TestContentCluster:
    def test_content_cluster_detected(self):
        """3 tasks with overlapping keywords -> 1 cluster pattern."""
        tasks = [
            _make_task("分析销售数据报表", keywords="分析 销售 数据 报表"),
            _make_task("销售数据趋势", keywords="销售 数据 趋势"),
            _make_task("查看销售数据", keywords="查看 销售 数据"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        cluster_patterns = [p for p in patterns if p.pattern_type == "content_cluster"]
        assert len(cluster_patterns) >= 1
        # Should contain "销售" and "数据" in the key.
        p = cluster_patterns[0]
        assert "销售" in p.pattern_key or "数据" in p.pattern_key

    def test_no_cluster_low_overlap(self):
        """Tasks with very different keywords -> no cluster."""
        tasks = [
            _make_task("分析数据", keywords="analysis data report"),
            _make_task("查看天气", keywords="weather forecast"),
            _make_task("写代码", keywords="python code develop"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        cluster_patterns = [p for p in patterns if p.pattern_type == "content_cluster"]
        assert len(cluster_patterns) == 0

    def test_cluster_from_content_fallback(self):
        """When keywords field is empty, extract from content."""
        tasks = [
            _make_task("帮我分析销售数据", keywords=""),
            _make_task("分析一下销售数据", keywords=""),
            _make_task("销售数据分析报告", keywords=""),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        cluster_patterns = [p for p in patterns if p.pattern_type == "content_cluster"]
        assert len(cluster_patterns) >= 1


# ---------------------------------------------------------------------------
# Time pattern detection
# ---------------------------------------------------------------------------

class TestTimePattern:
    def test_time_pattern_cron(self):
        """4/5 tasks at 9:00-11:00 -> suggested_cron with hour 9 or 10."""
        base = datetime(2026, 2, 15, 9, 30)
        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news"],
                       timestamp=(base + timedelta(days=i, minutes=i * 10)).isoformat())
            for i in range(4)
        ]
        # One task at a different time.
        tasks.append(
            _make_task("新闻 #4", skills_used=["news"],
                       timestamp=datetime(2026, 2, 20, 16, 0).isoformat())
        )
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
        p = skill_patterns[0]
        assert p.suggested_cron != ""
        # The hour should be 9 (since 4/5 tasks are in the 9-11 window).
        assert "9" in p.suggested_cron

    def test_no_time_pattern(self):
        """Tasks spread across all hours -> no suggested_cron."""
        tasks = [
            _make_task(f"任务 #{i}", skills_used=["tool"],
                       timestamp=datetime(2026, 2, 15, i * 5 % 24, 0).isoformat())
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        if skill_patterns:
            assert skill_patterns[0].suggested_cron == ""


# ---------------------------------------------------------------------------
# mark_proposed / mark_dismissed
# ---------------------------------------------------------------------------

class TestMarking:
    def test_mark_proposed(self):
        detector = HabitDetector(MagicMock())
        detector.mark_proposed("skill:test")
        assert "skill:test" in detector._proposed
        assert not detector._should_propose("skill:test")

    def test_mark_dismissed(self):
        detector = HabitDetector(MagicMock())
        detector.mark_dismissed("skill:test")
        assert "skill:test" in detector._dismissed
        assert not detector._should_propose("skill:test")


# ---------------------------------------------------------------------------
# Integration: executor -> state queue -> bot callback
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_executor_habit_to_queue(self):
        """Executor pushes habit proposals into state.habit_proposals queue."""
        from ring1.telegram_bot import SentinelState

        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        state = SentinelState()
        state.memory_store = mem

        detector = HabitDetector(mem)
        patterns = detector.detect()
        assert len(patterns) >= 1

        # Simulate _propose_habit putting into queue.
        pattern = patterns[0]
        text = f"发现重复模式: {pattern.pattern_key}"
        buttons = [[{"text": "OK", "callback_data": f"habit:schedule:{pattern.pattern_key}:0 9 * * *"}]]
        state.habit_proposals.put((text, buttons))

        assert not state.habit_proposals.empty()
        queued_text, queued_buttons = state.habit_proposals.get_nowait()
        assert "habit:schedule:" in queued_buttons[0][0]["callback_data"]

    def test_habit_schedule_callback(self):
        """Bot _handle_callback creates a scheduled task on habit:schedule:..."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store

        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:schedule:skill:news_summary:0 9 * * *")
        assert "已创建定时任务" in reply
        sched_store.add.assert_called_once()
        call_kwargs = sched_store.add.call_args
        assert call_kwargs[1]["cron_expr"] == "0 9 * * *" or call_kwargs[0][2] == "0 9 * * *"

    def test_habit_dismiss_callback(self):
        """Bot _handle_callback records dismissal on habit:dismiss:..."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:dismiss:skill:news_summary")
        assert "不再提醒" in reply
        assert "skill:news_summary" in state._habit_dismissed

    def test_habit_dismiss_synced_to_detector(self):
        """Dismissed patterns in state._habit_dismissed block future detection."""
        from ring1.telegram_bot import SentinelState

        tasks = [
            _make_task(f"新闻 #{i}", skills_used=["news_summary"])
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        state = SentinelState()
        state._habit_dismissed.add("skill:news_summary")

        detector = HabitDetector(mem)
        detector._dismissed = state._habit_dismissed
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_empty_tasks_no_crash(self):
        """Empty task history should not crash."""
        mem = _make_memory_store([])
        detector = HabitDetector(mem)
        patterns = detector.detect()
        assert patterns == []

    def test_memory_store_error_handled(self):
        """MemoryStore raises -> detect returns empty list."""
        mem = MagicMock()
        mem.get_by_type.side_effect = RuntimeError("db locked")
        detector = HabitDetector(mem)
        patterns = detector.detect()
        assert patterns == []


# ---------------------------------------------------------------------------
# Importance filtering
# ---------------------------------------------------------------------------

class TestImportanceFiltering:
    def test_low_importance_tasks_filtered(self):
        """Tasks with importance < 0.5 should not be considered."""
        tasks = [
            {**_make_task(f"commit #{i}", skills_used=["git"]),
             "importance": 0.3}
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_high_importance_tasks_detected(self):
        """Tasks with importance >= 0.5 should be detected."""
        tasks = [
            {**_make_task(f"分析数据 #{i}", skills_used=["analyzer"]),
             "importance": 0.7}
            for i in range(4)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1

    def test_mixed_importance_only_high_counted(self):
        """Only tasks above threshold contribute to pattern count."""
        tasks = [
            {**_make_task("分析数据", skills_used=["analyzer"]), "importance": 0.7},
            {**_make_task("分析数据", skills_used=["analyzer"]), "importance": 0.7},
            {**_make_task("分析数据", skills_used=["analyzer"]), "importance": 0.7},
            {**_make_task("commit", skills_used=["analyzer"]), "importance": 0.3},
            {**_make_task("ok", skills_used=["analyzer"]), "importance": 0.2},
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
        assert skill_patterns[0].count == 3  # only the 3 high-importance ones

    def test_default_importance_passes(self):
        """Tasks without importance field default to 0.5 and pass the filter."""
        tasks = [
            _make_task(f"查询天气 #{i}", skills_used=["weather"])
            for i in range(4)
        ]
        # No "importance" key in these tasks — should default to 0.5.
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        skill_patterns = [p for p in patterns if p.pattern_type == "skill_reuse"]
        assert len(skill_patterns) == 1
