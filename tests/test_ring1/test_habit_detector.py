"""Tests for ring1.habit_detector (two-layer: template + repetitive)."""

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
    _PROPOSE_COOLDOWN_SEC,
    load_templates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    content: str,
    keywords: str = "",
    timestamp: str | None = None,
    importance: float = 0.7,
) -> dict:
    """Create a mock task memory entry."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    return {
        "content": content,
        "metadata": {},
        "keywords": keywords,
        "timestamp": timestamp,
        "generation": 1,
        "entry_type": "task",
        "importance": importance,
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


FLIGHT_TEMPLATE = {
    "id": "flight_price_tracker",
    "name": "航班价格追踪",
    "category": "price_monitoring",
    "keywords": ["航班", "机票", "flight", "票价", "飞", "航空"],
    "regex_patterns": ["(从|自).+(到|飞).+", "flight.+from.+to"],
    "task_type": "realtime",
    "default_cron": "*/10 * * * *",
    "auto_stop_hours": 2,
    "min_hits": 2,
    "window_hours": 4,
}

NEWS_TEMPLATE = {
    "id": "daily_news_digest",
    "name": "每日新闻摘要",
    "category": "news_information",
    "keywords": ["新闻", "头条", "news", "今天发生", "时事", "资讯", "摘要"],
    "regex_patterns": ["今天.*新闻", "有什么.*头条", "latest.*news"],
    "task_type": "periodic",
    "default_cron": "0 7 * * *",
    "auto_stop_hours": 0,
    "min_hits": 2,
    "window_hours": 72,
}

MEETING_TEMPLATE = {
    "id": "meeting_summary",
    "name": "会议纪要生成",
    "category": "communication",
    "keywords": ["会议", "meeting", "纪要", "总结", "记录"],
    "regex_patterns": ["会议.*总结", "meeting.*notes", "整理.*纪要"],
    "task_type": "on_demand",
    "default_cron": "",
    "auto_stop_hours": 0,
    "min_hits": 3,
    "window_hours": 168,
}


# ---------------------------------------------------------------------------
# Template matching (Layer 1)
# ---------------------------------------------------------------------------

class TestTemplateMatching:
    def test_keyword_match(self):
        """Keyword hits above min_hits → detected."""
        tasks = [
            _make_task("帮我查机票价格"),
            _make_task("查一下航班信息"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == "template"
        assert p.template_name == "flight_price_tracker"
        assert p.count == 2

    def test_regex_match(self):
        """Regex hits (even if keywords miss) → detected."""
        tasks = [
            _make_task("从成都到新加坡多少钱"),
            _make_task("从北京飞上海的价格"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "template"

    def test_no_match_below_threshold(self):
        """Only 1 hit (min_hits=2) → no pattern."""
        tasks = [_make_task("查一下机票")]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_no_match_outside_window(self):
        """Hits outside window_hours → not counted."""
        old_ts = (datetime.now() - timedelta(hours=10)).isoformat()
        tasks = [
            _make_task("查机票", timestamp=old_ts),
            _make_task("查航班", timestamp=old_ts),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_auto_stop_hours_propagated(self):
        """Realtime template's auto_stop_hours is in the HabitPattern."""
        tasks = [
            _make_task("帮我查机票价格"),
            _make_task("查一下航班信息"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].auto_stop_hours == 2

    def test_on_demand_skipped(self):
        """on_demand templates don't generate proposals."""
        tasks = [
            _make_task("帮我整理会议纪要"),
            _make_task("会议总结一下"),
            _make_task("整理会议记录"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[MEETING_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_skip_already_scheduled(self):
        """Template name already in scheduled tasks → skip."""
        tasks = [
            _make_task("今天有什么新闻"),
            _make_task("给我看看最新新闻"),
        ]
        mem = _make_memory_store(tasks)
        sched = _make_scheduled_store([
            {"name": "auto_template_daily_news_digest", "task_text": "获取新闻"},
        ])
        detector = HabitDetector(mem, sched, templates=[NEWS_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 0

    def test_multiple_templates_match(self):
        """Multiple templates can match simultaneously."""
        tasks = [
            _make_task("查一下机票"),
            _make_task("帮我查航班"),
            _make_task("今天有什么新闻"),
            _make_task("最新头条是什么"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE, NEWS_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 2
        types = {p.template_name for p in patterns}
        assert "flight_price_tracker" in types
        assert "daily_news_digest" in types

    def test_default_cron_used(self):
        """Template's default_cron is used as suggested_cron."""
        tasks = [
            _make_task("帮我查机票"),
            _make_task("看看航班价格"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert patterns[0].suggested_cron == "*/10 * * * *"


# ---------------------------------------------------------------------------
# Repetitive detection (Layer 2)
# ---------------------------------------------------------------------------

class TestRepetitiveDetection:
    def test_detect_with_high_threshold(self):
        """5+ occurrences, 3+ days, LLM YES → detected."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task(
                "分析销售数据报表",
                keywords="分析 销售 数据 报表",
                timestamp=(base + timedelta(days=i)).isoformat(),
            )
            for i in range(6)
        ]
        mem = _make_memory_store(tasks)
        llm = MagicMock()
        llm.send_message.return_value = "YES, this is a useful pattern."
        detector = HabitDetector(mem, llm_client=llm)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) >= 1
        assert rep[0].count >= 5

    def test_below_occurrence_threshold(self):
        """Only 3 occurrences (need 5) → no repetitive pattern."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task(
                "分析销售数据",
                keywords="分析 销售 数据",
                timestamp=(base + timedelta(days=i)).isoformat(),
            )
            for i in range(3)
        ]
        mem = _make_memory_store(tasks)
        llm = MagicMock()
        llm.send_message.return_value = "YES"
        detector = HabitDetector(mem, llm_client=llm)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_insufficient_time_spread(self):
        """5 occurrences but same day → no pattern (need 3 days)."""
        ts = datetime(2026, 2, 15, 9, 0).isoformat()
        tasks = [
            _make_task("分析销售数据", keywords="分析 销售 数据", timestamp=ts)
            for _ in range(6)
        ]
        mem = _make_memory_store(tasks)
        llm = MagicMock()
        llm.send_message.return_value = "YES"
        detector = HabitDetector(mem, llm_client=llm)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_llm_rejects(self):
        """LLM says NO → no pattern."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task(
                "分析销售数据",
                keywords="分析 销售 数据",
                timestamp=(base + timedelta(days=i)).isoformat(),
            )
            for i in range(6)
        ]
        mem = _make_memory_store(tasks)
        llm = MagicMock()
        llm.send_message.return_value = "NO, this is not useful."
        detector = HabitDetector(mem, llm_client=llm)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_no_llm_conservative(self):
        """No LLM client → no repetitive patterns (conservative)."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task(
                "分析销售数据",
                keywords="分析 销售 数据",
                timestamp=(base + timedelta(days=i)).isoformat(),
            )
            for i in range(6)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, llm_client=None)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_l1_takes_precedence(self):
        """If L1 finds something, L2 is not run."""
        tasks = [
            _make_task("查一下机票"),
            _make_task("帮我查航班"),
        ]
        mem = _make_memory_store(tasks)
        llm = MagicMock()
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE], llm_client=llm)
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "template"
        # LLM should NOT have been called (L2 skipped).
        llm.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_simple_keywords_english(self):
        kw = HabitDetector._simple_keywords("Check weather forecast today")
        assert "check" in kw
        assert "weather" in kw
        assert "forecast" in kw
        assert "today" in kw

    def test_simple_keywords_chinese(self):
        kw = HabitDetector._simple_keywords("分析销售数据")
        assert "分析" in kw
        assert "销售" in kw
        assert "数据" in kw

    def test_jaccard_identical(self):
        assert HabitDetector._jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_empty(self):
        assert HabitDetector._jaccard(set(), {"a"}) == 0.0

    def test_jaccard_partial(self):
        j = HabitDetector._jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.3 < j < 0.6  # 2/4 = 0.5

    def test_extract_hour(self):
        assert HabitDetector._extract_hour("2026-02-15T09:30:00") == 9
        assert HabitDetector._extract_hour("") is None
        assert HabitDetector._extract_hour("invalid") is None

    def test_time_pattern_cron(self):
        """4/5 tasks at 9:00-10:30 → suggested_cron with hour 9."""
        base = datetime(2026, 2, 15, 9, 30)
        tasks = [
            _make_task(f"task #{i}",
                       timestamp=(base + timedelta(days=i, minutes=i * 10)).isoformat())
            for i in range(4)
        ]
        tasks.append(
            _make_task("task #4", timestamp=datetime(2026, 2, 20, 16, 0).isoformat())
        )
        detector = HabitDetector(MagicMock())
        cron = detector._detect_time_pattern(tasks)
        assert cron != ""
        assert "9" in cron

    def test_no_time_pattern(self):
        """Tasks spread across all hours → no cron."""
        tasks = [
            _make_task(f"task #{i}",
                       timestamp=datetime(2026, 2, 15, i * 5 % 24, 0).isoformat())
            for i in range(5)
        ]
        detector = HabitDetector(MagicMock())
        cron = detector._detect_time_pattern(tasks)
        assert cron == ""


# ---------------------------------------------------------------------------
# Cooldown / dismissal
# ---------------------------------------------------------------------------

class TestCooldownDismissal:
    def test_mark_proposed(self):
        detector = HabitDetector(MagicMock())
        detector.mark_proposed("template:test")
        assert not detector._should_propose("template:test")

    def test_mark_dismissed(self):
        detector = HabitDetector(MagicMock())
        detector.mark_dismissed("template:test")
        assert not detector._should_propose("template:test")

    def test_proposed_cooldown_expired(self):
        detector = HabitDetector(MagicMock())
        detector._proposed["template:test"] = time.time() - _PROPOSE_COOLDOWN_SEC - 1
        assert detector._should_propose("template:test")

    def test_dismissed_permanent(self):
        detector = HabitDetector(MagicMock())
        detector.mark_dismissed("template:test")
        # Even after a long time, dismissed stays dismissed.
        assert not detector._should_propose("template:test")


# ---------------------------------------------------------------------------
# load_templates
# ---------------------------------------------------------------------------

class TestLoadTemplates:
    def test_load_valid(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text(json.dumps({"version": "1.0", "templates": [{"id": "test"}]}))
        templates = load_templates(path)
        assert len(templates) == 1
        assert templates[0]["id"] == "test"

    def test_load_missing_file(self, tmp_path):
        templates = load_templates(tmp_path / "nonexistent.json")
        assert templates == []

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        templates = load_templates(path)
        assert templates == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tasks_no_crash(self):
        mem = _make_memory_store([])
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        assert detector.detect() == []

    def test_memory_store_error_handled(self):
        mem = MagicMock()
        mem.get_by_type.side_effect = RuntimeError("db locked")
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        assert detector.detect() == []

    def test_low_importance_filtered(self):
        tasks = [
            _make_task("查机票", importance=0.3),
            _make_task("查航班", importance=0.3),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        assert detector.detect() == []

    def test_no_templates_no_l1(self):
        """Without templates, L1 returns nothing, falls through to L2."""
        tasks = [_make_task("hello"), _make_task("world")]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[])
        patterns = detector.detect()
        # L2 also won't fire (not enough tasks, no LLM)
        assert patterns == []


# ---------------------------------------------------------------------------
# Integration: executor -> state queue -> bot callback
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_habit_schedule_callback(self):
        """Bot _handle_callback creates a scheduled task on habit:schedule:..."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store

        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:schedule:template:flight_price_tracker:*/10 * * * *|2")
        assert "已创建定时任务" in reply
        assert "2 小时后自动停止" in reply
        sched_store.add.assert_called_once()
        call_kw = sched_store.add.call_args
        # Check expires_at was passed
        assert call_kw[1].get("expires_at") is not None

    def test_habit_schedule_no_auto_stop(self):
        """Callback without auto_stop_hours → no expires_at."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store

        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:schedule:template:daily_news_digest:0 7 * * *")
        assert "已创建定时任务" in reply
        assert "自动停止" not in reply
        call_kw = sched_store.add.call_args
        assert call_kw[1].get("expires_at") is None

    def test_habit_dismiss_callback(self):
        """Bot _handle_callback records dismissal on habit:dismiss:..."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:dismiss:template:flight_price_tracker")
        assert "不再提醒" in reply
        assert "template:flight_price_tracker" in state._habit_dismissed

    def test_habit_dismiss_synced_to_detector(self):
        """Dismissed patterns in state._habit_dismissed block future detection."""
        from ring1.telegram_bot import SentinelState

        tasks = [
            _make_task("查机票"),
            _make_task("查航班"),
        ]
        mem = _make_memory_store(tasks)
        state = SentinelState()
        state._habit_dismissed.add("template:flight_price_tracker")

        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        detector._dismissed = state._habit_dismissed
        patterns = detector.detect()
        assert len(patterns) == 0
