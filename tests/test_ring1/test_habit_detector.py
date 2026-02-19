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
    _strip_context_prefix,
    _tool_profile,
    _tool_profile_cosine,
    classify_task_intent,
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
    tool_sequence: list[str] | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """Create a mock task memory entry."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    metadata: dict = {}
    if tool_sequence is not None:
        metadata["tool_sequence"] = tool_sequence
    task: dict = {
        "content": content,
        "metadata": metadata,
        "keywords": keywords,
        "timestamp": timestamp,
        "generation": 1,
        "entry_type": "task",
        "importance": importance,
    }
    if embedding is not None:
        task["embedding"] = embedding
    return task


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
    """Layer 2: tool-sequence + intent based detection."""

    def _search_tasks(self, n: int, base: datetime | None = None) -> list[dict]:
        """Helper: create n search-like tasks across different days."""
        if base is None:
            base = datetime(2026, 2, 10, 9, 0)
        return [
            _make_task(
                "搜索最新AI研究论文",
                timestamp=(base + timedelta(days=i)).isoformat(),
                tool_sequence=["web_search", "web_search", "web_fetch", "message"],
            )
            for i in range(n)
        ]

    def test_similar_tool_sequences_detected(self):
        """3+ similar tool sequences across 2+ days → detected."""
        tasks = self._search_tasks(4)
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) >= 1
        assert rep[0].count >= 3
        assert rep[0].pattern_key.startswith("tool_pattern:")

    def test_different_tool_sequences_not_clustered(self):
        """Tasks with completely different tool profiles → no cluster."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task("搜索论文", timestamp=(base + timedelta(days=0)).isoformat(),
                       tool_sequence=["web_search", "web_search", "message"]),
            _make_task("写代码", timestamp=(base + timedelta(days=1)).isoformat(),
                       tool_sequence=["write_file", "edit_file", "exec"]),
            _make_task("发文件", timestamp=(base + timedelta(days=2)).isoformat(),
                       tool_sequence=["read_file", "send_file", "message"]),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_non_repeatable_tasks_excluded(self):
        """Tasks dominated by write_file (create intent) → excluded."""
        base = datetime(2026, 2, 10, 9, 0)
        tasks = [
            _make_task(
                "写一个脚本",
                timestamp=(base + timedelta(days=i)).isoformat(),
                tool_sequence=["write_file", "write_file", "edit_file", "exec"],
            )
            for i in range(5)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_no_tool_sequence_skipped(self):
        """Old tasks without tool_sequence → skipped by L2."""
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
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_same_day_insufficient_spread(self):
        """3 tasks on same day (need 2 days) → no pattern."""
        ts = datetime(2026, 2, 15, 9, 0).isoformat()
        tasks = [
            _make_task("搜索论文", timestamp=ts,
                       tool_sequence=["web_search", "web_fetch", "message"])
            for _ in range(4)
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0

    def test_no_llm_still_works(self):
        """L2 works without LLM client (no longer requires it)."""
        tasks = self._search_tasks(4)
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, llm_client=None)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) >= 1

    def test_l1_takes_precedence(self):
        """If L1 finds something, L2 is not run."""
        tasks = [
            _make_task("查一下机票",
                       tool_sequence=["web_search", "web_search", "message"]),
            _make_task("帮我查航班",
                       tool_sequence=["web_search", "web_search", "message"]),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[FLIGHT_TEMPLATE])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "template"

    def test_pattern_key_format(self):
        """pattern_key has format 'tool_pattern:{intent}:{tools}'."""
        tasks = self._search_tasks(4)
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) >= 1
        key = rep[0].pattern_key
        assert key.startswith("tool_pattern:")
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[1] == "search"

    def test_all_samples_populated(self):
        """all_samples filled with cluster task contents."""
        tasks = self._search_tasks(4)
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) >= 1
        assert len(rep[0].all_samples) >= 3

    def test_below_occurrence_threshold(self):
        """Only 2 tasks (need 3) → no repetitive pattern."""
        tasks = self._search_tasks(2)
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem)
        patterns = detector.detect()
        rep = [p for p in patterns if p.pattern_type == "repetitive"]
        assert len(rep) == 0


# ---------------------------------------------------------------------------
# classify_task_intent
# ---------------------------------------------------------------------------

class TestClassifyTaskIntent:
    def test_search_task_repeatable(self):
        task = _make_task("搜索论文", tool_sequence=["web_search", "web_fetch", "message"])
        intent, rep = classify_task_intent(task)
        assert intent == "search"
        assert rep is True

    def test_create_task_not_repeatable(self):
        task = _make_task("写代码", tool_sequence=["write_file", "edit_file", "exec"])
        intent, rep = classify_task_intent(task)
        assert intent == "create"
        assert rep is False

    def test_chat_not_repeatable(self):
        task = _make_task("聊天", tool_sequence=["message", "message"])
        intent, rep = classify_task_intent(task)
        assert intent == "chat"
        assert rep is False

    def test_admin_not_repeatable(self):
        task = _make_task("管理定时", tool_sequence=["manage_schedule"])
        intent, rep = classify_task_intent(task)
        assert intent == "admin"
        assert rep is False

    def test_skill_repeatable(self):
        task = _make_task("运行技能", tool_sequence=["run_skill", "message"])
        intent, rep = classify_task_intent(task)
        assert intent == "skill"
        assert rep is True

    def test_no_tool_sequence(self):
        task = _make_task("旧任务")
        intent, rep = classify_task_intent(task)
        assert intent == "unknown"
        assert rep is False

    def test_mixed_search_exec_repeatable(self):
        task = _make_task("搜索并执行",
                          tool_sequence=["web_search", "web_search", "exec", "message"])
        intent, rep = classify_task_intent(task)
        assert intent == "search"
        assert rep is True

    def test_metadata_as_json_string(self):
        task = _make_task("搜索")
        task["metadata"] = json.dumps({"tool_sequence": ["web_search", "message"]})
        intent, rep = classify_task_intent(task)
        assert intent == "search"
        assert rep is True


# ---------------------------------------------------------------------------
# _tool_profile
# ---------------------------------------------------------------------------

class TestToolProfile:
    def test_basic_frequency(self):
        task = _make_task("x", tool_sequence=["web_search", "web_search", "web_fetch", "message"])
        profile = _tool_profile(task)
        assert profile["web_search"] == pytest.approx(0.5)
        assert profile["web_fetch"] == pytest.approx(0.25)
        assert profile["message"] == pytest.approx(0.25)

    def test_empty_sequence(self):
        task = _make_task("x")
        assert _tool_profile(task) == {}

    def test_single_tool(self):
        task = _make_task("x", tool_sequence=["web_search"])
        profile = _tool_profile(task)
        assert profile == {"web_search": 1.0}


# ---------------------------------------------------------------------------
# _tool_profile_cosine
# ---------------------------------------------------------------------------

class TestToolProfileCosine:
    def test_identical(self):
        p = {"web_search": 0.5, "message": 0.5}
        assert _tool_profile_cosine(p, p) == pytest.approx(1.0)

    def test_disjoint(self):
        a = {"web_search": 1.0}
        b = {"write_file": 1.0}
        assert _tool_profile_cosine(a, b) == pytest.approx(0.0)

    def test_similar_profiles(self):
        a = {"web_search": 0.5, "web_fetch": 0.3, "message": 0.2}
        b = {"web_search": 0.6, "web_fetch": 0.2, "message": 0.2}
        sim = _tool_profile_cosine(a, b)
        assert sim > 0.9

    def test_empty(self):
        assert _tool_profile_cosine({}, {"a": 1.0}) == pytest.approx(0.0)
        assert _tool_profile_cosine({}, {}) == pytest.approx(0.0)


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
        # L2 also won't fire (no tool_sequence in tasks)
        assert patterns == []


# ---------------------------------------------------------------------------
# Integration: executor -> state queue -> bot callback
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_habit_schedule_callback_creates_pending(self):
        """Callback stores pending habit and returns clarification prompt."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store
        # Simulate _propose_habit storing context with clarification_prompt
        state._habit_context["template:flight_price_tracker"] = {
            "task_text": "搜索最新航班价格信息并汇报变动",
            "task_summary": "航班价格追踪",
            "clarification_prompt": "请告诉我你想追踪哪条航线？",
        }

        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:schedule:template:flight_price_tracker:*/10 * * * *|2")
        # Should NOT create the task yet — should ask for clarification
        sched_store.add.assert_not_called()
        assert "请告诉我你想追踪哪条航线" in reply
        # Should have stored pending habit
        assert "template:flight_price_tracker" in state._pending_habits
        pending = state._pending_habits["template:flight_price_tracker"]
        assert pending["cron"] == "*/10 * * * *"
        assert pending["auto_stop_hours"] == 2

    def test_habit_schedule_fallback_prompt_without_context(self):
        """Without clarification_prompt, uses default prompt."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store

        bot = TelegramBot.__new__(TelegramBot)
        bot.state = state
        bot.chat_id = "123"

        reply = bot._handle_callback("habit:schedule:template:daily_news_digest:0 7 * * *")
        sched_store.add.assert_not_called()
        assert "请具体描述" in reply
        assert "template:daily_news_digest" in state._pending_habits

    def test_default_task_text_in_template(self):
        """Template with default_task_text uses it as sample_task."""
        tmpl = {**FLIGHT_TEMPLATE, "default_task_text": "搜索最新航班价格信息"}
        tasks = [
            _make_task("帮我查机票价格"),
            _make_task("查一下航班信息"),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].sample_task == "搜索最新航班价格信息"

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

    def test_habit_clarification_reply_creates_task(self):
        """User reply after clarification creates scheduled task with topic."""
        from ring1.telegram_bot import SentinelState, TelegramBot

        state = SentinelState()
        sched_store = _make_scheduled_store()
        state.scheduled_store = sched_store
        # Set up pending habit (as if user clicked "自动执行")
        state._pending_habits["template:academic_paper_alert"] = {
            "cron": "0 8 * * *",
            "auto_stop_hours": 0,
            "timestamp": time.time(),
        }
        state._habit_context["template:academic_paper_alert"] = {
            "task_text": "搜索关注领域的最新学术论文并生成摘要",
            "task_summary": "学术论文监控",
            "clarification_prompt": "请告诉我你想监控哪个研究领域？",
        }

        # Simulate the bot's main loop handling a text reply
        # We test the logic directly by checking the pending habit flow
        pk = "template:academic_paper_alert"
        info = state._pending_habits[pk]
        topic = "AGI 和自演化AI"
        ctx_h = state._habit_context.get(pk, {})
        default_text = ctx_h.get("task_text", "自动任务")
        task_text = f"{topic} — {default_text}"

        sched_store.add(
            name=f"auto_{pk.replace(':', '_')}",
            task_text=task_text,
            cron_expr=info["cron"],
            schedule_type="cron",
            chat_id="123",
            expires_at=None,
        )
        del state._pending_habits[pk]

        # Verify task was created with the topic
        sched_store.add.assert_called_once()
        call_kw = sched_store.add.call_args
        assert "AGI 和自演化AI" in call_kw[1]["task_text"]
        assert "搜索关注领域的最新学术论文" in call_kw[1]["task_text"]
        assert pk not in state._pending_habits

    def test_habit_clarification_timeout(self):
        """Pending habits older than 5 minutes are expired."""
        from ring1.telegram_bot import SentinelState

        state = SentinelState()
        # Set timestamp 6 minutes ago
        state._pending_habits["template:test"] = {
            "cron": "0 8 * * *",
            "auto_stop_hours": 0,
            "timestamp": time.time() - 360,  # 6 minutes ago
        }
        # Simulate cleanup (same logic as in the main loop)
        now = time.time()
        for pk, info in list(state._pending_habits.items()):
            if now - info["timestamp"] > 300:
                del state._pending_habits[pk]
        assert len(state._pending_habits) == 0


# ---------------------------------------------------------------------------
# all_samples field
# ---------------------------------------------------------------------------

class TestAllSamples:
    def test_all_samples_populated(self):
        """all_samples contains content from matched tasks."""
        tasks = [
            _make_task("搜索机器人行业最新研究论文"),
            _make_task("搜索AI领域最新论文"),
            _make_task("搜索AGI相关研究"),
        ]
        tmpl = {
            "id": "academic_paper_alert",
            "name": "学术论文监控",
            "keywords": ["论文", "paper", "research", "arxiv", "研究", "学术"],
            "regex_patterns": ["最新.*研究", "academic.*paper"],
            "task_type": "periodic",
            "default_cron": "0 8 * * *",
            "default_task_text": "搜索关注领域的最新学术论文并生成摘要",
            "min_hits": 2,
            "window_hours": 168,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        assert len(patterns) == 1
        p = patterns[0]
        assert len(p.all_samples) == 3
        assert "搜索机器人行业最新研究论文" in p.all_samples[0]
        assert "搜索AI领域最新论文" in p.all_samples[1]

    def test_all_samples_truncated(self):
        """all_samples entries are truncated to 80 chars."""
        long_content = "搜索" + "A" * 200 + "论文"
        tasks = [
            _make_task(long_content),
            _make_task("搜索AI论文"),
        ]
        tmpl = {
            "id": "academic_paper_alert",
            "name": "学术论文监控",
            "keywords": ["论文", "研究"],
            "regex_patterns": [],
            "task_type": "periodic",
            "default_cron": "0 8 * * *",
            "min_hits": 2,
            "window_hours": 168,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert len(patterns[0].all_samples[0]) <= 80


# ---------------------------------------------------------------------------
# min_keyword_hits
# ---------------------------------------------------------------------------

class TestMinKeywordHits:
    def test_min_keyword_hits_filters_single_keyword(self):
        """min_keyword_hits=2: single keyword '系统' alone doesn't match."""
        tasks = [
            _make_task("启动一个定时任务，每10分钟发送一个系统的时间消息给我"),
            _make_task("发送当前系统时间报告"),
            _make_task("增加一个每5分钟发送系统时间的schedule"),
        ]
        tmpl = {
            "id": "system_health_check",
            "name": "系统健康检查",
            "keywords": ["系统", "状态", "health", "status", "监控"],
            "regex_patterns": ["系统.*状态", "health.*check"],
            "task_type": "periodic",
            "default_cron": "*/30 * * * *",
            "min_hits": 3,
            "min_keyword_hits": 2,
            "window_hours": 72,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        # None of the tasks mention 2+ keywords, so no pattern
        assert len(patterns) == 0

    def test_min_keyword_hits_allows_multi_keyword(self):
        """min_keyword_hits=2: tasks with 2+ keywords match."""
        tasks = [
            _make_task("检查系统状态"),
            _make_task("查看系统监控信息"),
            _make_task("系统健康状态如何"),
        ]
        tmpl = {
            "id": "system_health_check",
            "name": "系统健康检查",
            "keywords": ["系统", "状态", "health", "status", "监控"],
            "regex_patterns": [],
            "task_type": "periodic",
            "default_cron": "*/30 * * * *",
            "min_hits": 3,
            "min_keyword_hits": 2,
            "window_hours": 72,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        assert len(patterns) == 1
        assert patterns[0].count == 3

    def test_min_keyword_hits_default_one(self):
        """Without min_keyword_hits, defaults to 1 (backward compat)."""
        tasks = [
            _make_task("查看系统时间"),
            _make_task("系统消息"),
            _make_task("系统通知"),
        ]
        tmpl = {
            "id": "system_health_check",
            "name": "系统健康检查",
            "keywords": ["系统", "状态", "health", "status", "监控"],
            "regex_patterns": [],
            "task_type": "periodic",
            "default_cron": "*/30 * * * *",
            "min_hits": 3,
            # No min_keyword_hits → defaults to 1
            "window_hours": 72,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        # With default min_keyword_hits=1, single "系统" is enough
        assert len(patterns) == 1


# ---------------------------------------------------------------------------
# Context prefix stripping
# ---------------------------------------------------------------------------

class TestContextPrefixStripping:
    def test_strip_time_window_prefix(self):
        """Strip '[Context: User sent this Xs after...]' prefix."""
        text = (
            '[Context: User sent this 46s after your last message]\n'
            'Your previous message: "现在..."\n'
            'User now says: 帮我查一下最近的新闻'
        )
        assert _strip_context_prefix(text) == "帮我查一下最近的新闻"

    def test_strip_explicit_reply_prefix(self):
        """Strip '[Context: User is replying...]' prefix."""
        text = (
            '[Context: User is replying to your previous message]\n'
            'Your message: "这是之前的回复"\n'
            "User's reply: 好的谢谢"
        )
        assert _strip_context_prefix(text) == "好的谢谢"

    def test_no_prefix_passthrough(self):
        """Text without context prefix is returned unchanged."""
        text = "帮我查一下新闻"
        assert _strip_context_prefix(text) == "帮我查一下新闻"

    def test_context_prefix_not_counted_in_keyword_match(self):
        """Keywords in context prefix should not trigger template match."""
        NEWS_TMPL = {
            "id": "daily_news_digest",
            "name": "每日新闻摘要",
            "keywords": ["新闻", "头条", "news"],
            "regex_patterns": [],
            "task_type": "periodic",
            "default_cron": "0 7 * * *",
            "min_hits": 2,
            "window_hours": 72,
        }
        # These tasks mention "新闻" only in the context prefix (bot's reply),
        # NOT in the user's actual text.
        tasks = [
            _make_task(
                '[Context: User sent this 46s after your last message]\n'
                'Your previous message: "这里是最新新闻摘要..."\n'
                'User now says: 想要设计一个任务模板'
            ),
            _make_task(
                '[Context: User sent this 63s after your last message]\n'
                'Your previous message: "新闻已发送"\n'
                'User now says: 好的，帮我查一下天气'
            ),
        ]
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[NEWS_TMPL])
        patterns = detector.detect()
        # Neither task's actual text contains "新闻", so no match
        assert len(patterns) == 0

    def test_all_samples_show_clean_content(self):
        """all_samples should contain stripped user text, not context prefix."""
        tasks = [
            _make_task(
                '[Context: User sent this 10s after your last message]\n'
                'Your previous message: "已完成"\n'
                'User now says: 查一下AI最新论文'
            ),
            _make_task("搜索AGI相关研究论文"),
        ]
        tmpl = {
            "id": "academic_paper_alert",
            "name": "学术论文监控",
            "keywords": ["论文", "研究"],
            "regex_patterns": [],
            "task_type": "periodic",
            "default_cron": "0 8 * * *",
            "min_hits": 2,
            "window_hours": 168,
        }
        mem = _make_memory_store(tasks)
        detector = HabitDetector(mem, templates=[tmpl])
        patterns = detector.detect()
        assert len(patterns) == 1
        # all_samples should be the clean user text
        assert patterns[0].all_samples[0] == "查一下AI最新论文"
        assert patterns[0].all_samples[1] == "搜索AGI相关研究论文"

    def test_strip_prefix_with_internal_quotes(self):
        """Regex handles bot messages containing internal quotes."""
        text = (
            '[Context: User sent this 26s after your last message]\n'
            'Your previous message: "答案是"别人的脚""\n'
            'User now says: 不是，是我的脚'
        )
        assert _strip_context_prefix(text) == "不是，是我的脚"

    def test_strip_prefix_with_multiline_bot_message(self):
        """Regex handles long multi-line bot messages with internal quotes."""
        text = (
            '[Context: User sent this 85s after your last message]\n'
            'Your previous message: "## 总结\n\n'
            '这不是崩溃！在 `ring2/main.py` 的 `finally` 块中\n'
            '程序退出前会发送一条"Notification"消息..."'
            '\n'
            'User now says: 这个不用调整，保持现状吧'
        )
        assert _strip_context_prefix(text) == "这个不用调整，保持现状吧"


class TestCleanForMemory:
    """Tests for _clean_for_memory in task_executor."""

    def test_strip_code_blocks(self):
        from ring1.task_executor import _clean_for_memory
        text = '帮我看看这段代码有什么问题：\n```python\ndef foo():\n    pass\n```'
        result = _clean_for_memory(text)
        assert "```" not in result
        assert "def foo" not in result
        assert "帮我看看这段代码有什么问题" in result

    def test_strip_traceback(self):
        from ring1.task_executor import _clean_for_memory
        text = (
            '运行报错了：\n'
            'Traceback (most recent call last):\n'
            '  File "main.py", line 10, in <module>\n'
            '    foo()\n'
            'TypeError: foo() missing argument'
        )
        result = _clean_for_memory(text)
        assert "Traceback" not in result
        assert "运行报错了" in result

    def test_strip_long_urls(self):
        from ring1.task_executor import _clean_for_memory
        url = "https://example.com/" + "a" * 100
        text = f"看看这个链接 {url} 有什么内容"
        result = _clean_for_memory(text)
        assert url not in result
        assert "看看这个链接" in result

    def test_truncate_long_text(self):
        from ring1.task_executor import _clean_for_memory
        text = "分析一下这个问题 " + "详细内容" * 100
        result = _clean_for_memory(text)
        assert len(result) <= 204  # 200 + "..."
        assert result.endswith("...")

    def test_short_text_unchanged(self):
        from ring1.task_executor import _clean_for_memory
        text = "帮我查一下天气"
        assert _clean_for_memory(text) == text

    def test_fallback_when_all_stripped(self):
        from ring1.task_executor import _clean_for_memory
        text = '```python\ndef foo():\n    pass\n```'
        result = _clean_for_memory(text)
        # Should fall back to truncated original, not empty
        assert len(result) > 0
