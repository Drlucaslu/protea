"""Tests for ring1.nudge — NudgeEngine."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from ring1.nudge import NudgeEngine, _task_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    llm_response: str = "ACTION: none",
    config: dict | None = None,
) -> NudgeEngine:
    """Create a NudgeEngine with mocked dependencies."""
    client = MagicMock()
    client.send_message.return_value = llm_response
    memory = MagicMock()
    memory.get_by_type.return_value = [{"content": "previous task"}]
    scheduled = MagicMock()
    profiler = MagicMock()
    profiler.get_profile_summary.return_value = "tech enthusiast"
    preference = MagicMock()
    cfg = config or {"interval_sec": 0, "max_daily_nudges": 100}
    return NudgeEngine(
        llm_client=client,
        memory_store=memory,
        scheduled_store=scheduled,
        user_profiler=profiler,
        preference_store=preference,
        config=cfg,
    )


_NUDGE_RESPONSE = """\
ACTION: nudge
TYPE: schedule
TEXT: 这个查询看起来你经常用到，要不要设为定期任务？
BUTTON_YES: 好的
BUTTON_NO: 不了
TASK: 查天气
"""

_EXPAND_RESPONSE = """\
ACTION: nudge
TYPE: expand
TEXT: 需要我详细展开吗？
BUTTON_YES: 展开
BUTTON_NO: 不用
TASK: 详细分析报告
"""

_TIP_RESPONSE = """\
ACTION: nudge
TYPE: tip
TEXT: 你知道 Protea 可以管理日历吗？试试 /calendar
BUTTON_YES: 试试
BUTTON_NO: 不了
TASK: 展示日历功能
"""

_PROACTIVE_RESPONSE = """\
ACTION: nudge
TYPE: proactive
TEXT: 早上好！要不要看看今天的天气？
BUTTON_YES: 看看
BUTTON_NO: 不了
TASK: 今天天气怎么样
"""

_NONE_RESPONSE = "ACTION: none"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPostTaskNudge:
    def test_returns_suggestion(self):
        engine = _make_engine(_NUDGE_RESPONSE)
        result = engine.post_task_nudge("查天气", "北京今天晴", {})
        assert result is not None
        text, buttons, meta = result
        assert "定期任务" in text
        assert len(buttons) == 1
        assert len(buttons[0]) == 2
        assert buttons[0][0]["callback_data"].startswith("nudge:schedule:")
        assert buttons[0][1]["callback_data"] == "nudge:dismiss"
        assert meta["nudge_type"] == "schedule"
        assert meta["suggested_task"] == "查天气"

    def test_returns_none_when_llm_says_no(self):
        engine = _make_engine(_NONE_RESPONSE)
        result = engine.post_task_nudge("ok", "好的", {})
        assert result is None

    def test_expand_nudge(self):
        engine = _make_engine(_EXPAND_RESPONSE)
        result = engine.post_task_nudge("分析报告", "简要分析...", {})
        assert result is not None
        text, buttons, meta = result
        assert meta["nudge_type"] == "expand"
        assert buttons[0][0]["callback_data"].startswith("nudge:expand:")

    def test_tip_nudge(self):
        engine = _make_engine(_TIP_RESPONSE)
        result = engine.post_task_nudge("hi", "你好", {})
        assert result is not None
        text, buttons, meta = result
        assert meta["nudge_type"] == "tip"
        # tip uses execute callback
        assert buttons[0][0]["callback_data"].startswith("nudge:execute:")


class TestProactiveNudge:
    def test_returns_suggestion(self):
        engine = _make_engine(_PROACTIVE_RESPONSE)
        result = engine.proactive_nudge()
        assert result is not None
        text, buttons, meta = result
        assert "天气" in text
        assert meta["nudge_type"] == "proactive"

    def test_returns_none_when_llm_says_no(self):
        engine = _make_engine(_NONE_RESPONSE)
        result = engine.proactive_nudge()
        assert result is None


class TestRateLimiting:
    def test_daily_cap(self):
        engine = _make_engine(_NUDGE_RESPONSE, config={
            "interval_sec": 0,
            "max_daily_nudges": 2,
        })
        # First two should succeed.
        r1 = engine.post_task_nudge("task1", "resp1", {})
        r2 = engine.post_task_nudge("task2", "resp2", {})
        assert r1 is not None
        assert r2 is not None
        # Third should be rate-limited.
        r3 = engine.post_task_nudge("task3", "resp3", {})
        assert r3 is None

    def test_interval_limiting(self):
        engine = _make_engine(_NUDGE_RESPONSE, config={
            "interval_sec": 9999,
            "max_daily_nudges": 100,
        })
        r1 = engine.post_task_nudge("task1", "resp1", {})
        assert r1 is not None
        # Second should be rate-limited by interval.
        r2 = engine.post_task_nudge("task2", "resp2", {})
        assert r2 is None


class TestParseNudgeResponse:
    def test_missing_text_returns_none(self):
        engine = _make_engine("ACTION: nudge\nTYPE: tip\n")
        result = engine.post_task_nudge("hi", "hello", {})
        assert result is None

    def test_garbage_response_returns_none(self):
        engine = _make_engine("I don't know what to say")
        result = engine.post_task_nudge("hi", "hello", {})
        assert result is None

    def test_llm_error_returns_none(self):
        engine = _make_engine("")
        engine._client.send_message.side_effect = Exception("LLM error")
        result = engine.post_task_nudge("hi", "hello", {})
        assert result is None


class TestNudgeDisabledWhenNoClient:
    def test_graceful_with_none_profiler(self):
        """Engine should work even when optional stores are None."""
        engine = _make_engine(_NUDGE_RESPONSE)
        engine._user_profiler = None
        engine._memory_store = None
        result = engine.post_task_nudge("hi", "hello", {})
        assert result is not None


class TestTaskHash:
    def test_deterministic(self):
        assert _task_hash("hello") == _task_hash("hello")

    def test_different_inputs(self):
        assert _task_hash("a") != _task_hash("b")

    def test_length(self):
        assert len(_task_hash("test")) == 8
