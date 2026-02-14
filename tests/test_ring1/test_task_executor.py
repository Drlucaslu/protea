"""Tests for ring1.task_executor."""

from __future__ import annotations

import json
import pathlib
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ring1.task_executor import (
    TASK_SYSTEM_PROMPT,
    P1_SYSTEM_PROMPT,
    WEB_TOOLS,
    TaskExecutor,
    _build_task_context,
    _execute_tool,
    _MAX_REPLY_LEN,
    create_executor,
    start_executor_thread,
)
from ring1.telegram_bot import SentinelState, Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state() -> SentinelState:
    state = SentinelState()
    with state.lock:
        state.generation = 5
        state.alive = True
    return state


def _make_executor(
    state: SentinelState | None = None,
    ring2_path: pathlib.Path | None = None,
    reply_fn=None,
    client=None,
) -> TaskExecutor:
    if state is None:
        state = _make_state()
    if client is None:
        client = MagicMock()
        client.send_message_with_tools.return_value = "LLM response"
    if reply_fn is None:
        reply_fn = MagicMock()
    if ring2_path is None:
        ring2_path = pathlib.Path("/tmp/ring2")
    return TaskExecutor(state, client, ring2_path, reply_fn)


# ---------------------------------------------------------------------------
# TestBuildTaskContext
# ---------------------------------------------------------------------------

class TestBuildTaskContext:
    def test_includes_generation(self):
        snap = {"generation": 7, "alive": True, "paused": False,
                "last_score": 0.9, "last_survived": True}
        ctx = _build_task_context(snap, "")
        assert "Generation: 7" in ctx

    def test_includes_source(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "print('hello')")
        assert "print('hello')" in ctx
        assert "```python" in ctx

    def test_truncates_long_source(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        long_source = "x" * 3000
        ctx = _build_task_context(snap, long_source)
        assert "truncated" in ctx
        # Only first 2000 chars of source
        assert "x" * 2000 in ctx

    def test_empty_source(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "")
        assert "```python" not in ctx

    def test_includes_memories(self):
        snap = {"generation": 5, "alive": True, "paused": False,
                "last_score": 1.0, "last_survived": True}
        memories = [
            {"generation": 3, "content": "CA patterns are stable"},
            {"generation": 4, "content": "Threads cause heartbeat loss"},
        ]
        ctx = _build_task_context(snap, "", memories=memories)
        assert "Recent Learnings" in ctx
        assert "CA patterns are stable" in ctx
        assert "Gen 3" in ctx
        assert "Gen 4" in ctx

    def test_no_memories_no_section(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "", memories=None)
        assert "Recent Learnings" not in ctx

    def test_empty_memories_no_section(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "", memories=[])
        assert "Recent Learnings" not in ctx

    def test_includes_skills(self):
        snap = {"generation": 0, "alive": True, "paused": False,
                "last_score": 1.0, "last_survived": True}
        skills = [
            {"name": "summarize", "description": "Summarize text"},
            {"name": "translate", "description": "Translate text"},
        ]
        ctx = _build_task_context(snap, "", skills=skills)
        assert "Available Skills" in ctx
        assert "summarize: Summarize text" in ctx
        assert "translate: Translate text" in ctx

    def test_no_skills_no_section(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "", skills=None)
        assert "Available Skills" not in ctx

    def test_empty_skills_no_section(self):
        snap = {"generation": 0, "alive": False, "paused": False,
                "last_score": 0.0, "last_survived": False}
        ctx = _build_task_context(snap, "", skills=[])
        assert "Available Skills" not in ctx


# ---------------------------------------------------------------------------
# TestTaskExecutor
# ---------------------------------------------------------------------------

class TestTaskExecutor:
    def test_execute_task_calls_llm_and_replies(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("print('hello')")

        client = MagicMock()
        client.send_message_with_tools.return_value = "Here is my answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="What is 2+2?", chat_id="123")

        executor._execute_task(task)

        client.send_message_with_tools.assert_called_once()
        call_args = client.send_message_with_tools.call_args
        assert "What is 2+2?" in call_args[0][1]  # user_message
        reply_fn.assert_called_once_with("Here is my answer")

    def test_p0_active_signal(self, tmp_path):
        """p0_active should be set during task execution and cleared after."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        p0_was_set = []

        def slow_send(system, user, *args, **kwargs):
            p0_was_set.append(state.p0_active.is_set())
            return "done"

        client = MagicMock()
        client.send_message_with_tools.side_effect = slow_send
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")

        assert not state.p0_active.is_set()
        executor._execute_task(task)
        assert p0_was_set == [True]  # was set during LLM call
        assert not state.p0_active.is_set()  # cleared after

    def test_llm_error_still_replies(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        from ring1.llm_client import LLMError
        client = MagicMock()
        client.send_message_with_tools.side_effect = LLMError("rate limited")
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once()
        assert "rate limited" in reply_fn.call_args[0][0]

    def test_long_response_truncated(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "x" * 5000
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        sent_text = reply_fn.call_args[0][0]
        assert len(sent_text) <= _MAX_REPLY_LEN + 20  # +20 for "... (truncated)"
        assert "truncated" in sent_text

    def test_clean_stop(self):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(state, client, pathlib.Path("/tmp"), reply_fn)

        thread = start_executor_thread(executor)
        assert thread.is_alive()
        executor.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_run_processes_queued_task(self, tmp_path):
        """Full run loop: enqueue a task, executor picks it up."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="hello", chat_id="123")
        state.task_queue.put(task)

        thread = start_executor_thread(executor)
        deadline = time.time() + 5
        while time.time() < deadline and not reply_fn.called:
            time.sleep(0.1)

        assert reply_fn.called
        assert reply_fn.call_args[0][0] == "answer"

        executor.stop()
        thread.join(timeout=5)

    def test_missing_ring2_file(self, tmp_path):
        """Executor should not crash when ring2/main.py is missing."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        # No main.py

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once_with("answer")

    def test_p0_active_cleared_on_exception(self, tmp_path):
        """p0_active should be cleared even if reply_fn raises."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock(side_effect=RuntimeError("send failed"))

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")

        # Should not raise — the exception is caught
        executor._execute_task(task)
        assert not state.p0_active.is_set()


# ---------------------------------------------------------------------------
# TestTaskExecutorWithMemory
# ---------------------------------------------------------------------------

class TestTaskExecutorWithMemory:
    """Test TaskExecutor memory_store integration."""

    def test_memory_context_injected(self, tmp_path):
        """When memory_store has entries, they appear in LLM context."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "reflection", "threads cause crashes")

        captured_messages = []
        def capture_send(system, user, *args, **kwargs):
            captured_messages.append(user)
            return "answer"

        client = MagicMock()
        client.send_message_with_tools.side_effect = capture_send
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn, memory_store=ms)
        task = Task(text="what have you learned?", chat_id="123")
        executor._execute_task(task)

        assert len(captured_messages) == 1
        assert "threads cause crashes" in captured_messages[0]
        assert "Recent Learnings" in captured_messages[0]

    def test_no_memory_store_works(self, tmp_path):
        """TaskExecutor without memory_store should work fine."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="hello", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once_with("answer")


class TestTaskHistoryRecording:
    """Test that completed P0 tasks are recorded in memory."""

    def test_task_recorded_in_memory(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn, memory_store=ms)
        task = Task(text="What is 2+2?", chat_id="123")
        executor._execute_task(task)

        # Should have one task entry
        tasks = ms.get_by_type("task")
        assert len(tasks) == 1
        assert tasks[0]["content"] == "What is 2+2?"
        assert "response_summary" in tasks[0]["metadata"]
        assert "duration_sec" in tasks[0]["metadata"]

    def test_no_memory_store_no_error(self, tmp_path):
        """Task recording should silently skip without memory_store."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)  # should not raise
        reply_fn.assert_called_once_with("answer")


class TestSkillInjection:
    """Test that skills are injected into task context."""

    def test_skills_in_context(self, tmp_path):
        from ring0.skill_store import SkillStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ss = SkillStore(tmp_path / "skills.db")
        ss.add("summarize", "Summarize text", "Please summarize: {{text}}")

        captured = []
        def capture(system, user, *args, **kwargs):
            captured.append(user)
            return "answer"

        client = MagicMock()
        client.send_message_with_tools.side_effect = capture
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn, skill_store=ss)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        assert len(captured) == 1
        assert "Available Skills" in captured[0]
        assert "summarize" in captured[0]


# ---------------------------------------------------------------------------
# TestP1IdleDetection
# ---------------------------------------------------------------------------

class TestP1IdleDetection:
    """Test P1 autonomous task idle detection logic."""

    def test_p1_disabled_no_check(self, tmp_path):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        executor = TaskExecutor(state, client, ring2, reply_fn, p1_enabled=False)
        executor._last_p0_time = 0  # long idle
        executor._check_p1_opportunity()
        client.send_message.assert_not_called()

    def test_p1_not_idle_enough(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "task", "test task")

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=600,
        )
        executor._last_p0_time = time.time()  # just now — not idle
        executor._check_p1_opportunity()
        client.send_message.assert_not_called()

    def test_p1_check_interval_respected(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "task", "test task")

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=0,  # always idle
            p1_check_interval_sec=9999,  # very long interval
        )
        executor._last_p0_time = 0
        executor._last_p1_check = time.time()  # just checked
        executor._check_p1_opportunity()
        client.send_message.assert_not_called()

    def test_p1_no_task_history_no_check(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        ms = MemoryStore(tmp_path / "mem.db")
        # No task entries

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=0,
            p1_check_interval_sec=0,
        )
        executor._last_p0_time = 0
        executor._check_p1_opportunity()
        client.send_message.assert_not_called()

    def test_p1_triggers_when_idle(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        state.p1_active = MagicMock()  # mock for set/clear tracking
        state.p1_active.set = MagicMock()
        state.p1_active.clear = MagicMock()

        client = MagicMock()
        # Decision call uses send_message
        client.send_message.return_value = (
            "## Decision\nYES\n\n## Task\nAnalyze patterns"
        )
        # Execution call uses send_message_with_tools
        client.send_message_with_tools.return_value = "Here's my analysis..."
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "task", "What is 2+2?")

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=0,
            p1_check_interval_sec=0,
        )
        executor._last_p0_time = 0
        executor._check_p1_opportunity()

        # Decision via send_message, execution via send_message_with_tools
        assert client.send_message.call_count == 1
        assert client.send_message_with_tools.call_count == 1
        # Should have reported via Telegram
        reply_fn.assert_called_once()
        assert "[P1 Autonomous Work]" in reply_fn.call_args[0][0]

    def test_p1_decision_no_skips(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        client = MagicMock()
        client.send_message.return_value = "## Decision\nNO\n\n## Task\nNothing to do"
        reply_fn = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "task", "test task")

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=0,
            p1_check_interval_sec=0,
        )
        executor._last_p0_time = 0
        executor._check_p1_opportunity()

        # Only 1 LLM call for decision (no execution)
        assert client.send_message.call_count == 1
        client.send_message_with_tools.assert_not_called()
        reply_fn.assert_not_called()


# ---------------------------------------------------------------------------
# TestCreateExecutor
# ---------------------------------------------------------------------------

class TestCreateExecutor:
    def test_no_api_key_returns_none(self):
        cfg = MagicMock()
        cfg.claude_api_key = ""
        state = _make_state()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock())
        assert result is None

    def test_valid_config_returns_executor(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = True
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        state = _make_state()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock())
        assert isinstance(result, TaskExecutor)
        assert result.p1_enabled is True

    def test_skill_store_passed_through(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = False
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        state = _make_state()
        skill_store = MagicMock()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock(), skill_store=skill_store)
        assert result.skill_store is skill_store


# ---------------------------------------------------------------------------
# TestWebToolsIntegration
# ---------------------------------------------------------------------------

class TestExecuteTool:
    """Test the _execute_tool dispatcher."""

    def test_web_search_dispatch(self, monkeypatch):
        import ring1.task_executor as mod
        monkeypatch.setattr(mod, "web_search", lambda query, max_results=5: [
            {"title": "Result", "url": "https://r.com", "snippet": "A snippet"},
        ])
        result = _execute_tool("web_search", {"query": "test"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Result"

    def test_web_search_with_max_results(self, monkeypatch):
        import ring1.task_executor as mod
        captured = {}
        def fake_search(query, max_results=5):
            captured["max_results"] = max_results
            return []
        monkeypatch.setattr(mod, "web_search", fake_search)
        _execute_tool("web_search", {"query": "test", "max_results": 3})
        assert captured["max_results"] == 3

    def test_web_fetch_dispatch(self, monkeypatch):
        import ring1.task_executor as mod
        monkeypatch.setattr(mod, "web_fetch", lambda url, max_chars=5000: "Page content")
        result = _execute_tool("web_fetch", {"url": "https://example.com"})
        assert result == "Page content"

    def test_web_fetch_with_max_chars(self, monkeypatch):
        import ring1.task_executor as mod
        captured = {}
        def fake_fetch(url, max_chars=5000):
            captured["max_chars"] = max_chars
            return "text"
        monkeypatch.setattr(mod, "web_fetch", fake_fetch)
        _execute_tool("web_fetch", {"url": "https://example.com", "max_chars": 1000})
        assert captured["max_chars"] == 1000

    def test_unknown_tool(self):
        result = _execute_tool("unknown_tool", {})
        assert "Unknown tool" in result


class TestWebToolsSchema:
    """Test WEB_TOOLS schema and TASK_SYSTEM_PROMPT content."""

    def test_web_tools_has_two_tools(self):
        assert len(WEB_TOOLS) == 2

    def test_web_search_schema(self):
        search_tool = WEB_TOOLS[0]
        assert search_tool["name"] == "web_search"
        assert "query" in search_tool["input_schema"]["properties"]
        assert "query" in search_tool["input_schema"]["required"]

    def test_web_fetch_schema(self):
        fetch_tool = WEB_TOOLS[1]
        assert fetch_tool["name"] == "web_fetch"
        assert "url" in fetch_tool["input_schema"]["properties"]
        assert "url" in fetch_tool["input_schema"]["required"]

    def test_system_prompt_mentions_web_tools(self):
        assert "web_search" in TASK_SYSTEM_PROMPT
        assert "web_fetch" in TASK_SYSTEM_PROMPT
