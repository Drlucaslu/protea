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
    _TASK_SYSTEM_PROMPT_BASE as TASK_SYSTEM_PROMPT,
    _P1_SYSTEM_PROMPT_BASE as P1_SYSTEM_PROMPT,
    TaskExecutor,
    _build_task_context,
    _match_skills,
    _MAX_REPLY_LEN,
    create_executor,
    start_executor_thread,
)
from ring1.tool_registry import Tool, ToolRegistry
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


def _make_registry() -> ToolRegistry:
    """Create a simple test registry."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="test_tool",
        description="Test tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        execute=lambda inp: "tool result",
    ))
    return reg


def _make_executor(
    state: SentinelState | None = None,
    ring2_path: pathlib.Path | None = None,
    reply_fn=None,
    client=None,
    registry=None,
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
    return TaskExecutor(state, client, ring2_path, reply_fn, registry=registry)


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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="What is 2+2?", chat_id="123")

        executor._execute_task(task)

        client.send_message_with_tools.assert_called_once()
        call_args = client.send_message_with_tools.call_args
        assert "What is 2+2?" in call_args[0][1]  # user_message
        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("Here is my answer")

    def test_execute_task_passes_registry(self, tmp_path):
        """Registry schemas and executor should be passed to LLM."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        call_kwargs = client.send_message_with_tools.call_args
        tools = call_kwargs[1]["tools"] if "tools" in (call_kwargs[1] or {}) else call_kwargs[0][2]
        assert any(t["name"] == "test_tool" for t in tools)

    def test_execute_without_registry_uses_send_message(self, tmp_path):
        """Without registry, should fall back to send_message."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message.return_value = "simple answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=None)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        client.send_message.assert_called_once()
        client.send_message_with_tools.assert_not_called()
        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("simple answer")

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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
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
        # Must exceed _MAX_REPLY_LEN (8000) to trigger truncation.
        client.send_message_with_tools.return_value = "x" * 10000
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        # With segmented sending, all segments combined should contain the truncation marker.
        all_sent = "".join(call[0][0] for call in reply_fn.call_args_list)
        assert "truncated" in all_sent

    def test_long_response_segmented(self, tmp_path):
        """Responses > 4000 chars are split into multiple Telegram messages."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        # 6000 chars: under _MAX_REPLY_LEN (no truncation) but over _TG_MSG_LIMIT (segmented)
        client.send_message_with_tools.return_value = "x" * 6000
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        assert reply_fn.call_count >= 2  # Should be split into at least 2 segments

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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="hello", chat_id="123")
        state.task_queue.put(task)

        thread = start_executor_thread(executor)
        deadline = time.time() + 5
        while time.time() < deadline and not reply_fn.called:
            time.sleep(0.1)

        assert reply_fn.called
        assert reply_fn.call_args[0][0].startswith("answer")

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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("answer")

    def test_p0_active_cleared_on_exception(self, tmp_path):
        """p0_active should be cleared even if reply_fn raises."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock(side_effect=RuntimeError("send failed"))
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")

        # Should not raise — the exception is caught
        executor._execute_task(task)
        assert not state.p0_active.is_set()

    def test_max_tool_rounds_passed(self, tmp_path):
        """max_tool_rounds should be passed to send_message_with_tools."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, max_tool_rounds=15,
        )
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)

        call_kwargs = client.send_message_with_tools.call_args[1]
        assert call_kwargs["max_rounds"] == 15


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
        ms.add(1, "observation", "threads cause crashes")

        captured_messages = []
        def capture_send(system, user, *args, **kwargs):
            captured_messages.append(user)
            return "answer"

        client = MagicMock()
        client.send_message_with_tools.side_effect = capture_send
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry, memory_store=ms)
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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="hello", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("answer")


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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry, memory_store=ms)
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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")
        executor._execute_task(task)  # should not raise
        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("answer")


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
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry, skill_store=ss)
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

        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry,
            memory_store=ms, p1_enabled=True,
            p1_idle_threshold_sec=0,
            p1_check_interval_sec=0,
        )
        executor._last_p0_time = 0
        executor._check_p1_opportunity()

        # Decision via send_message + intent extraction for skill matching
        assert client.send_message.call_count == 2
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
        ms.add(1, "task", "test task", importance=0.5)

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
        from ring1.llm_base import LLMError
        cfg = MagicMock()
        cfg.get_llm_client.side_effect = LLMError("API key is not set")
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
        cfg.workspace_path = "."
        cfg.shell_timeout = 30
        cfg.max_tool_rounds = 10
        state = _make_state()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock())
        assert isinstance(result, TaskExecutor)
        assert result.p1_enabled is True
        assert result.registry is not None

    def test_executor_has_registry_tools(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = False
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        cfg.workspace_path = "."
        cfg.shell_timeout = 30
        cfg.max_tool_rounds = 10
        state = _make_state()
        reply_fn = MagicMock()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), reply_fn)
        assert result is not None
        tool_names = result.registry.tool_names()
        assert "web_search" in tool_names
        assert "web_fetch" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names
        assert "list_dir" in tool_names
        assert "exec" in tool_names
        assert "message" in tool_names
        assert "spawn" in tool_names

    def test_skill_store_passed_through(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = False
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        cfg.workspace_path = "."
        cfg.shell_timeout = 30
        cfg.max_tool_rounds = 10
        state = _make_state()
        skill_store = MagicMock()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock(), skill_store=skill_store)
        assert result.skill_store is skill_store

    def test_subagent_manager_attached(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = False
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        cfg.workspace_path = "."
        cfg.shell_timeout = 30
        cfg.max_tool_rounds = 10
        state = _make_state()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock())
        assert hasattr(result, "subagent_manager")
        assert result.subagent_manager is not None


# ---------------------------------------------------------------------------
# TestSystemPrompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Test TASK_SYSTEM_PROMPT content."""

    def test_mentions_all_tools(self):
        assert "web_search" in TASK_SYSTEM_PROMPT
        assert "web_fetch" in TASK_SYSTEM_PROMPT
        assert "read_file" in TASK_SYSTEM_PROMPT
        assert "write_file" in TASK_SYSTEM_PROMPT
        assert "edit_file" in TASK_SYSTEM_PROMPT
        assert "list_dir" in TASK_SYSTEM_PROMPT
        assert "exec" in TASK_SYSTEM_PROMPT
        assert "message" in TASK_SYSTEM_PROMPT
        assert "spawn" in TASK_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TestTaskPersistence
# ---------------------------------------------------------------------------

class TestTaskPersistence:
    """Test TaskExecutor task_store integration."""

    def test_task_store_marked_executing_then_completed(self, tmp_path):
        from ring0.task_store import TaskStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ts = TaskStore(tmp_path / "tasks.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, task_store=ts,
        )
        task = Task(text="What is 2+2?", chat_id="123")
        ts.add(task.task_id, task.text, task.chat_id, task.created_at)

        executor._execute_task(task)

        rows = ts.get_recent(1)
        assert rows[0]["status"] == "completed"
        assert rows[0]["result"] != ""
        assert rows[0]["completed_at"] is not None

    def test_task_store_not_required(self, tmp_path):
        """Executor without task_store should work fine."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, task_store=None,
        )
        task = Task(text="hello", chat_id="123")
        executor._execute_task(task)
        reply_fn.assert_called_once()
        assert reply_fn.call_args[0][0].startswith("answer")

    def test_last_task_completion_updated(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(state, client, ring2, reply_fn, registry=registry)
        task = Task(text="test", chat_id="123")

        before = time.time()
        executor._execute_task(task)
        after = time.time()

        assert before <= state.last_task_completion <= after

    def test_create_executor_passes_task_store(self):
        cfg = MagicMock()
        cfg.claude_api_key = "sk-test"
        cfg.claude_model = "test-model"
        cfg.claude_max_tokens = 4096
        cfg.p1_enabled = False
        cfg.p1_idle_threshold_sec = 600
        cfg.p1_check_interval_sec = 60
        cfg.workspace_path = "."
        cfg.shell_timeout = 30
        cfg.max_tool_rounds = 10
        state = _make_state()
        ts = MagicMock()
        result = create_executor(
            cfg, state, pathlib.Path("/tmp"), MagicMock(), task_store=ts,
        )
        assert result.task_store is ts


# ---------------------------------------------------------------------------
# TestTaskRecovery
# ---------------------------------------------------------------------------

class TestTaskRecovery:
    """Test _recover_tasks restores pending/executing tasks after restart."""

    def test_recover_pending_tasks(self, tmp_path):
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")
        now = time.time()
        ts.add("t-1", "task one", "c1", created_at=now - 60)
        ts.add("t-2", "task two", "c1", created_at=now - 30)

        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, tmp_path, reply_fn, task_store=ts,
        )
        executor._recover_tasks()

        assert state.task_queue.qsize() == 2
        t1 = state.task_queue.get_nowait()
        assert t1.task_id == "t-1"
        t2 = state.task_queue.get_nowait()
        assert t2.task_id == "t-2"

    def test_recover_executing_reset_to_pending(self, tmp_path):
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")
        ts.add("t-1", "interrupted", "c1", created_at=time.time())
        ts.set_status("t-1", "executing")

        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, tmp_path, reply_fn, task_store=ts,
        )
        executor._recover_tasks()

        # Should be re-enqueued
        assert state.task_queue.qsize() == 1
        task = state.task_queue.get_nowait()
        assert task.task_id == "t-1"
        assert task.text == "interrupted"

    def test_recover_skips_completed(self, tmp_path):
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")
        ts.add("t-1", "done", "c1", created_at=time.time())
        ts.set_status("t-1", "completed", "result")
        ts.add("t-2", "pending", "c1", created_at=time.time())

        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, tmp_path, reply_fn, task_store=ts,
        )
        executor._recover_tasks()

        assert state.task_queue.qsize() == 1
        task = state.task_queue.get_nowait()
        assert task.task_id == "t-2"

    def test_recover_expires_stale_tasks(self, tmp_path):
        """Tasks older than _RECOVER_MAX_AGE_SEC are expired, not re-enqueued."""
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")
        now = time.time()
        ts.add("t-old", "stale task", "c1", created_at=now - 600)  # 10 min ago
        ts.add("t-new", "fresh task", "c1", created_at=now - 60)   # 1 min ago

        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, tmp_path, reply_fn, task_store=ts,
        )
        executor._recover_tasks()

        # Only the fresh task should be recovered.
        assert state.task_queue.qsize() == 1
        task = state.task_queue.get_nowait()
        assert task.task_id == "t-new"

    def test_recover_no_store(self):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, pathlib.Path("/tmp"), reply_fn, task_store=None,
        )
        executor._recover_tasks()  # should not raise
        assert state.task_queue.qsize() == 0

    def test_recover_empty_store(self, tmp_path):
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")

        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(
            state, client, tmp_path, reply_fn, task_store=ts,
        )
        executor._recover_tasks()
        assert state.task_queue.qsize() == 0

    def test_run_calls_recover(self, tmp_path):
        """run() should call _recover_tasks before entering the loop."""
        from ring0.task_store import TaskStore
        state = _make_state()
        ts = TaskStore(tmp_path / "tasks.db")
        ts.add("t-1", "recover me", "c1")

        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "recovered answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, task_store=ts,
        )

        thread = start_executor_thread(executor)
        deadline = time.time() + 5
        while time.time() < deadline and not reply_fn.called:
            time.sleep(0.1)

        assert reply_fn.called
        executor.stop()
        thread.join(timeout=5)

        # Task should be completed in store
        rows = ts.get_recent(1)
        assert rows[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# TestChatHistory
# ---------------------------------------------------------------------------

class TestChatHistory:
    """Conversation history should carry context across tasks."""

    def test_record_and_get_history(self):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(state, client, pathlib.Path("/tmp"), reply_fn)

        executor._record_history("find credentials.json", "Found 2 files: /a, /b")
        executor._record_history("read /a", "Contents of /a: ...")

        history = executor._get_recent_history()
        assert len(history) == 2
        assert history[0] == ("find credentials.json", "Found 2 files: /a, /b")
        assert history[1] == ("read /a", "Contents of /a: ...")

    def test_history_expires_after_ttl(self):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(state, client, pathlib.Path("/tmp"), reply_fn)

        # Insert an entry with old timestamp
        executor._chat_history.append((time.time() - 700, "old q", "old a"))
        executor._record_history("new q", "new a")

        history = executor._get_recent_history()
        assert len(history) == 1
        assert history[0][0] == "new q"

    def test_history_max_entries(self):
        state = _make_state()
        client = MagicMock()
        reply_fn = MagicMock()
        executor = TaskExecutor(state, client, pathlib.Path("/tmp"), reply_fn)

        for i in range(10):
            executor._record_history(f"q{i}", f"a{i}")

        history = executor._get_recent_history()
        assert len(history) == 5  # _chat_history_max = 5
        assert history[0] == ("q5", "a5")
        assert history[-1] == ("q9", "a9")

    def test_history_included_in_context(self):
        executor_history = [("find file.txt", "Found at /tmp/file.txt")]
        context = _build_task_context(
            {"generation": 1, "alive": True, "paused": False},
            "",
            chat_history=executor_history,
        )
        assert "## Recent Conversation" in context
        assert "find file.txt" in context
        assert "/tmp/file.txt" in context

    def test_empty_history_not_in_context(self):
        context = _build_task_context(
            {"generation": 1, "alive": True, "paused": False},
            "",
            chat_history=[],
        )
        assert "Recent Conversation" not in context

    def test_history_recorded_after_task(self, tmp_path):
        """_execute_task should record conversation history."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "Found 2 files"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn, registry=registry,
        )

        task = Task(text="find credentials.json", chat_id="chat1", task_id="t-1")
        executor._execute_task(task)

        history = executor._get_recent_history()
        assert len(history) == 1
        assert history[0][0] == "find credentials.json"
        assert history[0][1] == "Found 2 files"


# ---------------------------------------------------------------------------
# TestSkillHitTracking
# ---------------------------------------------------------------------------

class TestSkillHitTracking:
    """Test that skill usage is tracked in metadata and footer."""

    def _make_skill_registry(self):
        """Create a test registry with run_skill tool."""
        reg = ToolRegistry()
        reg.register(Tool(
            name="run_skill",
            description="Run a skill",
            input_schema={
                "type": "object",
                "properties": {"skill_name": {"type": "string"}},
                "required": ["skill_name"],
            },
            execute=lambda inp: "skill output",
        ))
        return reg

    def test_skill_used_recorded_in_metadata(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()

        def fake_send(system, user, tools=None, tool_executor=None, max_rounds=None):
            if tool_executor:
                tool_executor("run_skill", {"skill_name": "my_skill"})
            return "Skill result"

        client.send_message_with_tools.side_effect = fake_send
        reply_fn = MagicMock()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=self._make_skill_registry(), memory_store=ms,
        )
        task = Task(text="run my skill", chat_id="123")
        executor._execute_task(task)

        tasks = ms.get_by_type("task")
        assert len(tasks) == 1
        assert tasks[0]["metadata"]["skills_used"] == ["my_skill"]

    def test_no_skill_empty_list(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=self._make_skill_registry(), memory_store=ms,
        )
        task = Task(text="just answer", chat_id="123")
        executor._execute_task(task)

        tasks = ms.get_by_type("task")
        assert len(tasks) == 1
        assert tasks[0]["metadata"]["skills_used"] == []

    def test_multiple_skills_tracked(self, tmp_path):
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()

        def fake_send(system, user, tools=None, tool_executor=None, max_rounds=None):
            if tool_executor:
                tool_executor("run_skill", {"skill_name": "skill_a"})
                tool_executor("run_skill", {"skill_name": "skill_b"})
            return "Done"

        client.send_message_with_tools.side_effect = fake_send
        reply_fn = MagicMock()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=self._make_skill_registry(), memory_store=ms,
        )
        task = Task(text="use skills", chat_id="123")
        executor._execute_task(task)

        tasks = ms.get_by_type("task")
        assert tasks[0]["metadata"]["skills_used"] == ["skill_a", "skill_b"]

    def test_footer_with_skill(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()

        def fake_send(system, user, tools=None, tool_executor=None, max_rounds=None):
            if tool_executor:
                tool_executor("run_skill", {"skill_name": "web_dash"})
            return "Started"

        client.send_message_with_tools.side_effect = fake_send
        reply_fn = MagicMock()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=self._make_skill_registry(),
        )
        task = Task(text="start web dash", chat_id="123")
        executor._execute_task(task)

        sent = reply_fn.call_args[0][0]
        assert "skill: web_dash" in sent
        assert "---" in sent

    def test_footer_llm_only(self, tmp_path):
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "LLM answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=self._make_skill_registry(),
        )
        task = Task(text="what is 2+2", chat_id="123")
        executor._execute_task(task)

        sent = reply_fn.call_args[0][0]
        assert "llm |" in sent
        assert "---" in sent


# ---------------------------------------------------------------------------
# TestCorrectionDetection
# ---------------------------------------------------------------------------

class TestCorrectionDetection:
    """Test correction pattern detection and semantic_rule storage."""

    def test_correction_detected_and_stored(self, tmp_path):
        """Task text with correction pattern stored as semantic_rule."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "OK, 我记住了"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="不对，你应该用中文回复", chat_id="123")
        executor._execute_task(task)

        rules = ms.get_semantic_rules(limit=10)
        assert len(rules) == 1
        assert "你应该用中文回复" in rules[0]["content"]

    def test_remember_pattern_stored(self, tmp_path):
        """'记住...' pattern should trigger semantic_rule storage."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "Got it"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="记住，新闻要用中文源。", chat_id="123")
        executor._execute_task(task)

        rules = ms.get_semantic_rules(limit=10)
        assert len(rules) == 1

    def test_future_directive_stored(self, tmp_path):
        """'下次...要...' pattern should trigger semantic_rule storage."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "Understood"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="下次查新闻要用中文搜索", chat_id="123")
        executor._execute_task(task)

        rules = ms.get_semantic_rules(limit=10)
        assert len(rules) == 1

    def test_english_correction_stored(self, tmp_path):
        """English correction patterns work too."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "Noted"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="That's wrong, you should always reply in Chinese", chat_id="123")
        executor._execute_task(task)

        rules = ms.get_semantic_rules(limit=10)
        assert len(rules) == 1

    def test_non_correction_not_stored(self, tmp_path):
        """Normal task text does not trigger semantic_rule creation."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        client = MagicMock()
        client.send_message_with_tools.return_value = "Here's the weather"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="今天天气怎么样", chat_id="123")
        executor._execute_task(task)

        rules = ms.get_semantic_rules(limit=10)
        assert len(rules) == 0

    def test_no_memory_store_no_crash(self, tmp_path):
        """Correction detection gracefully handles missing memory_store."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "OK"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=None,
        )
        task = Task(text="不对，你应该用中文回复", chat_id="123")
        executor._execute_task(task)  # should not raise
        reply_fn.assert_called_once()


# ---------------------------------------------------------------------------
# TestSemanticRulesContext
# ---------------------------------------------------------------------------

class TestSemanticRulesContext:
    """Test that semantic_rules are injected into task context."""

    def test_semantic_rules_in_context(self):
        """_build_task_context includes semantic_rules section."""
        snap = {"generation": 1, "alive": True, "paused": False,
                "last_score": 0.9, "last_survived": True}
        rules = [
            {"content": "你应该用中文回复", "generation": 1},
            {"content": "新闻要用中文源", "generation": 2},
        ]
        ctx = _build_task_context(snap, "", semantic_rules=rules)
        assert "## Correction Rules (MUST follow)" in ctx
        assert "你应该用中文回复" in ctx
        assert "新闻要用中文源" in ctx

    def test_semantic_rules_empty_no_section(self):
        """No semantic_rules -> no section in context."""
        snap = {"generation": 1, "alive": True, "paused": False,
                "last_score": 0.9, "last_survived": True}
        ctx = _build_task_context(snap, "", semantic_rules=[])
        assert "Correction Rules" not in ctx

    def test_semantic_rules_none_no_section(self):
        """semantic_rules=None -> no section in context."""
        snap = {"generation": 1, "alive": True, "paused": False,
                "last_score": 0.9, "last_survived": True}
        ctx = _build_task_context(snap, "", semantic_rules=None)
        assert "Correction Rules" not in ctx

    def test_semantic_rules_injected_in_p0(self, tmp_path):
        """P0 task context should contain semantic_rules from memory."""
        from ring0.memory import MemoryStore
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        ms.add(1, "semantic_rule", "你应该用中文回复", importance=0.8)

        captured = []
        def capture(system, user, *args, **kwargs):
            captured.append(user)
            return "answer"

        client = MagicMock()
        client.send_message_with_tools.side_effect = capture
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms,
        )
        task = Task(text="今天有什么新闻", chat_id="123")
        executor._execute_task(task)

        assert len(captured) == 1
        assert "Correction Rules" in captured[0]
        assert "你应该用中文回复" in captured[0]


# ---------------------------------------------------------------------------
# TestExtractProfileIntent
# ---------------------------------------------------------------------------

class TestExtractProfileIntent:
    def test_returns_llm_translation(self):
        executor = _make_executor()
        executor.client.send_message.return_value = "Analyze stock data"
        result = executor._extract_profile_intent("帮我分析股票数据")
        assert result == "Analyze stock data"
        executor.client.send_message.assert_called_once()

    def test_unclear_returns_empty(self):
        executor = _make_executor()
        executor.client.send_message.return_value = "unclear"
        result = executor._extract_profile_intent("好的，谢谢你的帮助")
        assert result == ""

    def test_short_text_skips_llm(self):
        executor = _make_executor()
        result = executor._extract_profile_intent("ok")
        assert result == "ok"
        executor.client.send_message.assert_not_called()

    def test_fallback_on_llm_failure(self):
        executor = _make_executor()
        executor.client.send_message.side_effect = RuntimeError("API down")
        result = executor._extract_profile_intent("帮我分析股票数据")
        assert result == "帮我分析股票数据"

    def test_no_client_returns_raw(self):
        executor = _make_executor()
        executor.client = None
        result = executor._extract_profile_intent("analyze data")
        assert result == "analyze data"

    def test_truncates_long_response(self):
        executor = _make_executor()
        executor.client.send_message.return_value = "x" * 300
        result = executor._extract_profile_intent("some long input text here")
        assert len(result) == 200


# ---------------------------------------------------------------------------
# TestSkillsMatchedRecording
# ---------------------------------------------------------------------------

class TestSkillsMatchedRecording:
    """Test that skills_matched is recorded in task metadata."""

    def test_skills_matched_recorded_in_metadata(self, tmp_path):
        """Matched skill names should appear in metadata.skills_matched."""
        from ring0.memory import MemoryStore
        from ring0.skill_store import SkillStore

        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        ss = SkillStore(tmp_path / "skills.db")
        ss.add("summarize", "Summarize text", "Please summarize: {{text}}")
        ss.add("translate", "Translate text", "Translate: {{text}}")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms, skill_store=ss,
        )
        # Use text that matches "summarize" skill (needs >=2 token matches)
        task = Task(text="summarize this text for me", chat_id="123")
        executor._execute_task(task)

        tasks = ms.get_by_type("task")
        assert len(tasks) == 1
        meta = tasks[0]["metadata"]
        assert "skills_matched" in meta
        assert "summarize" in meta["skills_matched"]

    def test_skills_matched_empty_when_no_match(self, tmp_path):
        """When no skills match, skills_matched should be an empty list."""
        from ring0.memory import MemoryStore
        from ring0.skill_store import SkillStore

        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        ms = MemoryStore(tmp_path / "mem.db")
        ss = SkillStore(tmp_path / "skills.db")
        ss.add("quantum_physics", "Quantum calculations", "Calculate: {{expr}}")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        reply_fn = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, reply_fn,
            registry=registry, memory_store=ms, skill_store=ss,
        )
        task = Task(text="what is the weather today", chat_id="123")
        executor._execute_task(task)

        tasks = ms.get_by_type("task")
        assert len(tasks) == 1
        meta = tasks[0]["metadata"]
        assert "skills_matched" in meta
        assert meta["skills_matched"] == []


# ---------------------------------------------------------------------------
# TestMatchSkillsFiltering
# ---------------------------------------------------------------------------

class TestMatchSkillsFiltering:
    """Test the three-way filtering in _match_skills()."""

    def _skill(self, name, description="", tags=None):
        return {"name": name, "description": description, "tags": tags or []}

    def test_single_token_match_rejected(self):
        """A single token overlap should not be enough to recommend."""
        skills = [self._skill("stock_backtest", "Backtest stock strategies")]
        recommended, other = _match_skills("check stock prices today", skills)
        assert recommended == []
        assert len(other) == 1

    def test_two_token_match_accepted(self):
        """Two matching tokens should recommend the skill."""
        skills = [self._skill("summarize", "Summarize text")]
        recommended, other = _match_skills("summarize this text for me", skills)
        assert len(recommended) == 1
        assert recommended[0]["name"] == "summarize"
        assert other == []

    def test_low_ratio_rejected(self):
        """Two matches but too many tokens → low ratio → rejected."""
        # 2 matches out of many tokens: "data" + "analysis" match,
        # but 2/15 = 0.13 < 0.15 threshold
        long_text = "please help me understand the data analysis of this very long complicated research paper about many topics"
        skills = [self._skill("data_analysis", "Analyze data sets")]
        recommended, other = _match_skills(long_text, skills)
        assert recommended == []
        assert len(other) == 1

    def test_max_ten_recommended(self):
        """At most 10 skills should be recommended."""
        # Create 15 skills that all match "data analysis report"
        skills = [
            self._skill(f"skill_{i}", "data analysis report processing")
            for i in range(15)
        ]
        recommended, other = _match_skills("data analysis report", skills)
        assert len(recommended) == 10
        assert len(other) == 5

    def test_empty_task_returns_all_as_other(self):
        """Empty task text should not recommend any skill."""
        skills = [self._skill("s1", "desc"), self._skill("s2", "desc")]
        recommended, other = _match_skills("", skills)
        assert recommended == []
        assert len(other) == 2

    def test_sorted_by_score_descending(self):
        """Recommended skills should be sorted by score descending."""
        skills = [
            self._skill("low_match", "text processing"),
            self._skill("high_match", "summarize text document"),
        ]
        recommended, other = _match_skills("summarize this text document", skills)
        # high_match: "summarize" + "text" + "document" = 3
        # low_match: "text" + "processing"... "processing" not in task. "text" = 1? No, score=1 < 2
        assert len(recommended) >= 1
        assert recommended[0]["name"] == "high_match"

    def test_chinese_text_no_match(self):
        """Pure Chinese text should not match English skills.

        Chinese bigrams are skipped; callers should translate to English first.
        """
        skills = [
            self._skill("health_research", "Research health topics"),
            self._skill("summarize", "Summarize text"),
        ]
        recommended, other = _match_skills("帮我研究NK细胞疗法的临床数据", skills)
        assert recommended == []
        assert len(other) == 2

    def test_mixed_text_extracts_english_only(self):
        """Mixed Chinese-English text should only use English tokens."""
        skills = [
            self._skill("health_research", "Research health topics and therapy"),
        ]
        # "HIIT" and "therapy" are English tokens that can match
        recommended, other = _match_skills("帮我做HIIT训练 and look into therapy options", skills)
        assert len(recommended) == 1
        assert recommended[0]["name"] == "health_research"


class TestGenePatternContext:
    """Test that gene_patterns are injected into task context."""

    _snap = {"generation": 1, "alive": True, "paused": False,
             "last_score": 0.9, "last_survived": True}

    def test_gene_patterns_in_context(self):
        """_build_task_context includes Proven Code Patterns section."""
        genes = [
            {"score": 0.85, "total_task_hits": 3, "gene_summary": "Use asyncio for I/O"},
            {"score": 0.72, "total_task_hits": 1, "gene_summary": "Cache API responses"},
        ]
        ctx = _build_task_context(self._snap, "", gene_patterns=genes)
        assert "## Proven Code Patterns" in ctx
        assert "[score=0.85, tasks=3] Use asyncio for I/O" in ctx
        assert "[score=0.72, tasks=1] Cache API responses" in ctx

    def test_empty_gene_patterns_no_section(self):
        """Empty gene_patterns -> no section in context."""
        ctx = _build_task_context(self._snap, "", gene_patterns=[])
        assert "Proven Code Patterns" not in ctx

    def test_none_gene_patterns_no_section(self):
        """gene_patterns=None -> no section in context."""
        ctx = _build_task_context(self._snap, "", gene_patterns=None)
        assert "Proven Code Patterns" not in ctx

    def test_gene_summary_truncation(self):
        """Gene summaries longer than 200 chars are truncated."""
        long_summary = "A" * 250
        genes = [{"score": 0.5, "total_task_hits": 0, "gene_summary": long_summary}]
        ctx = _build_task_context(self._snap, "", gene_patterns=genes)
        assert "Proven Code Patterns" in ctx
        # Should be truncated to 197 chars + "..."
        assert "A" * 197 + "..." in ctx
        assert "A" * 200 not in ctx


# ---------------------------------------------------------------------------
# TestReplyFnFactory
# ---------------------------------------------------------------------------

class TestReplyFnFactory:
    """Test per-task reply routing via reply_fn_factory."""

    def test_execute_task_with_reply_fn_factory(self, tmp_path):
        """When reply_fn_factory is set, task replies go through it."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "group answer"
        default_reply = MagicMock()
        factory_reply = MagicMock()

        def factory(chat_id, reply_to_id=None):
            return factory_reply

        registry = _make_registry()
        executor = TaskExecutor(
            state, client, ring2, default_reply,
            registry=registry, reply_fn_factory=factory,
        )
        task = Task(text="hello", chat_id="-100999", reply_to_message_id=42)
        executor._execute_task(task)

        # Should use factory reply, not default
        factory_reply.assert_called()
        default_reply.assert_not_called()
        assert factory_reply.call_args[0][0].startswith("group answer")

    def test_execute_task_default_reply_fn(self, tmp_path):
        """Without reply_fn_factory, default reply_fn is used."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        client = MagicMock()
        client.send_message_with_tools.return_value = "answer"
        default_reply = MagicMock()
        registry = _make_registry()

        executor = TaskExecutor(
            state, client, ring2, default_reply,
            registry=registry, reply_fn_factory=None,
        )
        task = Task(text="hello", chat_id="12345")
        executor._execute_task(task)

        default_reply.assert_called_once()
        assert default_reply.call_args[0][0].startswith("answer")
