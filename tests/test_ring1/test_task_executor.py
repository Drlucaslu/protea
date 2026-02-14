"""Tests for ring1.task_executor."""

from __future__ import annotations

import pathlib
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ring1.task_executor import (
    TASK_SYSTEM_PROMPT,
    TaskExecutor,
    _build_task_context,
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
        client.send_message.return_value = "LLM response"
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
        client.send_message.return_value = "Here is my answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="What is 2+2?", chat_id="123")

        executor._execute_task(task)

        client.send_message.assert_called_once()
        call_args = client.send_message.call_args
        assert "What is 2+2?" in call_args[0][1]  # user_message
        reply_fn.assert_called_once_with("Here is my answer")

    def test_p0_active_signal(self, tmp_path):
        """p0_active should be set during task execution and cleared after."""
        state = _make_state()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("code")

        p0_was_set = []

        def slow_send(system, user):
            p0_was_set.append(state.p0_active.is_set())
            return "done"

        client = MagicMock()
        client.send_message.side_effect = slow_send
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
        client.send_message.side_effect = LLMError("rate limited")
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
        client.send_message.return_value = "x" * 5000
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
        client.send_message.return_value = "answer"
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
        client.send_message.return_value = "answer"
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
        client.send_message.return_value = "answer"
        reply_fn = MagicMock(side_effect=RuntimeError("send failed"))

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="test", chat_id="123")

        # Should not raise — the exception is caught
        executor._execute_task(task)
        assert not state.p0_active.is_set()


# ---------------------------------------------------------------------------
# TestCreateExecutor
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
        def capture_send(system, user):
            captured_messages.append(user)
            return "answer"

        client = MagicMock()
        client.send_message.side_effect = capture_send
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
        client.send_message.return_value = "answer"
        reply_fn = MagicMock()

        executor = TaskExecutor(state, client, ring2, reply_fn)
        task = Task(text="hello", chat_id="123")
        executor._execute_task(task)

        reply_fn.assert_called_once_with("answer")


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
        state = _make_state()
        result = create_executor(cfg, state, pathlib.Path("/tmp"), MagicMock())
        assert isinstance(result, TaskExecutor)
