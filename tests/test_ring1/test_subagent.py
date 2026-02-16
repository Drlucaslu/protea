"""Tests for ring1.subagent."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from ring1.subagent import SubagentManager, SubagentResult
from ring1.tool_registry import Tool, ToolRegistry


def _make_registry() -> ToolRegistry:
    """Create a minimal registry with a test tool."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="Echo back input",
        input_schema={"type": "object", "properties": {}, "required": []},
        execute=lambda inp: "echoed",
    ))
    reg.register(Tool(
        name="spawn",
        description="Spawn tool (should be removed in subagent)",
        input_schema={"type": "object", "properties": {}, "required": []},
        execute=lambda inp: "spawned",
    ))
    reg.register(Tool(
        name="message",
        description="Message tool (should be removed in subagent)",
        input_schema={"type": "object", "properties": {}, "required": []},
        execute=lambda inp: "messaged",
    ))
    return reg


class TestSubagentResult:
    def test_to_dict(self):
        r = SubagentResult("bg-123", "test task")
        d = r.to_dict()
        assert d["task_id"] == "bg-123"
        assert d["description"] == "test task"
        assert d["done"] is False

    def test_done_flag(self):
        r = SubagentResult("bg-123", "test task")
        assert not r.done.is_set()
        r.done.set()
        assert r.to_dict()["done"] is True


class TestSubagentManager:
    def test_spawn_returns_status(self):
        client = MagicMock()
        client.send_message_with_tools.return_value = "result"
        reply_fn = MagicMock()
        reg = _make_registry()

        mgr = SubagentManager(client, reg, reply_fn)
        result = mgr.spawn("Analyze code")
        assert "Background task started" in result
        assert "Analyze code" in result

    def test_spawn_completes_and_notifies(self):
        client = MagicMock()
        client.send_message_with_tools.return_value = "Analysis complete"
        reply_fn = MagicMock()
        reg = _make_registry()

        mgr = SubagentManager(client, reg, reply_fn)
        mgr.spawn("Analyze code")

        # Wait for completion
        deadline = time.time() + 5
        while time.time() < deadline:
            tasks = mgr.get_active()
            if tasks and tasks[0]["done"]:
                break
            time.sleep(0.1)

        tasks = mgr.get_active()
        assert len(tasks) == 1
        assert tasks[0]["done"] is True

        # Should have notified via reply_fn
        assert reply_fn.called
        call_text = reply_fn.call_args[0][0]
        assert "Background Task Complete" in call_text
        assert "Analysis complete" in call_text

    def test_concurrent_limit(self):
        client = MagicMock()
        # Make the LLM call slow so tasks stay active
        def slow_response(*args, **kwargs):
            time.sleep(2)
            return "done"
        client.send_message_with_tools.side_effect = slow_response
        reply_fn = MagicMock()
        reg = _make_registry()

        mgr = SubagentManager(client, reg, reply_fn, max_concurrent=2)

        r1 = mgr.spawn("task 1")
        r2 = mgr.spawn("task 2")
        assert "started" in r1
        assert "started" in r2

        # Third should be rejected
        r3 = mgr.spawn("task 3")
        assert "Cannot spawn" in r3

    def test_tool_isolation(self):
        """Subagent should not have access to spawn or message tools."""
        client = MagicMock()
        captured_tools = []

        def capture_tools(system, user, tools, tool_executor, **kwargs):
            captured_tools.extend([t["name"] for t in tools])
            return "done"

        client.send_message_with_tools.side_effect = capture_tools
        reply_fn = MagicMock()
        reg = _make_registry()

        mgr = SubagentManager(client, reg, reply_fn)
        mgr.spawn("test")

        # Wait for completion
        deadline = time.time() + 5
        while time.time() < deadline:
            tasks = mgr.get_active()
            if tasks and tasks[0]["done"]:
                break
            time.sleep(0.1)

        assert "echo" in captured_tools
        assert "spawn" not in captured_tools
        assert "message" not in captured_tools  # excluded to prevent direct messaging

    def test_get_active_empty(self):
        client = MagicMock()
        reply_fn = MagicMock()
        reg = _make_registry()
        mgr = SubagentManager(client, reg, reply_fn)
        assert mgr.get_active() == []

    def test_llm_error_handled(self):
        from ring1.llm_client import LLMError

        client = MagicMock()
        client.send_message_with_tools.side_effect = LLMError("rate limited")
        reply_fn = MagicMock()
        reg = _make_registry()

        mgr = SubagentManager(client, reg, reply_fn)
        mgr.spawn("test")

        deadline = time.time() + 5
        while time.time() < deadline:
            tasks = mgr.get_active()
            if tasks and tasks[0]["done"]:
                break
            time.sleep(0.1)

        tasks = mgr.get_active()
        assert tasks[0]["done"] is True
        assert tasks[0]["error"] == "rate limited"

        # Should still notify
        assert reply_fn.called
