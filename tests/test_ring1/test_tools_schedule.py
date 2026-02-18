"""Tests for ring1.tools.schedule."""

from __future__ import annotations

import pathlib

import pytest

from ring0.scheduled_task_store import ScheduledTaskStore
from ring1.tools.schedule import make_schedule_tool


@pytest.fixture
def store(tmp_path: pathlib.Path):
    return ScheduledTaskStore(tmp_path / "test.db")


@pytest.fixture
def tool(store):
    return make_schedule_tool(store)


class TestScheduleToolSchema:
    def test_name(self, tool):
        assert tool.name == "manage_schedule"

    def test_required_fields(self, tool):
        assert "action" in tool.input_schema["required"]

    def test_action_enum(self, tool):
        actions = tool.input_schema["properties"]["action"]["enum"]
        assert set(actions) == {"create", "list", "remove", "enable", "disable"}


class TestCreate:
    def test_create_cron(self, tool, store):
        result = tool.execute({
            "action": "create",
            "name": "test_timer",
            "cron_expr": "*/5 * * * *",
            "task_text": "show current time",
        })
        assert "Schedule created" in result
        assert "test_timer" in result
        # Verify in store
        task = store.get_by_name("test_timer")
        assert task is not None
        assert task["task_text"] == "show current time"
        assert task["cron_expr"] == "*/5 * * * *"
        assert task["schedule_type"] == "cron"
        assert task["enabled"] == 1

    def test_create_once(self, tool, store):
        result = tool.execute({
            "action": "create",
            "name": "one_shot",
            "cron_expr": "2026-12-31T23:59:00",
            "schedule_type": "once",
            "task_text": "happy new year",
        })
        assert "Schedule created" in result
        task = store.get_by_name("one_shot")
        assert task is not None
        assert task["schedule_type"] == "once"

    def test_create_missing_name(self, tool):
        result = tool.execute({
            "action": "create",
            "cron_expr": "*/5 * * * *",
            "task_text": "test",
        })
        assert "Error" in result
        assert "name" in result

    def test_create_missing_cron_expr(self, tool):
        result = tool.execute({
            "action": "create",
            "name": "test",
            "task_text": "test",
        })
        assert "Error" in result
        assert "cron_expr" in result

    def test_create_missing_task_text(self, tool):
        result = tool.execute({
            "action": "create",
            "name": "test",
            "cron_expr": "*/5 * * * *",
        })
        assert "Error" in result
        assert "task_text" in result

    def test_create_invalid_cron(self, tool):
        result = tool.execute({
            "action": "create",
            "name": "bad",
            "cron_expr": "not a cron",
            "task_text": "test",
        })
        assert "Error" in result
        assert "invalid cron" in result

    def test_create_invalid_once_datetime(self, tool):
        result = tool.execute({
            "action": "create",
            "name": "bad",
            "cron_expr": "not-a-date",
            "schedule_type": "once",
            "task_text": "test",
        })
        assert "Error" in result
        assert "invalid ISO" in result

    def test_create_invalid_schedule_type(self, tool):
        result = tool.execute({
            "action": "create",
            "name": "bad",
            "cron_expr": "*/5 * * * *",
            "schedule_type": "weekly",
            "task_text": "test",
        })
        assert "Error" in result
        assert "schedule_type" in result

    def test_create_duplicate_name(self, tool):
        tool.execute({
            "action": "create",
            "name": "dup",
            "cron_expr": "*/5 * * * *",
            "task_text": "first",
        })
        result = tool.execute({
            "action": "create",
            "name": "dup",
            "cron_expr": "*/10 * * * *",
            "task_text": "second",
        })
        assert "Error" in result
        assert "already exists" in result


class TestList:
    def test_list_empty(self, tool):
        result = tool.execute({"action": "list"})
        assert "No scheduled tasks" in result

    def test_list_with_tasks(self, tool):
        tool.execute({
            "action": "create",
            "name": "timer_a",
            "cron_expr": "*/5 * * * *",
            "task_text": "task A",
        })
        tool.execute({
            "action": "create",
            "name": "timer_b",
            "cron_expr": "0 9 * * *",
            "task_text": "task B",
        })
        result = tool.execute({"action": "list"})
        assert "timer_a" in result
        assert "timer_b" in result
        assert "2" in result  # count


class TestRemove:
    def test_remove_existing(self, tool, store):
        tool.execute({
            "action": "create",
            "name": "to_remove",
            "cron_expr": "*/5 * * * *",
            "task_text": "test",
        })
        result = tool.execute({"action": "remove", "name": "to_remove"})
        assert "removed" in result
        assert store.get_by_name("to_remove") is None

    def test_remove_missing_name(self, tool):
        result = tool.execute({"action": "remove"})
        assert "Error" in result
        assert "name" in result

    def test_remove_not_found(self, tool):
        result = tool.execute({"action": "remove", "name": "nonexistent"})
        assert "Error" in result
        assert "no schedule" in result


class TestEnable:
    def test_enable(self, tool, store):
        tool.execute({
            "action": "create",
            "name": "toggle",
            "cron_expr": "*/5 * * * *",
            "task_text": "test",
        })
        tool.execute({"action": "disable", "name": "toggle"})
        assert store.get_by_name("toggle")["enabled"] == 0
        result = tool.execute({"action": "enable", "name": "toggle"})
        assert "enabled" in result
        assert store.get_by_name("toggle")["enabled"] == 1

    def test_enable_not_found(self, tool):
        result = tool.execute({"action": "enable", "name": "nope"})
        assert "Error" in result

    def test_enable_missing_name(self, tool):
        result = tool.execute({"action": "enable"})
        assert "Error" in result


class TestDisable:
    def test_disable(self, tool, store):
        tool.execute({
            "action": "create",
            "name": "toggle",
            "cron_expr": "*/5 * * * *",
            "task_text": "test",
        })
        result = tool.execute({"action": "disable", "name": "toggle"})
        assert "disabled" in result
        assert store.get_by_name("toggle")["enabled"] == 0

    def test_disable_not_found(self, tool):
        result = tool.execute({"action": "disable", "name": "nope"})
        assert "Error" in result

    def test_disable_missing_name(self, tool):
        result = tool.execute({"action": "disable"})
        assert "Error" in result


class TestUnknownAction:
    def test_unknown(self, tool):
        result = tool.execute({"action": "restart"})
        assert "Unknown action" in result
