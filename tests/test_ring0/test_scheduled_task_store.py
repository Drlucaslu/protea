"""Tests for ring0.scheduled_task_store."""

from __future__ import annotations

import time

import pytest

from ring0.scheduled_task_store import ScheduledTaskStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ScheduledTaskStore(tmp_path / "protea.db")


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_table(self, tmp_path):
        store = ScheduledTaskStore(tmp_path / "protea.db")
        assert store.count() == 0

    def test_idempotent_init(self, tmp_path):
        db = tmp_path / "protea.db"
        ScheduledTaskStore(db)
        ScheduledTaskStore(db)  # should not raise


# ---------------------------------------------------------------------------
# TestAdd
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_returns_schedule_id(self, store):
        sid = store.add("test", "echo hello", "*/5 * * * *")
        assert sid.startswith("sched-")

    def test_add_increments_count(self, store):
        store.add("a", "task a", "0 9 * * *")
        store.add("b", "task b", "0 18 * * *")
        assert store.count() == 2

    def test_add_with_chat_id(self, store):
        store.add("test", "echo", "* * * * *", chat_id="chat-123")
        task = store.get_by_name("test")
        assert task["chat_id"] == "chat-123"

    def test_add_once_type(self, store):
        store.add("once-test", "run once", "2026-12-31T23:59:00", schedule_type="once")
        task = store.get_by_name("once-test")
        assert task["schedule_type"] == "once"
        assert task["next_run_at"] is not None

    def test_add_computes_next_run(self, store):
        store.add("hourly", "check", "0 * * * *")
        task = store.get_by_name("hourly")
        assert task["next_run_at"] is not None
        assert task["next_run_at"] > time.time()

    def test_add_default_enabled(self, store):
        store.add("test", "task", "* * * * *")
        task = store.get_by_name("test")
        assert task["enabled"] == 1


# ---------------------------------------------------------------------------
# TestGetAll
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_empty(self, store):
        assert store.get_all() == []

    def test_returns_all_including_disabled(self, store):
        sid1 = store.add("a", "task a", "0 9 * * *")
        sid2 = store.add("b", "task b", "0 18 * * *")
        store.disable(sid2)
        assert len(store.get_all()) == 2


# ---------------------------------------------------------------------------
# TestGetEnabled
# ---------------------------------------------------------------------------

class TestGetEnabled:
    def test_empty(self, store):
        assert store.get_enabled() == []

    def test_excludes_disabled(self, store):
        sid1 = store.add("a", "task a", "0 9 * * *")
        sid2 = store.add("b", "task b", "0 18 * * *")
        store.disable(sid2)
        enabled = store.get_enabled()
        assert len(enabled) == 1
        assert enabled[0]["name"] == "a"


# ---------------------------------------------------------------------------
# TestGetDue
# ---------------------------------------------------------------------------

class TestGetDue:
    def test_empty(self, store):
        assert store.get_due() == []

    def test_returns_due_tasks(self, store):
        store.add("past", "should fire", "* * * * *")
        # Manually set next_run_at to the past
        task = store.get_by_name("past")
        with store._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE schedule_id = ?",
                (time.time() - 60, task["schedule_id"]),
            )
        due = store.get_due()
        assert len(due) == 1
        assert due[0]["name"] == "past"

    def test_excludes_future_tasks(self, store):
        store.add("future", "not yet", "0 0 1 1 *")  # Jan 1 at midnight
        due = store.get_due()
        assert len(due) == 0

    def test_excludes_disabled(self, store):
        sid = store.add("disabled", "nope", "* * * * *")
        with store._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE schedule_id = ?",
                (time.time() - 60, sid),
            )
        store.disable(sid)
        assert store.get_due() == []


# ---------------------------------------------------------------------------
# TestUpdateAfterRun
# ---------------------------------------------------------------------------

class TestUpdateAfterRun:
    def test_updates_fields(self, store):
        sid = store.add("test", "task", "0 * * * *")
        future = time.time() + 3600
        store.update_after_run(sid, future)
        task = store.get_by_id(sid)
        assert task["run_count"] == 1
        assert task["last_run_at"] is not None
        assert task["next_run_at"] == future

    def test_increments_run_count(self, store):
        sid = store.add("test", "task", "0 * * * *")
        store.update_after_run(sid, time.time() + 3600)
        store.update_after_run(sid, time.time() + 7200)
        task = store.get_by_id(sid)
        assert task["run_count"] == 2

    def test_null_next_run(self, store):
        sid = store.add("once", "task", "2026-12-31T23:59:00", schedule_type="once")
        store.update_after_run(sid, None)
        task = store.get_by_id(sid)
        assert task["next_run_at"] is None


# ---------------------------------------------------------------------------
# TestEnableDisable
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable(self, store):
        sid = store.add("test", "task", "* * * * *")
        assert store.disable(sid)
        task = store.get_by_id(sid)
        assert task["enabled"] == 0

    def test_enable(self, store):
        sid = store.add("test", "task", "* * * * *")
        store.disable(sid)
        assert store.enable(sid)
        task = store.get_by_id(sid)
        assert task["enabled"] == 1

    def test_disable_nonexistent_returns_false(self, store):
        assert not store.disable("nonexistent")

    def test_enable_nonexistent_returns_false(self, store):
        assert not store.enable("nonexistent")


# ---------------------------------------------------------------------------
# TestRemove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove(self, store):
        sid = store.add("test", "task", "* * * * *")
        assert store.remove(sid)
        assert store.count() == 0

    def test_remove_nonexistent(self, store):
        assert not store.remove("nonexistent")


# ---------------------------------------------------------------------------
# TestGetByName
# ---------------------------------------------------------------------------

class TestGetByName:
    def test_found(self, store):
        store.add("my-task", "hello", "0 9 * * *")
        task = store.get_by_name("my-task")
        assert task is not None
        assert task["name"] == "my-task"
        assert task["task_text"] == "hello"

    def test_not_found(self, store):
        assert store.get_by_name("nonexistent") is None


# ---------------------------------------------------------------------------
# TestGetById
# ---------------------------------------------------------------------------

class TestGetById:
    def test_found(self, store):
        sid = store.add("test", "task", "* * * * *")
        task = store.get_by_id(sid)
        assert task is not None
        assert task["schedule_id"] == sid

    def test_not_found(self, store):
        assert store.get_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_store(self, store):
        store.add("a", "task a", "* * * * *")
        store.add("b", "task b", "0 * * * *")
        store.clear()
        assert store.count() == 0
