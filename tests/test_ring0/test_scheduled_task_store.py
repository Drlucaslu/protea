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
# TestExpiry
# ---------------------------------------------------------------------------

class TestExpiry:
    def test_add_with_expires_at(self, store):
        expires = time.time() + 7200
        sid = store.add("expiring", "task", "*/10 * * * *", expires_at=expires)
        task = store.get_by_id(sid)
        assert task["expires_at"] is not None
        assert abs(task["expires_at"] - expires) < 1.0

    def test_get_due_filters_expired(self, store):
        """Expired tasks should not be returned by get_due."""
        past_expire = time.time() - 100
        sid = store.add("expired", "task", "* * * * *", expires_at=past_expire)
        # Set next_run_at to the past so it would be due
        with store._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE schedule_id = ?",
                (time.time() - 60, sid),
            )
        due = store.get_due()
        assert len(due) == 0

    def test_no_expiry_always_fires(self, store):
        """Tasks without expires_at are always eligible."""
        sid = store.add("no-expire", "task", "* * * * *")
        with store._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE schedule_id = ?",
                (time.time() - 60, sid),
            )
        due = store.get_due()
        assert len(due) == 1
        assert due[0]["name"] == "no-expire"

    def test_future_expiry_still_fires(self, store):
        """Tasks with future expires_at are still eligible."""
        future_expire = time.time() + 7200
        sid = store.add("future-expire", "task", "* * * * *", expires_at=future_expire)
        with store._connect() as con:
            con.execute(
                "UPDATE scheduled_tasks SET next_run_at = ? WHERE schedule_id = ?",
                (time.time() - 60, sid),
            )
        due = store.get_due()
        assert len(due) == 1
        assert due[0]["name"] == "future-expire"

    def test_migration_adds_column(self, tmp_path):
        """Opening a DB created without expires_at should auto-migrate."""
        db = tmp_path / "old.db"
        # Create a v1 schema without expires_at
        import sqlite3
        con = sqlite3.connect(str(db))
        con.execute("""\
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id            INTEGER PRIMARY KEY,
            schedule_id   TEXT    NOT NULL UNIQUE,
            name          TEXT    NOT NULL,
            task_text     TEXT    NOT NULL,
            chat_id       TEXT    NOT NULL DEFAULT '',
            cron_expr     TEXT    NOT NULL,
            schedule_type TEXT    NOT NULL DEFAULT 'cron',
            enabled       INTEGER NOT NULL DEFAULT 1,
            created_at    REAL    NOT NULL,
            last_run_at   REAL    DEFAULT NULL,
            next_run_at   REAL    DEFAULT NULL,
            run_count     INTEGER NOT NULL DEFAULT 0
        )""")
        con.execute(
            "INSERT INTO scheduled_tasks "
            "(schedule_id, name, task_text, chat_id, cron_expr, schedule_type, enabled, created_at, next_run_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
            ("sched-old", "old-task", "hello", "", "* * * * *", "cron", time.time(), time.time() - 60),
        )
        con.commit()
        con.close()

        # Open with the new store — should auto-migrate
        store = ScheduledTaskStore(db)
        task = store.get_by_name("old-task")
        assert task is not None
        assert task.get("expires_at") is None  # migrated column, default NULL

        # Can now add with expires_at
        sid = store.add("new", "task", "* * * * *", expires_at=time.time() + 3600)
        task2 = store.get_by_id(sid)
        assert task2["expires_at"] is not None


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_store(self, store):
        store.add("a", "task a", "* * * * *")
        store.add("b", "task b", "0 * * * *")
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# TestGetPublishable
# ---------------------------------------------------------------------------

class TestGetPublishable:
    def test_returns_tasks_with_enough_runs(self, store):
        sid1 = store.add("task-1", "News summary", "0 9 * * *")
        sid2 = store.add("task-2", "Weather check", "0 8 * * *")
        # Give task-1 two runs
        store.update_after_run(sid1, time.time() + 3600)
        store.update_after_run(sid1, time.time() + 7200)
        # Give task-2 only one run
        store.update_after_run(sid2, time.time() + 3600)

        publishable = store.get_publishable(min_runs=2)
        assert len(publishable) == 1
        assert publishable[0]["name"] == "task-1"

    def test_excludes_disabled(self, store):
        sid = store.add("disabled-task", "Do something", "0 9 * * *")
        store.update_after_run(sid, time.time() + 3600)
        store.update_after_run(sid, time.time() + 7200)
        store.disable(sid)
        publishable = store.get_publishable(min_runs=2)
        assert len(publishable) == 0

    def test_empty_store(self, store):
        assert store.get_publishable() == []


# ---------------------------------------------------------------------------
# TestExtractTemplate
# ---------------------------------------------------------------------------

class TestExtractTemplate:
    def test_basic_extraction(self, store):
        task = {
            "name": "daily-news",
            "task_text": "Summarize today's tech news",
            "cron_expr": "0 9 * * *",
            "schedule_type": "cron",
        }
        tmpl = store.extract_template(task)
        assert tmpl["name"] == "daily-news"
        assert tmpl["cron_expr"] == "0 9 * * *"
        assert tmpl["template_hash"]
        assert isinstance(tmpl["tags"], list)

    def test_tags_extraction(self, store):
        task = {
            "name": "weather",
            "task_text": "Check weather and send daily summary report",
            "cron_expr": "0 8 * * *",
            "schedule_type": "cron",
        }
        tmpl = store.extract_template(task)
        assert "weather" in tmpl["tags"]
        assert "daily" in tmpl["tags"]
        assert "summary" in tmpl["tags"]
        assert "report" in tmpl["tags"]
        assert "check" in tmpl["tags"]


# ---------------------------------------------------------------------------
# TestInstallFromTemplate
# ---------------------------------------------------------------------------

class TestInstallFromTemplate:
    def test_basic_install(self, store):
        template = {
            "name": "imported-news",
            "task_text": "Summarize news for {topic}",
            "cron_expr": "0 9 * * *",
            "schedule_type": "cron",
        }
        sid = store.install_from_template(template, params={"topic": "AI"}, chat_id="123")
        task = store.get_by_id(sid)
        assert task is not None
        assert task["task_text"] == "Summarize news for AI"
        assert task["chat_id"] == "123"

    def test_no_params(self, store):
        template = {
            "name": "simple",
            "task_text": "Check server status",
            "cron_expr": "*/30 * * * *",
            "schedule_type": "cron",
        }
        sid = store.install_from_template(template)
        task = store.get_by_id(sid)
        assert task["task_text"] == "Check server status"


# ---------------------------------------------------------------------------
# TestMarkTemplatePublished
# ---------------------------------------------------------------------------

class TestMarkTemplatePublished:
    def test_mark_published(self, store):
        sid = store.add("test", "task", "0 9 * * *")
        store.mark_template_published(sid, "abc123")
        task = store.get_by_id(sid)
        assert task["published_template_hash"] == "abc123"

    def test_published_hash_default_null(self, store):
        sid = store.add("test", "task", "0 9 * * *")
        task = store.get_by_id(sid)
        assert task.get("published_template_hash") is None
