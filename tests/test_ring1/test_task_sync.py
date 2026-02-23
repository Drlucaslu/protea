"""Tests for ring1.task_sync — TaskSyncer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ring1.task_sync import TaskSyncer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(name, run_count=3, published_hash=None, schedule_id=None):
    return {
        "schedule_id": schedule_id or f"sched-{name}",
        "name": name,
        "task_text": f"Do {name} task",
        "cron_expr": "0 9 * * *",
        "schedule_type": "cron",
        "enabled": 1,
        "run_count": run_count,
        "published_template_hash": published_hash,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSync:
    def test_full_sync_publishes_and_discovers(self):
        store = MagicMock()
        store.get_publishable.return_value = [
            _make_task("news", run_count=5),
        ]
        store.extract_template.return_value = {
            "name": "news",
            "task_text": "Do news task",
            "cron_expr": "0 9 * * *",
            "schedule_type": "cron",
            "template_hash": "abc123",
            "tags": ["news"],
        }
        store.get_all.return_value = [_make_task("news")]

        registry = MagicMock()
        registry.publish_task_template.return_value = {"name": "news"}
        registry.search_task_templates.return_value = [
            {"name": "weather-check", "task_text": "Check weather"},
        ]

        profiler = MagicMock()
        profiler.get_profile.return_value = {"interests": ["weather"]}

        syncer = TaskSyncer(store, registry, profiler, max_discover=5)
        result = syncer.sync()

        assert result["published"] == 1
        assert result["discovered"] == 1
        assert result["errors"] == 0
        store.mark_template_published.assert_called_once()

    def test_skips_already_published(self):
        store = MagicMock()
        store.get_publishable.return_value = [
            _make_task("news", published_hash="existing"),
        ]
        store.get_all.return_value = []

        registry = MagicMock()
        registry.search_task_templates.return_value = []

        syncer = TaskSyncer(store, registry, max_discover=5)
        result = syncer.sync()

        assert result["published"] == 0
        registry.publish_task_template.assert_not_called()

    def test_discover_skips_local(self):
        store = MagicMock()
        store.get_publishable.return_value = []
        store.get_all.return_value = [_make_task("news")]

        registry = MagicMock()
        registry.search_task_templates.return_value = [
            {"name": "news", "task_text": "News summary"},  # already local
            {"name": "weather", "task_text": "Weather check"},
        ]

        syncer = TaskSyncer(store, registry, max_discover=5)
        result = syncer.sync()

        assert result["discovered"] == 1
        assert len(syncer.discovered_templates) == 1
        assert syncer.discovered_templates[0]["name"] == "weather"

    def test_respects_max_discover(self):
        store = MagicMock()
        store.get_publishable.return_value = []
        store.get_all.return_value = []

        registry = MagicMock()
        registry.search_task_templates.return_value = [
            {"name": f"t{i}", "task_text": f"Task {i}"} for i in range(10)
        ]

        syncer = TaskSyncer(store, registry, max_discover=3)
        result = syncer.sync()

        assert result["discovered"] == 3

    def test_handles_publish_failure(self):
        store = MagicMock()
        store.get_publishable.return_value = [_make_task("failing")]
        store.extract_template.return_value = {
            "name": "failing", "task_text": "Do failing task",
            "cron_expr": "0 9 * * *", "schedule_type": "cron",
            "template_hash": "abc", "tags": [],
        }
        store.get_all.return_value = []

        registry = MagicMock()
        registry.publish_task_template.return_value = None  # failed
        registry.search_task_templates.return_value = []

        syncer = TaskSyncer(store, registry, max_discover=5)
        result = syncer.sync()

        assert result["published"] == 0
        store.mark_template_published.assert_not_called()

    def test_no_profiler_uses_default_query(self):
        store = MagicMock()
        store.get_publishable.return_value = []
        store.get_all.return_value = []

        registry = MagicMock()
        registry.search_task_templates.return_value = []

        syncer = TaskSyncer(store, registry, user_profiler=None, max_discover=5)
        result = syncer.sync()

        # Should fall back to "daily" query.
        registry.search_task_templates.assert_called_once_with(query="daily", limit=10)
