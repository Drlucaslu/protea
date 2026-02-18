"""Tests for ring1.dashboard — Dashboard server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from unittest.mock import MagicMock

from ring1.dashboard import (
    Dashboard,
    DashboardHandler,
    _render_fitness_svg,
    _render_category_bars_svg,
    create_dashboard,
)


class TestRenderFitnessSvg:
    def test_empty_history(self):
        result = _render_fitness_svg([])
        assert "No fitness data" in result

    def test_single_entry(self):
        history = [{"generation": 1, "score": 0.75, "survived": True}]
        svg = _render_fitness_svg(history)
        assert "<svg" in svg
        assert "circle" in svg

    def test_multiple_entries(self):
        history = [
            {"generation": i, "score": 0.5 + i * 0.05, "survived": i % 2 == 0}
            for i in range(5)
        ]
        svg = _render_fitness_svg(history)
        assert "<svg" in svg
        assert "polyline" in svg
        assert svg.count("circle") == 5


class TestRenderCategoryBarsSvg:
    def test_empty_categories(self):
        result = _render_category_bars_svg({})
        assert "No profile data" in result

    def test_single_category(self):
        svg = _render_category_bars_svg({"coding": 10.0})
        assert "<svg" in svg
        assert "coding" in svg

    def test_multiple_categories(self):
        svg = _render_category_bars_svg({"coding": 10.0, "data": 5.0, "web": 3.0})
        assert "<svg" in svg
        assert "coding" in svg
        assert "data" in svg


class TestCreateDashboard:
    def test_disabled(self):
        cfg = {"ring1": {"dashboard": {"enabled": False}}}
        assert create_dashboard(MagicMock(), cfg) is None

    def test_missing_config(self):
        assert create_dashboard(MagicMock(), {}) is None

    def test_enabled(self):
        cfg = {"ring1": {"dashboard": {"enabled": True, "port": 0}}}
        dashboard = create_dashboard(MagicMock(), cfg)
        assert isinstance(dashboard, Dashboard)


class TestDashboardServer:
    """Integration tests for the HTTP server."""

    def _start(self, **data_sources):
        """Start a dashboard on a random port and return (dashboard, base_url)."""
        dashboard = Dashboard(host="127.0.0.1", port=0, **data_sources)
        thread = threading.Thread(target=dashboard.run, daemon=True)
        thread.start()
        # Wait for server to start.
        for _ in range(50):
            time.sleep(0.05)
            if dashboard._server is not None:
                break
        port = dashboard.actual_port
        return dashboard, f"http://127.0.0.1:{port}"

    def _get(self, url: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def _get_json(self, url: str):
        code, body = self._get(url)
        return code, json.loads(body) if code == 200 else body

    def test_overview_page(self):
        dashboard, base = self._start()
        try:
            code, body = self._get(f"{base}/")
            assert code == 200
            assert "Protea Dashboard" in body
            assert "Memory" in body
        finally:
            dashboard.stop()

    def test_memory_page(self):
        mock_memory = MagicMock()
        mock_memory.get_recent.return_value = [
            {"id": 1, "generation": 5, "entry_type": "task", "tier": "hot",
             "importance": 0.7, "content": "test task", "timestamp": "2026-01-01",
             "metadata": {}, "keywords": "test"},
        ]
        dashboard, base = self._start(memory_store=mock_memory)
        try:
            code, body = self._get(f"{base}/memory")
            assert code == 200
            assert "Memory Browser" in body
            assert "test task" in body
        finally:
            dashboard.stop()

    def test_skills_page(self):
        mock_skills = MagicMock()
        mock_skills.get_active.return_value = [
            {"name": "web_dash", "description": "Web dashboard", "tags": ["web"],
             "usage_count": 3, "source": "crystallized"},
        ]
        dashboard, base = self._start(skill_store=mock_skills)
        try:
            code, body = self._get(f"{base}/skills")
            assert code == 200
            assert "web_dash" in body
        finally:
            dashboard.stop()

    def test_intent_page(self):
        mock_memory = MagicMock()
        mock_memory.get_by_type.return_value = [
            {"generation": 10, "content": "optimize: survived",
             "metadata": {"blast_radius": {"scope": "minor", "lines_changed": 5}},
             "entry_type": "evolution_intent"},
        ]
        dashboard, base = self._start(memory_store=mock_memory)
        try:
            code, body = self._get(f"{base}/intent")
            assert code == 200
            assert "OPTIMIZE" in body
        finally:
            dashboard.stop()

    def test_profile_page(self):
        mock_profiler = MagicMock()
        mock_profiler.get_category_distribution.return_value = {"coding": 10.0, "data": 5.0}
        mock_profiler.get_stats.return_value = {
            "interaction_count": 47,
            "topic_count": 15,
            "earliest_interaction": "2026-01-15",
            "latest_interaction": "2026-02-17",
        }
        mock_profiler.get_top_topics.return_value = [
            {"topic": "python", "category": "coding", "weight": 5.0, "hit_count": 10},
        ]
        dashboard, base = self._start(user_profiler=mock_profiler)
        try:
            code, body = self._get(f"{base}/profile")
            assert code == 200
            assert "python" in body
            assert "47" in body  # interaction count
        finally:
            dashboard.stop()

    def test_api_status(self):
        dashboard, base = self._start()
        try:
            code, data = self._get_json(f"{base}/api/status")
            assert code == 200
            assert data["dashboard"] == "running"
        finally:
            dashboard.stop()

    def test_api_memory_stats(self):
        mock_memory = MagicMock()
        mock_memory.get_stats.return_value = {"total": 42, "by_tier": {"hot": 10}, "by_type": {"task": 5}}
        dashboard, base = self._start(memory_store=mock_memory)
        try:
            code, data = self._get_json(f"{base}/api/memory/stats")
            assert code == 200
            assert data["total"] == 42
        finally:
            dashboard.stop()

    def test_api_profile(self):
        mock_profiler = MagicMock()
        mock_profiler.get_category_distribution.return_value = {"coding": 10.0}
        mock_profiler.get_stats.return_value = {"interaction_count": 5}
        mock_profiler.get_top_topics.return_value = []
        dashboard, base = self._start(user_profiler=mock_profiler)
        try:
            code, data = self._get_json(f"{base}/api/profile")
            assert code == 200
            assert "coding" in data["categories"]
        finally:
            dashboard.stop()

    def test_404(self):
        dashboard, base = self._start()
        try:
            code, _ = self._get(f"{base}/nonexistent")
            assert code == 404
        finally:
            dashboard.stop()

    def test_no_data_sources(self):
        """Dashboard should work with no data sources (all None)."""
        dashboard, base = self._start()
        try:
            code, body = self._get(f"{base}/")
            assert code == 200
            code, body = self._get(f"{base}/memory")
            assert code == 200
            code, body = self._get(f"{base}/skills")
            assert code == 200
            code, body = self._get(f"{base}/intent")
            assert code == 200
            code, body = self._get(f"{base}/profile")
            assert code == 200
        finally:
            dashboard.stop()
