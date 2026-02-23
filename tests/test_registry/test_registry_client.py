"""Tests for ring1.registry_client — RegistryClient with mocked HTTP."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from ring1.registry_client import RegistryClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict | list, status: int = 200) -> MagicMock:
    """Create a mock urllib response that works as a context manager."""
    raw = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = raw
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStats:
    @patch("urllib.request.urlopen")
    def test_get_stats(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"total_skills": 5, "total_nodes": 2, "total_downloads": 10}
        )
        client = RegistryClient("http://registry:8761", node_id="test-node")
        stats = client.get_stats()
        assert stats is not None
        assert stats["total_skills"] == 5
        assert stats["total_nodes"] == 2
        req = mock_urlopen.call_args[0][0]
        assert req.method == "GET"
        assert req.full_url == "http://registry:8761/api/stats"


class TestPublishTaskTemplate:
    @patch("urllib.request.urlopen")
    def test_publish_returns_dict(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"name": "daily-news", "node_id": "test-node"}
        )
        client = RegistryClient("http://registry:8761", node_id="test-node")
        result = client.publish_task_template(
            "daily-news", "Summarize today's news", "0 9 * * *",
            tags=["news", "daily"],
        )
        assert result is not None
        assert result["name"] == "daily-news"
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        assert req.full_url == "http://registry:8761/api/task-templates"
        body = json.loads(req.data.decode("utf-8"))
        assert body["node_id"] == "test-node"
        assert body["name"] == "daily-news"
        assert body["tags"] == ["news", "daily"]
        assert body["cron_expr"] == "0 9 * * *"

    @patch("urllib.request.urlopen")
    def test_publish_with_category(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"name": "t1"})
        client = RegistryClient("http://registry:8761", node_id="test-node")
        result = client.publish_task_template(
            "t1", "task", "0 8 * * *", category="monitoring",
        )
        assert result is not None
        body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert body["category"] == "monitoring"


class TestSearchTaskTemplates:
    @patch("urllib.request.urlopen")
    def test_search_all(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"name": "t1", "task_text": "News summary"},
            {"name": "t2", "task_text": "Weather check"},
        ])
        client = RegistryClient("http://registry:8761", node_id="test-node")
        results = client.search_task_templates()
        assert len(results) == 2
        req = mock_urlopen.call_args[0][0]
        assert req.method == "GET"
        assert "limit=20" in req.full_url

    @patch("urllib.request.urlopen")
    def test_search_query(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"name": "news-digest", "task_text": "News digest"},
        ])
        client = RegistryClient("http://registry:8761", node_id="test-node")
        results = client.search_task_templates(query="news")
        assert len(results) == 1
        req = mock_urlopen.call_args[0][0]
        assert "q=news" in req.full_url

    @patch("urllib.request.urlopen")
    def test_search_tag(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"name": "t1", "tags": ["daily"]},
        ])
        client = RegistryClient("http://registry:8761", node_id="test-node")
        results = client.search_task_templates(tag="daily")
        assert len(results) == 1
        req = mock_urlopen.call_args[0][0]
        assert "tag=daily" in req.full_url

    @patch("urllib.request.urlopen")
    def test_search_category(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([])
        client = RegistryClient("http://registry:8761", node_id="test-node")
        client.search_task_templates(category="monitoring")
        req = mock_urlopen.call_args[0][0]
        assert "category=monitoring" in req.full_url

    @patch("urllib.request.urlopen")
    def test_search_empty(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([])
        client = RegistryClient("http://registry:8761", node_id="test-node")
        results = client.search_task_templates()
        assert results == []


class TestDownloadTaskTemplate:
    @patch("urllib.request.urlopen")
    def test_download_returns_template(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"name": "daily-news", "downloads": 1}
        )
        client = RegistryClient("http://registry:8761", node_id="test-node")
        tmpl = client.download_task_template("other-node", "daily-news")
        assert tmpl is not None
        assert tmpl["name"] == "daily-news"
        req = mock_urlopen.call_args[0][0]
        assert req.method == "GET"
        assert req.full_url == "http://registry:8761/api/task-templates/other-node/daily-news"

    @patch("urllib.request.urlopen")
    def test_download_not_found(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs={}, fp=io.BytesIO(b""),
        )
        client = RegistryClient("http://registry:8761", node_id="test-node", timeout=1)
        assert client.download_task_template("x", "missing") is None


class TestConnectionError:
    def test_graceful_degradation(self):
        """Client should return None/[] when registry is unreachable."""
        client = RegistryClient("http://127.0.0.1:1", node_id="test", timeout=1)
        assert client.get_stats() is None
        assert client.publish_task_template("t", "text", "0 * * * *") is None
        assert client.search_task_templates() == []
        assert client.download_task_template("n", "t") is None
