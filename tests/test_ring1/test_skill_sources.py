"""Tests for ring1.skill_sources — federated skill discovery adapters."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ring1.skill_sources import (
    ClawHubSource,
    ProteaHubSource,
    SkillSource,
    SkillsShSource,
    _http_get,
)


class TestSkillSourceBase:
    """Base class returns empty defaults."""

    def test_search_returns_empty(self):
        source = SkillSource()
        assert source.search("query") == []

    def test_download_returns_none(self):
        source = SkillSource()
        assert source.download("name") is None

    def test_name_is_empty(self):
        source = SkillSource()
        assert source.name == ""


class TestProteaHubSource:
    """ProteaHubSource wraps RegistryClient."""

    def test_search_normalizes(self):
        client = MagicMock()
        client.search.return_value = [
            {"name": "greet", "node_id": "n1", "description": "d",
             "prompt_template": "t", "tags": ["web"], "source_code": "x=1"},
        ]
        source = ProteaHubSource(client)
        results = source.search("web")
        assert len(results) == 1
        assert results[0]["name"] == "greet"
        assert results[0]["source"] == "hub:protea-hub"
        client.search.assert_called_once_with(query="web", limit=10)

    def test_download_normalizes(self):
        client = MagicMock()
        client._request.return_value = {
            "name": "greet", "node_id": "n1", "description": "d",
            "prompt_template": "t", "tags": [], "source_code": "x=1",
        }
        source = ProteaHubSource(client)
        result = source.download("greet")
        assert result is not None
        assert result["source"] == "hub:protea-hub"

    def test_download_returns_none_on_failure(self):
        client = MagicMock()
        client._request.return_value = None
        source = ProteaHubSource(client)
        assert source.download("missing") is None

    def test_name(self):
        client = MagicMock()
        source = ProteaHubSource(client)
        assert source.name == "protea-hub"


class TestClawHubSource:
    """ClawHubSource fetches from OpenClaw registry."""

    @patch("ring1.skill_sources._http_get")
    def test_search(self, mock_get):
        mock_get.return_value = [
            {"name": "web-tool", "description": "Web utility",
             "prompt": "do stuff", "tags": ["web"], "code": "x=1"},
        ]
        source = ClawHubSource("https://example.com")
        results = source.search("web", limit=5)
        assert len(results) == 1
        assert results[0]["name"] == "web-tool"
        assert results[0]["source"] == "hub:clawhub"
        assert results[0]["prompt_template"] == "do stuff"
        assert results[0]["source_code"] == "x=1"

    @patch("ring1.skill_sources._http_get")
    def test_search_empty(self, mock_get):
        mock_get.return_value = None
        source = ClawHubSource()
        assert source.search("query") == []

    @patch("ring1.skill_sources._http_get")
    def test_download(self, mock_get):
        mock_get.return_value = {
            "name": "tool1", "description": "desc",
            "prompt": "p", "tags": ["t"], "code": "c",
        }
        source = ClawHubSource()
        result = source.download("tool1")
        assert result is not None
        assert result["source"] == "hub:clawhub"

    @patch("ring1.skill_sources._http_get")
    def test_download_not_found(self, mock_get):
        mock_get.return_value = None
        source = ClawHubSource()
        assert source.download("missing") is None

    def test_name(self):
        source = ClawHubSource()
        assert source.name == "clawhub"


class TestSkillsShSource:
    """SkillsShSource fetches from Skills.sh registry."""

    @patch("ring1.skill_sources._http_get")
    def test_search(self, mock_get):
        mock_get.return_value = [
            {"name": "agent-tool", "description": "Agent skill",
             "prompt": "do agent stuff", "tags": ["agent"], "code": "y=2"},
        ]
        source = SkillsShSource("https://skills.example.com")
        results = source.search("agent", limit=5)
        assert len(results) == 1
        assert results[0]["name"] == "agent-tool"
        assert results[0]["source"] == "hub:skills.sh"

    @patch("ring1.skill_sources._http_get")
    def test_search_returns_empty_on_error(self, mock_get):
        mock_get.return_value = None
        source = SkillsShSource()
        assert source.search("query") == []

    @patch("ring1.skill_sources._http_get")
    def test_download(self, mock_get):
        mock_get.return_value = {
            "name": "tool1", "author": "someone",
            "description": "desc", "prompt": "p", "code": "c",
        }
        source = SkillsShSource()
        result = source.download("tool1")
        assert result is not None
        assert result["node_id"] == "someone"
        assert result["source"] == "hub:skills.sh"

    def test_name(self):
        source = SkillsShSource()
        assert source.name == "skills.sh"


class TestHttpGet:
    """_http_get() should handle errors gracefully."""

    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"ok": True}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        result = _http_get("https://example.com/api")
        assert result == {"ok": True}

    @patch("urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")
        result = _http_get("https://example.com/api")
        assert result is None
