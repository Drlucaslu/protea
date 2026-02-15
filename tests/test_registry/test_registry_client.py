"""Tests for ring1.registry_client — RegistryClient against a live server."""

from __future__ import annotations

import threading
import time

import pytest

from registry.server import RegistryServer
from registry.store import RegistryStore
from ring1.registry_client import RegistryClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_server(tmp_path):
    """Start a temporary RegistryServer and return (client, store)."""
    store = RegistryStore(tmp_path / "reg.db")
    srv = RegistryServer(store, host="127.0.0.1", port=0)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(50):
        if srv.actual_port != 0:
            break
        time.sleep(0.05)
    url = f"http://127.0.0.1:{srv.actual_port}"
    client = RegistryClient(url, node_id="test-node")
    yield client, store
    srv.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_returns_dict(self, live_server):
        client, store = live_server
        result = client.publish("greet", "Greeting skill", "Hello {{name}}", tags=["greeting"])
        assert result is not None
        assert result["name"] == "greet"
        assert result["version"] == 1

    def test_publish_update_bumps_version(self, live_server):
        client, store = live_server
        client.publish("greet", "v1", "t1")
        result = client.publish("greet", "v2", "t2")
        assert result["version"] == 2


class TestSearch:
    def test_search_all(self, live_server):
        client, store = live_server
        client.publish("alpha", "Alpha skill", "t")
        client.publish("beta", "Beta skill", "t")
        results = client.search()
        assert len(results) == 2

    def test_search_query(self, live_server):
        client, store = live_server
        client.publish("web-tool", "Web utility", "t")
        client.publish("data-tool", "Data utility", "t")
        results = client.search(query="web")
        assert len(results) == 1
        assert results[0]["name"] == "web-tool"

    def test_search_tag(self, live_server):
        client, store = live_server
        client.publish("s1", "d", "t", tags=["api"])
        client.publish("s2", "d", "t", tags=["web"])
        results = client.search(tag="api")
        assert len(results) == 1
        assert results[0]["name"] == "s1"


class TestDownload:
    def test_download_increments_count(self, live_server):
        client, store = live_server
        client.publish("greet", "d", "t")
        skill = client.download("test-node", "greet")
        assert skill is not None
        assert skill["downloads"] == 1
        skill = client.download("test-node", "greet")
        assert skill["downloads"] == 2

    def test_download_not_found(self, live_server):
        client, store = live_server
        assert client.download("x", "missing") is None


class TestRate:
    def test_rate(self, live_server):
        client, store = live_server
        client.publish("s", "d", "t")
        assert client.rate("test-node", "s", up=True) is True
        skill = store.get("test-node", "s")
        assert skill["rating_up"] == 1


class TestUnpublish:
    def test_unpublish(self, live_server):
        client, store = live_server
        client.publish("s", "d", "t")
        assert client.unpublish("s") is True
        assert store.get("test-node", "s") is None

    def test_unpublish_nonexistent(self, live_server):
        client, store = live_server
        assert client.unpublish("missing") is False


class TestStats:
    def test_get_stats(self, live_server):
        client, store = live_server
        client.publish("s1", "d", "t")
        stats = client.get_stats()
        assert stats is not None
        assert stats["total_skills"] == 1


class TestConnectionError:
    def test_graceful_degradation(self):
        """Client should return None/[] when registry is unreachable."""
        client = RegistryClient("http://127.0.0.1:1", node_id="test", timeout=1)
        assert client.publish("s", "d", "t") is None
        assert client.search() == []
        assert client.download("n", "s") is None
        assert client.rate("n", "s") is False
        assert client.unpublish("s") is False
        assert client.get_stats() is None
