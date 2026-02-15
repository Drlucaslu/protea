"""Tests for registry.server — RegistryServer HTTP API."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from registry.server import RegistryServer
from registry.store import RegistryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def server(tmp_path):
    """Start a RegistryServer on an OS-assigned port and yield (url, store)."""
    store = RegistryStore(tmp_path / "reg.db")
    srv = RegistryServer(store, host="127.0.0.1", port=0)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    # Wait for server to bind.
    for _ in range(50):
        if srv.actual_port != 0:
            break
        time.sleep(0.05)
    url = f"http://127.0.0.1:{srv.actual_port}"
    yield url, store
    srv.stop()


def _get(url: str) -> tuple[int, dict | list | str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(data)
            except json.JSONDecodeError:
                return resp.status, data
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(data)
        except json.JSONDecodeError:
            return exc.code, data


def _post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _delete(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAPISkills:
    """GET /api/skills and POST /api/skills."""

    def test_empty_list(self, server):
        url, store = server
        code, body = _get(f"{url}/api/skills")
        assert code == 200
        assert body == []

    def test_publish_and_list(self, server):
        url, store = server
        code, body = _post(f"{url}/api/skills", {
            "node_id": "n1",
            "name": "greet",
            "description": "Greeting skill",
            "prompt_template": "Hello {{name}}",
            "tags": ["greeting"],
        })
        assert code == 201
        assert body["name"] == "greet"
        assert body["version"] == 1

        code, skills = _get(f"{url}/api/skills")
        assert code == 200
        assert len(skills) == 1
        assert skills[0]["name"] == "greet"

    def test_search_query(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "web-tool", "description": "Web utility", "prompt_template": "t"})
        _post(f"{url}/api/skills", {"node_id": "n", "name": "data-tool", "description": "Data utility", "prompt_template": "t"})
        code, results = _get(f"{url}/api/skills?q=web")
        assert code == 200
        assert len(results) == 1
        assert results[0]["name"] == "web-tool"

    def test_search_tag(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s1", "description": "d", "prompt_template": "t", "tags": ["api"]})
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s2", "description": "d", "prompt_template": "t", "tags": ["web"]})
        code, results = _get(f"{url}/api/skills?tag=api")
        assert code == 200
        assert len(results) == 1
        assert results[0]["name"] == "s1"

    def test_publish_missing_fields(self, server):
        url, store = server
        code, body = _post(f"{url}/api/skills", {"name": "test"})
        assert code == 400

    def test_publish_update_version(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s", "description": "v1", "prompt_template": "t"})
        code, body = _post(f"{url}/api/skills", {"node_id": "n", "name": "s", "description": "v2", "prompt_template": "t"})
        assert code == 201
        assert body["version"] == 2


class TestAPISkillDetail:
    """GET /api/skills/<node_id>/<name>."""

    def test_get_increments_downloads(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "greet", "description": "d", "prompt_template": "t"})
        code, skill = _get(f"{url}/api/skills/n/greet")
        assert code == 200
        assert skill["name"] == "greet"
        assert skill["downloads"] == 1

        code, skill = _get(f"{url}/api/skills/n/greet")
        assert skill["downloads"] == 2

    def test_not_found(self, server):
        url, store = server
        code, body = _get(f"{url}/api/skills/n/missing")
        assert code == 404


class TestAPIRate:
    """POST /api/skills/<node_id>/<name>/rate."""

    def test_rate_up(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s", "description": "d", "prompt_template": "t"})
        code, body = _post(f"{url}/api/skills/n/s/rate", {"up": True})
        assert code == 200
        assert body["ok"] is True
        skill = store.get("n", "s")
        assert skill["rating_up"] == 1

    def test_rate_down(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s", "description": "d", "prompt_template": "t"})
        _post(f"{url}/api/skills/n/s/rate", {"up": False})
        skill = store.get("n", "s")
        assert skill["rating_down"] == 1


class TestAPIDelete:
    """DELETE /api/skills/<node_id>/<name>."""

    def test_delete_existing(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n", "name": "s", "description": "d", "prompt_template": "t"})
        code, body = _delete(f"{url}/api/skills/n/s")
        assert code == 200
        assert body["ok"] is True
        assert store.get("n", "s") is None

    def test_delete_nonexistent(self, server):
        url, store = server
        code, body = _delete(f"{url}/api/skills/n/missing")
        assert code == 404


class TestAPIStats:
    """GET /api/stats."""

    def test_stats(self, server):
        url, store = server
        _post(f"{url}/api/skills", {"node_id": "n1", "name": "s1", "description": "d", "prompt_template": "t"})
        _post(f"{url}/api/skills", {"node_id": "n2", "name": "s2", "description": "d", "prompt_template": "t"})
        code, stats = _get(f"{url}/api/stats")
        assert code == 200
        assert stats["total_skills"] == 2
        assert stats["total_nodes"] == 2


class TestDashboard:
    """GET / should return HTML."""

    def test_dashboard_html(self, server):
        url, store = server
        code, body = _get(url)
        assert code == 200
        assert isinstance(body, str)
        assert "Protea Skill Registry" in body
