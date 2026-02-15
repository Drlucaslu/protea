"""Tests for registry.store — RegistryStore."""

from __future__ import annotations

import pytest

from registry.store import RegistryStore


class TestInit:
    """Table creation should be idempotent."""

    def test_creates_table(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        assert store.stats()["total_skills"] == 0

    def test_idempotent(self, tmp_path):
        db = tmp_path / "reg.db"
        RegistryStore(db)
        store2 = RegistryStore(db)
        assert store2.stats()["total_skills"] == 0


class TestPublish:
    """publish() should insert and update skills."""

    def test_insert_returns_rowid(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        rid = store.publish("node1", "greet", "Greeting", "Hello {{name}}")
        assert rid >= 1

    def test_initial_version_is_one(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("node1", "greet", "Greeting", "Hello")
        skill = store.get("node1", "greet")
        assert skill["version"] == 1

    def test_update_bumps_version(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("node1", "greet", "Greeting v1", "Hello v1")
        store.publish("node1", "greet", "Greeting v2", "Hello v2")
        skill = store.get("node1", "greet")
        assert skill["version"] == 2
        assert skill["description"] == "Greeting v2"

    def test_different_nodes_same_name(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("node1", "greet", "Node1 greeting", "Hello from node1")
        store.publish("node2", "greet", "Node2 greeting", "Hello from node2")
        s1 = store.get("node1", "greet")
        s2 = store.get("node2", "greet")
        assert s1["description"] == "Node1 greeting"
        assert s2["description"] == "Node2 greeting"

    def test_parameters_stored_as_json(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t", parameters={"key": "val"})
        skill = store.get("n", "s")
        assert skill["parameters"] == {"key": "val"}

    def test_tags_stored_as_json(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t", tags=["web", "api"])
        skill = store.get("n", "s")
        assert skill["tags"] == ["web", "api"]

    def test_defaults(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t")
        skill = store.get("n", "s")
        assert skill["parameters"] == {}
        assert skill["tags"] == []
        assert skill["downloads"] == 0
        assert skill["rating_up"] == 0
        assert skill["rating_down"] == 0


class TestSearch:
    """search() should support keyword and tag filtering."""

    def test_empty_returns_all(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "alpha", "Alpha skill", "t")
        store.publish("n", "beta", "Beta skill", "t")
        results = store.search()
        assert len(results) == 2

    def test_keyword_search_name(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "web-scraper", "Scrapes web", "t")
        store.publish("n", "data-viz", "Visualize data", "t")
        results = store.search(query="scraper")
        assert len(results) == 1
        assert results[0]["name"] == "web-scraper"

    def test_keyword_search_description(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "tool1", "Machine learning helper", "t")
        store.publish("n", "tool2", "File manager", "t")
        results = store.search(query="learning")
        assert len(results) == 1
        assert results[0]["name"] == "tool1"

    def test_tag_filter(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s1", "d", "t", tags=["web"])
        store.publish("n", "s2", "d", "t", tags=["data"])
        results = store.search(tags=["web"])
        assert len(results) == 1
        assert results[0]["name"] == "s1"

    def test_keyword_and_tag(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "web-tool", "Web utility", "t", tags=["web"])
        store.publish("n", "web-other", "Web other", "t", tags=["other"])
        store.publish("n", "data-tool", "Data utility", "t", tags=["web"])
        results = store.search(query="web", tags=["web"])
        assert len(results) == 1
        assert results[0]["name"] == "web-tool"

    def test_limit(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        for i in range(10):
            store.publish("n", f"skill-{i}", "d", "t")
        results = store.search(limit=3)
        assert len(results) == 3

    def test_ordered_by_downloads(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "popular", "d", "t")
        store.publish("n", "obscure", "d", "t")
        for _ in range(5):
            store.increment_downloads("n", "popular")
        results = store.search()
        assert results[0]["name"] == "popular"


class TestGet:
    """get() should return a skill or None."""

    def test_found(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "greet", "Greeting", "Hello")
        skill = store.get("n", "greet")
        assert skill is not None
        assert skill["name"] == "greet"

    def test_not_found(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        assert store.get("n", "missing") is None


class TestDownloads:
    """increment_downloads() should bump the counter."""

    def test_increment(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t")
        assert store.get("n", "s")["downloads"] == 0
        store.increment_downloads("n", "s")
        assert store.get("n", "s")["downloads"] == 1
        store.increment_downloads("n", "s")
        assert store.get("n", "s")["downloads"] == 2


class TestRate:
    """rate() should increment rating_up or rating_down."""

    def test_up(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t")
        store.rate("n", "s", up=True)
        store.rate("n", "s", up=True)
        skill = store.get("n", "s")
        assert skill["rating_up"] == 2
        assert skill["rating_down"] == 0

    def test_down(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t")
        store.rate("n", "s", up=False)
        skill = store.get("n", "s")
        assert skill["rating_up"] == 0
        assert skill["rating_down"] == 1


class TestDelete:
    """delete() should remove a skill."""

    def test_existing(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("n", "s", "d", "t")
        assert store.delete("n", "s") is True
        assert store.get("n", "s") is None

    def test_nonexistent(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        assert store.delete("n", "missing") is False


class TestStats:
    """stats() should return correct counts."""

    def test_empty(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        s = store.stats()
        assert s == {"total_skills": 0, "total_nodes": 0, "total_downloads": 0}

    def test_with_data(self, tmp_path):
        store = RegistryStore(tmp_path / "reg.db")
        store.publish("node1", "s1", "d", "t")
        store.publish("node1", "s2", "d", "t")
        store.publish("node2", "s3", "d", "t")
        store.increment_downloads("node1", "s1")
        store.increment_downloads("node1", "s1")
        store.increment_downloads("node2", "s3")
        s = store.stats()
        assert s["total_skills"] == 3
        assert s["total_nodes"] == 2
        assert s["total_downloads"] == 3
