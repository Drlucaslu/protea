"""Tests for ring0.memory — MemoryStore."""

from __future__ import annotations

from ring0.memory import MemoryStore


class TestAdd:
    """add() should insert rows and return their rowid."""

    def test_insert_returns_rowid(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        rid = store.add(1, "observation", "Gen 1 survived 60s")
        assert rid == 1

    def test_successive_inserts_increment_rowid(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        r1 = store.add(1, "observation", "first")
        r2 = store.add(2, "reflection", "second")
        assert r2 == r1 + 1

    def test_metadata_stored_as_json(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test", metadata={"key": "value", "n": 42})
        entries = store.get_recent(1)
        assert len(entries) == 1
        assert entries[0]["metadata"] == {"key": "value", "n": 42}

    def test_metadata_default_empty_dict(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test")
        entries = store.get_recent(1)
        assert entries[0]["metadata"] == {}


class TestGetRecent:
    """get_recent() should return entries in reverse chronological order."""

    def test_returns_reverse_order(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "first")
        store.add(2, "reflection", "second")
        store.add(3, "directive", "third")

        recent = store.get_recent(10)
        contents = [e["content"] for e in recent]
        assert contents == ["third", "second", "first"]

    def test_respects_limit(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        for i in range(10):
            store.add(i, "observation", f"entry-{i}")

        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[0]["content"] == "entry-9"

    def test_empty_database_returns_empty_list(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        assert store.get_recent() == []

    def test_returns_all_fields(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(5, "reflection", "interesting pattern")
        entries = store.get_recent(1)
        e = entries[0]
        assert e["generation"] == 5
        assert e["entry_type"] == "reflection"
        assert e["content"] == "interesting pattern"
        assert "timestamp" in e
        assert "id" in e


class TestGetByType:
    """get_by_type() should filter entries by type."""

    def test_filters_by_type(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "obs1")
        store.add(2, "reflection", "ref1")
        store.add(3, "observation", "obs2")
        store.add(4, "directive", "dir1")

        obs = store.get_by_type("observation")
        assert len(obs) == 2
        assert all(e["entry_type"] == "observation" for e in obs)

    def test_returns_most_recent_first(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "first")
        store.add(2, "reflection", "second")

        refs = store.get_by_type("reflection")
        assert refs[0]["content"] == "second"
        assert refs[1]["content"] == "first"

    def test_respects_limit(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        for i in range(10):
            store.add(i, "observation", f"obs-{i}")

        obs = store.get_by_type("observation", limit=3)
        assert len(obs) == 3

    def test_empty_for_missing_type(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test")
        assert store.get_by_type("reflection") == []


class TestCount:
    """count() should return total number of entries."""

    def test_empty_database(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        assert store.count() == 0

    def test_after_inserts(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "a")
        store.add(2, "reflection", "b")
        store.add(3, "directive", "c")
        assert store.count() == 3


class TestClear:
    """clear() should delete all entries."""

    def test_clears_all(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "a")
        store.add(2, "reflection", "b")
        assert store.count() == 2

        store.clear()
        assert store.count() == 0
        assert store.get_recent() == []

    def test_clear_empty_database(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.clear()  # Should not raise
        assert store.count() == 0

    def test_add_after_clear(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "before")
        store.clear()
        store.add(2, "observation", "after")
        assert store.count() == 1
        assert store.get_recent(1)[0]["content"] == "after"


class TestSharedDatabase:
    """MemoryStore should coexist with FitnessTracker in same db."""

    def test_coexists_with_fitness(self, tmp_path):
        from ring0.fitness import FitnessTracker

        db_path = tmp_path / "protea.db"
        fitness = FitnessTracker(db_path)
        memory = MemoryStore(db_path)

        fitness.record(1, "abc", 0.9, 60.0, True)
        memory.add(1, "observation", "survived")

        assert len(fitness.get_history()) == 1
        assert memory.count() == 1
