"""Tests for status snapshots, stale detection, and conflict handling."""

from __future__ import annotations

import json
import sqlite3

from ring0.memory import MemoryStore


class TestStatusSnapshot:
    """Status snapshot entry type."""

    def test_add_status_snapshot(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        rid = store.add(10, "status_snapshot", "Generation: 10\nScore: 0.850\nSurvived: yes")
        assert rid > 0
        entry = store.get_recent(1)[0]
        assert entry["entry_type"] == "status_snapshot"
        assert entry["importance"] == 0.6

    def test_get_by_type_returns_snapshots(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(10, "status_snapshot", "Generation: 10\nScore: 0.850")
        store.add(11, "status_snapshot", "Generation: 11\nScore: 0.900")
        store.add(12, "observation", "some other observation entry")

        snaps = store.get_by_type("status_snapshot")
        assert len(snaps) == 2
        assert all(s["entry_type"] == "status_snapshot" for s in snaps)

    def test_compact_keeps_recent_snapshots(self, tmp_path):
        """Latest 3 status_snapshots stay in hot tier."""
        store = MemoryStore(tmp_path / "mem.db")
        for i in range(5):
            store.add(i + 1, "status_snapshot", f"Generation: {i+1}\nScore: 0.8{i}0")

        store.compact(current_generation=50)
        hot = store.get_by_tier("hot")
        hot_snaps = [e for e in hot if e["entry_type"] == "status_snapshot"]
        assert len(hot_snaps) == 3

    def test_compact_demotes_old_snapshots(self, tmp_path):
        """Older status_snapshots go to cold tier."""
        store = MemoryStore(tmp_path / "mem.db")
        for i in range(5):
            store.add(i + 1, "status_snapshot", f"Generation: {i+1}\nScore: 0.8{i}0")

        store.compact(current_generation=50)
        cold = store.get_by_tier("cold")
        cold_snaps = [e for e in cold if e["entry_type"] == "status_snapshot"]
        assert len(cold_snaps) == 2


class TestStaleDetection:
    """Stale entry detection and downranking."""

    def test_mark_stale_old_unhit_entries(self, tmp_path):
        """Entries > 100 gens old with no hits get marked stale."""
        store = MemoryStore(tmp_path / "mem.db")
        # Insert warm entry that's old enough.
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'observation', 'old untouched entry', '{}', 0.3, 'warm', 'old untouched entry', 0, 0, 'active')",
        )
        con.commit()
        con.close()

        stale_count = store._mark_stale(current_generation=200)
        assert stale_count >= 1

        with store._connect() as con:
            row = con.execute("SELECT status FROM memory WHERE content = 'old untouched entry'").fetchone()
            assert row["status"] == "stale"

    def test_recently_hit_not_stale(self, tmp_path):
        """Entries hit within 100 generations should stay active."""
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'observation', 'recently hit entry', '{}', 0.3, 'warm', 'recently hit entry', 5, 150, 'active')",
        )
        con.commit()
        con.close()

        stale_count = store._mark_stale(current_generation=200)
        assert stale_count == 0

        with store._connect() as con:
            row = con.execute("SELECT status FROM memory WHERE content = 'recently hit entry'").fetchone()
            assert row["status"] == "active"

    def test_stale_entries_downranked_in_search(self, tmp_path):
        """Stale entries should have lower effective importance in get_relevant."""
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        # Active entry with lower base importance.
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'task', 'python analysis active', '{}', 0.5, 'hot', 'python analysis active', 0, 0, 'active')",
        )
        # Stale entry with higher base importance.
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'task', 'python research stale', '{}', 0.8, 'warm', 'python research stale', 0, 0, 'stale')",
        )
        con.commit()
        con.close()

        results = store.get_relevant(["python"])
        assert len(results) == 2
        # Active entry (0.5) should rank above stale entry (0.8 * 0.5 = 0.4).
        assert results[0]["content"] == "python analysis active"

    def test_stale_in_compact_result(self, tmp_path):
        """compact() should return stale count in result dict."""
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        # Use cold tier with high enough importance to avoid being archived,
        # but old enough to be stale.
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'semantic_rule', 'will be stale', '{}', 0.7, 'cold', 'will be stale', 0, 0, 'active')",
        )
        con.commit()
        con.close()

        result = store.compact(current_generation=200)
        assert "stale" in result
        assert result["stale"] >= 1

    def test_stale_downrank_in_hybrid_search(self, tmp_path):
        """Stale entries should have lower score in hybrid_search."""
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'task', 'python project analysis', '{}', 0.7, 'hot', 'python project analysis', 0, 0, 'stale')",
        )
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (10, 'task', 'python project review', '{}', 0.7, 'hot', 'python project review', 0, 0, 'active')",
        )
        con.commit()
        con.close()

        results = store.hybrid_search(["python", "project"])
        assert len(results) == 2
        # Active entry should have higher score.
        active = [r for r in results if r["status"] == "active"][0]
        stale = [r for r in results if r["status"] == "stale"][0]
        assert active["search_score"] > stale["search_score"]

    def test_hot_entries_not_marked_stale(self, tmp_path):
        """Hot tier entries should never be marked stale."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(10, "observation", "hot tier entry that is old", importance=0.3)
        stale_count = store._mark_stale(current_generation=200)
        assert stale_count == 0


class TestConflictDetection:
    """Conflict action in curation."""

    def test_apply_curation_conflict_action(self, tmp_path):
        """Conflict action sets status and metadata."""
        store = MemoryStore(tmp_path / "mem.db")
        id1 = store.add(1, "semantic_rule", "User prefers dark mode", importance=0.8)
        id2 = store.add(2, "semantic_rule", "User prefers light mode", importance=0.8)

        decisions = [
            {"id": id2, "action": "conflict", "conflict_with": id1},
        ]
        store._apply_curation(decisions, current_generation=10)

        # Check id2 is marked as conflict.
        with store._connect() as con:
            row = con.execute("SELECT * FROM memory WHERE id = ?", (id2,)).fetchone()
            assert row["status"] == "conflict"
            assert row["tier"] == "cold"
            meta = json.loads(row["metadata"])
            assert meta["conflict"] is True
            assert meta["conflict_with"] == id1

    def test_conflict_entries_preserved(self, tmp_path):
        """Conflict entries are not deleted, just demoted to cold."""
        store = MemoryStore(tmp_path / "mem.db")
        id1 = store.add(1, "semantic_rule", "Rule A about config settings", importance=0.8)

        decisions = [{"id": id1, "action": "conflict"}]
        store._apply_curation(decisions, current_generation=10)

        # Entry should still exist, in cold tier.
        cold = store.get_by_tier("cold")
        assert any(e["id"] == id1 for e in cold)

    def test_conflict_with_reference(self, tmp_path):
        """Conflict metadata contains conflict_with id."""
        store = MemoryStore(tmp_path / "mem.db")
        id1 = store.add(1, "semantic_rule", "Always use JSON output", importance=0.8)
        id2 = store.add(2, "semantic_rule", "Use plain text output format", importance=0.8)

        decisions = [{"id": id2, "action": "conflict", "conflict_with": id1}]
        store._apply_curation(decisions, current_generation=10)

        with store._connect() as con:
            row = con.execute("SELECT metadata FROM memory WHERE id = ?", (id2,)).fetchone()
            meta = json.loads(row["metadata"])
            assert meta["conflict_with"] == id1

    def test_conflict_without_reference(self, tmp_path):
        """Conflict action without conflict_with still sets status."""
        store = MemoryStore(tmp_path / "mem.db")
        id1 = store.add(1, "semantic_rule", "Some conflicting rule about patterns", importance=0.8)

        decisions = [{"id": id1, "action": "conflict"}]
        store._apply_curation(decisions, current_generation=10)

        with store._connect() as con:
            row = con.execute("SELECT * FROM memory WHERE id = ?", (id1,)).fetchone()
            assert row["status"] == "conflict"
            meta = json.loads(row["metadata"])
            assert meta["conflict"] is True
            assert "conflict_with" not in meta
