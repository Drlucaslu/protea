"""Tests for memory temperature model and hit tracking."""

from __future__ import annotations

import math
import sqlite3

from ring0.memory import MemoryStore, _compute_temperature


class TestComputeTemperature:
    """Unit tests for the temperature formula."""

    def test_new_entry_high_temperature(self):
        """Entry at current generation should have T ~ 0.7."""
        t = _compute_temperature(importance=0.5, generation=100, current_generation=100)
        # T = 0.4*0.5 + 0.3*1.0 + 0.3*0.0 = 0.50
        # (no hits, so ref_factor = 0)
        assert 0.4 <= t <= 0.6

    def test_old_entry_low_temperature(self):
        """Entry 200 generations old with no hits should have low T."""
        t = _compute_temperature(importance=0.3, generation=0, current_generation=200)
        assert t < 0.2

    def test_high_importance_stays_warm(self):
        """High importance (0.9) keeps temperature above 0.4 for longer."""
        t = _compute_temperature(importance=0.9, generation=50, current_generation=100)
        assert t > 0.4

    def test_frequent_hits_boost_temperature(self):
        """hit_count=10 should significantly boost temperature."""
        t_no_hits = _compute_temperature(
            importance=0.3, generation=50, current_generation=100,
        )
        t_with_hits = _compute_temperature(
            importance=0.3, generation=50, current_generation=100,
            hit_count=10, last_hit_gen=95,
        )
        assert t_with_hits > t_no_hits
        assert t_with_hits - t_no_hits > 0.1

    def test_recent_hit_vs_old_hit(self):
        """Same hit_count, but recent hit should score higher."""
        t_recent = _compute_temperature(
            importance=0.3, generation=50, current_generation=100,
            hit_count=5, last_hit_gen=98,
        )
        t_old = _compute_temperature(
            importance=0.3, generation=50, current_generation=100,
            hit_count=5, last_hit_gen=60,
        )
        assert t_recent > t_old

    def test_zero_age_max_age_factor(self):
        """Zero age gives max age_factor (1.0)."""
        t = _compute_temperature(importance=0.0, generation=100, current_generation=100)
        # T = 0.4*0.0 + 0.3*1.0 + 0.3*0.0 = 0.3
        assert abs(t - 0.3) < 0.01

    def test_temperature_range(self):
        """Temperature should stay within [0, 1]."""
        # Max case
        t_max = _compute_temperature(importance=1.0, generation=100, current_generation=100,
                                     hit_count=100, last_hit_gen=100)
        assert 0.0 <= t_max <= 1.0
        # Min case
        t_min = _compute_temperature(importance=0.0, generation=0, current_generation=1000)
        assert 0.0 <= t_min <= 1.0

    def test_half_life_approximately_46_gens(self):
        """Age factor should be ~0.5 at 46 generations."""
        age_factor = math.exp(-0.015 * 46)
        assert abs(age_factor - 0.5) < 0.02


class TestHitTracking:
    """Integration tests for hit count tracking in search methods."""

    def test_get_relevant_increments_hit_count(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis", importance=0.7)
        # Search with current_generation > 0 to trigger hit tracking.
        results = store.get_relevant(["python"], current_generation=10)
        assert len(results) == 1
        # Check hit_count was incremented.
        entry = store.get_recent(1)[0]
        assert entry["hit_count"] == 1
        assert entry["last_hit_gen"] == 10

    def test_hybrid_search_increments_hit_count(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis", importance=0.7)
        results = store.hybrid_search(["python", "code"], current_generation=20)
        assert len(results) >= 1
        entry = store.get_recent(1)[0]
        assert entry["hit_count"] == 1
        assert entry["last_hit_gen"] == 20

    def test_no_tracking_when_generation_zero(self, tmp_path):
        """Default current_generation=0 should not record hits."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis", importance=0.7)
        store.get_relevant(["python"])  # No current_generation
        entry = store.get_recent(1)[0]
        assert entry["hit_count"] == 0

    def test_multiple_hits_accumulate(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis", importance=0.7)
        store.get_relevant(["python"], current_generation=10)
        store.get_relevant(["python"], current_generation=15)
        store.get_relevant(["python"], current_generation=20)
        entry = store.get_recent(1)[0]
        assert entry["hit_count"] == 3
        assert entry["last_hit_gen"] == 20

    def test_hit_count_survives_compaction(self, tmp_path):
        """Hit data should be preserved when entries move between tiers."""
        store = MemoryStore(tmp_path / "mem.db")
        rid = store.add(1, "observation", "test pattern data entry", importance=0.3)
        # Record some hits.
        store._record_hits([rid], current_generation=5)
        store._record_hits([rid], current_generation=10)
        # Compact to move entry to warm tier.
        store.compact(current_generation=200)
        # Verify hit_count preserved.
        with store._connect() as con:
            row = con.execute("SELECT hit_count, last_hit_gen FROM memory WHERE id = ?", (rid,)).fetchone()
            assert row["hit_count"] == 2
            assert row["last_hit_gen"] == 10


class TestTemperatureCompact:
    """Compaction with temperature-based thresholds."""

    def test_frequently_hit_entry_stays_hot(self, tmp_path):
        """High hit_count prevents demotion even for old entries."""
        store = MemoryStore(tmp_path / "mem.db")
        rid = store.add(1, "observation", "frequently accessed pattern", importance=0.3)
        # Simulate frequent access.
        for gen in range(5, 50, 5):
            store._record_hits([rid], current_generation=gen)
        result = store.compact(current_generation=50)
        # Entry should remain hot because hit_count boosts temperature.
        hot = store.get_by_tier("hot")
        assert any(e["id"] == rid for e in hot)

    def test_unhit_old_entry_demoted(self, tmp_path):
        """Old entry with no hits gets demoted to warm."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "never accessed old data entry", importance=0.3)
        result = store.compact(current_generation=200)
        assert result["hot_to_warm"] > 0

    def test_warm_to_cold_temperature_threshold(self, tmp_path):
        """Warm entries with T < 0.3 become candidates for cold transition."""
        store = MemoryStore(tmp_path / "mem.db")
        # Manually insert a warm entry that's old and low importance.
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (1, 'observation', 'old warm entry', '{}', 0.2, 'warm', 'old warm entry', 0, 0, 'active')",
        )
        con.commit()
        con.close()

        candidates = store._get_warm_candidates(current_generation=200)
        assert len(candidates) >= 1
        assert candidates[0]["temperature"] < 0.3

    def test_cleanup_cold_temperature(self, tmp_path):
        """Cold entries with T < 0.15 get archived."""
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, hit_count, last_hit_gen, status) "
            "VALUES (1, 'observation', 'ancient cold entry', '{}', 0.1, 'cold', 'ancient cold entry', 0, 0, 'active')",
        )
        con.commit()
        con.close()

        result = store.compact(current_generation=500)
        assert result["deleted"] >= 1
        archived = store.get_by_tier("archive")
        assert any(e["content"] == "ancient cold entry" for e in archived)

    def test_get_temperature_method(self, tmp_path):
        """get_temperature() returns the correct value for a specific entry."""
        store = MemoryStore(tmp_path / "mem.db")
        rid = store.add(50, "observation", "test observation entry", importance=0.5)
        t = store.get_temperature(rid, current_generation=50)
        expected = _compute_temperature(0.5, 50, 50)
        assert abs(t - expected) < 0.001

    def test_get_temperature_missing_entry(self, tmp_path):
        """get_temperature() returns 0.0 for missing entry."""
        store = MemoryStore(tmp_path / "mem.db")
        assert store.get_temperature(999, current_generation=50) == 0.0
