"""Tests for gene task-hit feedback loop.

Covers: task_hit columns, record_task_hits, apply_task_boost,
reduced apply_boost weight, differentiated decay, get_relevant
task_hit weight, skill_lineage CRUD, and end-to-end attribution.
"""

import json
import pathlib

import pytest

from ring0.gene_pool import GenePool
from ring0.skill_store import SkillStore


# --- Sample Ring 2 source code for testing ---

SAMPLE_SOURCE = '''\
import os, pathlib, time, threading

class StreamAnalyzer:
    """Real-time anomaly detection in data streams."""

    def __init__(self, window_size=100):
        self.window_size = window_size

    def analyze(self, value):
        """Detect anomalies using z-score method."""
        pass

def compute_fibonacci(n):
    """Calculate fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''

SAMPLE_SOURCE_B = '''\
import os, pathlib, time, threading

class WeatherForecast:
    """Atmospheric pressure prediction engine."""

    def predict(self, data):
        """Generate forecast."""
        pass

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''


class TestMigrationAddsColumns:
    def test_task_hit_columns_exist(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE)
        top = gp.get_top(1)
        assert "task_hit_count" in top[0]
        assert "last_task_hit_gen" in top[0]
        assert top[0]["task_hit_count"] == 0
        assert top[0]["last_task_hit_gen"] == 0


class TestRecordTaskHits:
    def test_increments_correctly(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        gp.record_task_hits([gene_id], generation=5)
        top = gp.get_top(1)
        assert top[0]["task_hit_count"] == 1
        assert top[0]["last_task_hit_gen"] == 5

        gp.record_task_hits([gene_id], generation=8)
        top = gp.get_top(1)
        assert top[0]["task_hit_count"] == 2
        assert top[0]["last_task_hit_gen"] == 8

    def test_empty_list_no_error(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.record_task_hits([], generation=1)  # should not raise


class TestApplyTaskBoost:
    def test_boosts_by_005_per_hit(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        gp.record_task_hits([gene_id], generation=5)
        gp.record_task_hits([gene_id], generation=6)
        boosted = gp.apply_task_boost()

        assert boosted == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.80) < 0.001  # 0.70 + 2*0.05
        assert top[0]["task_hit_count"] == 0  # reset

    def test_cap_at_1_0(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.95, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        # 3 hits → +0.15, but capped at 1.0
        for i in range(3):
            gp.record_task_hits([gene_id], generation=i)
        boosted = gp.apply_task_boost()

        assert boosted == 1
        top = gp.get_top(1)
        assert top[0]["score"] == 1.0

    def test_no_boost_when_zero_hits(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE)
        boosted = gp.apply_task_boost()
        assert boosted == 0


class TestApplyBoostReduced:
    def test_reduced_weight(self, tmp_path):
        """apply_boost now gives +0.01 per 3 hits (down from +0.03)."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.70, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        for i in range(3):
            gp.record_hits([gene_id], generation=i)
        boosted = gp.apply_boost()

        assert boosted == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.71) < 0.001  # +0.01


class TestDecayZeroTaskHits:
    def test_accelerated_decay(self, tmp_path):
        """Genes with task_hit_count=0 decay at -0.03."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE)
        # No task hits, stale gene
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.77) < 0.001


class TestDecayWithTaskHits:
    def test_gentle_decay(self, tmp_path):
        """Genes with task_hit_count > 0 decay at -0.01."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        # Give it a task hit (but make both last_hit_gen and last_task_hit_gen stale)
        gp.record_task_hits([gene_id], generation=1)
        # Apply task boost to process the hit but keep task_hit_count > 0
        # Actually, apply_task_boost resets count to 0. Let's add hits
        # after the stale window check for correct setup.
        # Instead: we need task_hit_count > 0, but last_task_hit_gen stale.
        # record_task_hits sets last_task_hit_gen=1, which is < 20-10=10, so stale.
        # But apply_task_boost resets task_hit_count. We need to NOT call it.
        # The task_hit_count stays at 1 after record_task_hits.

        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 1
        top = gp.get_top(1)
        assert abs(top[0]["score"] - 0.79) < 0.001  # -0.01

    def test_no_decay_if_task_hit_recent(self, tmp_path):
        """Genes with recent last_task_hit_gen are not decayed."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE)
        gene_id = gp.get_top(1)[0]["id"]

        # last_hit_gen=0 (stale), but last_task_hit_gen=15 (recent)
        gp.record_task_hits([gene_id], generation=15)
        decayed = gp.apply_decay(current_generation=20)
        assert decayed == 0
        assert gp.get_top(1)[0]["score"] == 0.80


class TestGetRelevantTaskHitWeight:
    def test_task_hit_high_ranks_higher(self, tmp_path):
        """A gene with high task_hits ranks above a gene with higher base score."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        # Gene A: lower score, but many task hits
        gp.add(1, 0.60, SAMPLE_SOURCE)
        gene_a_id = gp.get_top(1)[0]["id"]
        for _ in range(5):
            gp.record_task_hits([gene_a_id], generation=1)

        # Gene B: higher score, no task hits
        gp.add(2, 0.90, SAMPLE_SOURCE_B)

        # Query with context that matches both
        relevant = gp.get_relevant("stream anomaly weather forecast prediction", 2)
        assert len(relevant) == 2
        # Gene A should rank first due to task_hit boost (5 * 0.5 = 2.5 extra)
        assert relevant[0]["id"] == gene_a_id


class TestSkillLineageCrud:
    def test_record_and_get_lineage(self, tmp_path):
        db = tmp_path / "test.db"
        ss = SkillStore(db)

        ss.record_lineage("my_skill", [10, 20, 30], generation=5)
        lineage = ss.get_lineage("my_skill")
        assert len(lineage) == 3
        gene_ids = {e["gene_id"] for e in lineage}
        assert gene_ids == {10, 20, 30}
        assert all(e["generation"] == 5 for e in lineage)

    def test_dedup_on_skill_gene_pair(self, tmp_path):
        db = tmp_path / "test.db"
        ss = SkillStore(db)

        ss.record_lineage("my_skill", [10], generation=5)
        ss.record_lineage("my_skill", [10], generation=6)  # duplicate
        lineage = ss.get_lineage("my_skill")
        assert len(lineage) == 1

    def test_get_gene_skills(self, tmp_path):
        db = tmp_path / "test.db"
        ss = SkillStore(db)

        ss.record_lineage("skill_a", [10], generation=5)
        ss.record_lineage("skill_b", [10], generation=6)
        ss.record_lineage("skill_c", [20], generation=7)

        skills = ss.get_gene_skills(10)
        assert set(skills) == {"skill_a", "skill_b"}

        skills_20 = ss.get_gene_skills(20)
        assert skills_20 == ["skill_c"]

    def test_empty_lineage(self, tmp_path):
        db = tmp_path / "test.db"
        ss = SkillStore(db)

        assert ss.get_lineage("nonexistent") == []
        assert ss.get_gene_skills(999) == []

    def test_record_empty_gene_ids(self, tmp_path):
        db = tmp_path / "test.db"
        ss = SkillStore(db)
        ss.record_lineage("my_skill", [], generation=5)  # should not raise
        assert ss.get_lineage("my_skill") == []


class TestTaskHitAttributionE2E:
    """End-to-end: skill usage → lineage lookup → gene task_hit."""

    def test_full_attribution_flow(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        ss = SkillStore(db)

        # 1. Add genes
        gp.add(1, 0.80, SAMPLE_SOURCE)
        gp.add(2, 0.70, SAMPLE_SOURCE_B)
        genes = gp.get_top(0)
        gene_a_id = genes[0]["id"]  # StreamAnalyzer gene
        gene_b_id = genes[1]["id"]  # WeatherForecast gene

        # 2. Record lineage: skill "stream_detect" comes from gene_a
        ss.record_lineage("stream_detect", [gene_a_id], generation=3)

        # 3. Simulate task metadata with skills_used
        task_meta = {"skills_used": ["stream_detect"]}

        # 4. Look up lineage and attribute
        attributed_gene_ids = set()
        for skill_name in task_meta["skills_used"]:
            for entry in ss.get_lineage(skill_name):
                attributed_gene_ids.add(entry["gene_id"])

        assert attributed_gene_ids == {gene_a_id}

        # 5. Record task hits
        gp.record_task_hits(list(attributed_gene_ids), generation=5)

        # 6. Verify
        genes = gp.get_top(0)
        gene_a = next(g for g in genes if g["id"] == gene_a_id)
        gene_b = next(g for g in genes if g["id"] == gene_b_id)
        assert gene_a["task_hit_count"] == 1
        assert gene_a["last_task_hit_gen"] == 5
        assert gene_b["task_hit_count"] == 0

        # 7. Apply task boost — gene_a gets +0.05
        boosted = gp.apply_task_boost()
        assert boosted == 1
        genes = gp.get_top(0)
        gene_a = next(g for g in genes if g["id"] == gene_a_id)
        assert abs(gene_a["score"] - 0.85) < 0.001


class TestAttributionUsesSkillsMatched:
    """Sentinel attribution should also read skills_matched from metadata."""

    def test_attribution_uses_skills_matched(self, tmp_path):
        """skills_matched entries should contribute to attributed_gene_ids."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        ss = SkillStore(db)

        # Add genes
        gp.add(1, 0.80, SAMPLE_SOURCE)
        gp.add(2, 0.70, SAMPLE_SOURCE_B)
        genes = gp.get_top(0)
        gene_a_id = genes[0]["id"]
        gene_b_id = genes[1]["id"]

        # Record lineage
        ss.record_lineage("stream_detect", [gene_a_id], generation=3)
        ss.record_lineage("weather_predict", [gene_b_id], generation=3)

        # Simulate task metadata with skills_matched (not skills_used)
        task_meta = {"skills_used": [], "skills_matched": ["stream_detect", "weather_predict"]}

        # Attribution logic (mirrors sentinel.py)
        attributed_gene_ids: set[int] = set()
        for skill_name in task_meta.get("skills_used", []):
            for entry in ss.get_lineage(skill_name):
                attributed_gene_ids.add(entry["gene_id"])
        for skill_name in task_meta.get("skills_matched", []):
            for entry in ss.get_lineage(skill_name):
                attributed_gene_ids.add(entry["gene_id"])

        assert attributed_gene_ids == {gene_a_id, gene_b_id}

        # Record and verify
        gp.record_task_hits(list(attributed_gene_ids), generation=5)
        genes = gp.get_top(0)
        gene_a = next(g for g in genes if g["id"] == gene_a_id)
        gene_b = next(g for g in genes if g["id"] == gene_b_id)
        assert gene_a["task_hit_count"] == 1
        assert gene_b["task_hit_count"] == 1


class TestLineageViaSourceHashLookup:
    """Simulates the sentinel flow: add gene → get_id_by_hash → record_lineage → verify."""

    def test_lineage_recorded_via_source_hash_lookup(self, tmp_path):
        import hashlib

        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        ss = SkillStore(db)

        # 1. Sentinel adds gene to pool (happens in gene_pool.add)
        gp.add(1, 0.80, SAMPLE_SOURCE)

        # 2. Compute source_hash (same as sentinel does)
        source_hash = hashlib.sha256(SAMPLE_SOURCE.encode()).hexdigest()

        # 3. Look up gene_id by hash (the new path)
        gene_id = gp.get_id_by_hash(source_hash)
        assert gene_id is not None

        # 4. Record lineage with the looked-up gene_id
        ss.record_lineage("my_skill", [gene_id], generation=1)

        # 5. Verify lineage exists
        lineage = ss.get_lineage("my_skill")
        assert len(lineage) == 1
        assert lineage[0]["gene_id"] == gene_id

        # 6. Verify attribution works end-to-end
        gp.record_task_hits([gene_id], generation=5)
        top = gp.get_top(1)
        assert top[0]["task_hit_count"] == 1
