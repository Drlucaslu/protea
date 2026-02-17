"""Tests for ring0.memory — MemoryStore."""

from __future__ import annotations

from ring0.memory import MemoryStore, _compute_importance, _extract_keywords, _cosine_similarity


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


# -----------------------------------------------------------------------
# New: importance, tier, keywords, embedding, compact, search
# -----------------------------------------------------------------------


class TestImportance:
    """Importance scoring and auto-assignment."""

    def test_directive_highest(self):
        assert _compute_importance("directive", "short") == 0.9

    def test_crash_log(self):
        assert _compute_importance("crash_log", "short") == 0.8

    def test_task_substantive(self):
        """Substantive task with enough length gets high importance."""
        assert _compute_importance("task", "帮我分析一下这个系统的性能问题，找出瓶颈所在") == 0.7

    def test_task_short_followup(self):
        """Very short task messages get low importance."""
        assert _compute_importance("task", "好的") == 0.2

    def test_task_operational_chinese(self):
        """Chinese operational commands get low importance."""
        assert _compute_importance("task", "发给我看一下") <= 0.35

    def test_task_operational_english(self):
        """English operational commands get low importance."""
        assert _compute_importance("task", "commit and push it") <= 0.35

    def test_task_confirmation(self):
        """Simple confirmations get minimal importance."""
        assert _compute_importance("task", "yes") == 0.2
        assert _compute_importance("task", "可以") == 0.2

    def test_task_substantive_long(self):
        """Long substantive requests get boosted importance."""
        long_request = "帮我实现一个用户认证系统，需要支持 OAuth 和 JWT 两种方式，" + "详细说明" * 20
        assert _compute_importance("task", long_request) == 0.75

    def test_task_medium_nonsubstantive(self):
        """Short messages without substantive keywords get medium-low importance."""
        assert _compute_importance("task", "调一下那个参数") <= 0.35

    def test_task_record_memory(self):
        """'记到memory' is a follow-up pattern."""
        assert _compute_importance("task", "把这个记到 memory 里") <= 0.35

    def test_long_content_bonus(self):
        long_content = "x" * 600
        assert _compute_importance("observation", long_content) == 0.55

    def test_unknown_type_defaults(self):
        assert _compute_importance("custom", "short") == 0.5

    def test_add_auto_computes_importance(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "directive", "important")
        entries = store.get_recent(1)
        assert entries[0]["importance"] == 0.9

    def test_add_explicit_importance(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test", importance=0.99)
        entries = store.get_recent(1)
        assert entries[0]["importance"] == 0.99


class TestSessionAwareness:
    """Session-based importance boost for task entries."""

    def test_first_task_gets_session_boost(self, tmp_path):
        """First task ever (no prior tasks) = new session → gets boost."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "帮我分析一下这个系统的性能问题，找出瓶颈所在")
        entries = store.get_recent(1)
        # Base 0.7 + 0.1 session boost = 0.8
        assert entries[0]["importance"] == 0.8

    def test_followup_no_boost(self, tmp_path):
        """Task immediately after another task = same session → no boost."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "帮我实现一个功能，需要支持多种认证方式来保护API端点")
        store.add(1, "task", "帮我设计一个缓存系统，要支持LRU淘汰策略和过期时间设置")
        entries = store.get_recent(1)
        # Second task is within same session, no boost
        assert entries[0]["importance"] == 0.7

    def test_non_task_no_boost(self, tmp_path):
        """Non-task entries never get session boost."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "Gen 1 survived")
        entries = store.get_recent(1)
        assert entries[0]["importance"] == 0.5  # No boost


class TestKeywords:
    """Keyword extraction."""

    def test_extracts_words(self):
        kw = _extract_keywords("Python code debug function")
        assert "python" in kw
        assert "code" in kw
        assert "debug" in kw

    def test_filters_short_words(self):
        kw = _extract_keywords("a to it")
        assert kw == ""

    def test_deduplicates(self):
        kw = _extract_keywords("python python python")
        assert kw.count("python") == 1


class TestTier:
    """Tier assignment and get_by_tier."""

    def test_new_entries_are_hot(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test")
        entries = store.get_recent(1)
        assert entries[0]["tier"] == "hot"

    def test_get_by_tier(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "hot entry")
        hot = store.get_by_tier("hot")
        assert len(hot) == 1
        assert hot[0]["content"] == "hot entry"

    def test_get_by_tier_empty(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "test")
        assert store.get_by_tier("cold") == []


class TestGetRelevant:
    """Keyword-based relevance search."""

    def test_finds_matching_entries(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis")
        store.add(2, "task", "stock market trading")
        results = store.get_relevant(["python"])
        assert len(results) == 1
        assert "python" in results[0]["content"]

    def test_empty_keywords(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code")
        assert store.get_relevant([]) == []

    def test_no_matches(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code")
        assert store.get_relevant(["nonexistent_keyword_xyz"]) == []


class TestGetStats:
    """Memory statistics."""

    def test_empty_stats(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["by_tier"] == {}
        assert stats["by_type"] == {}

    def test_counts_by_tier_and_type(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "task1")
        store.add(2, "task", "task2")
        store.add(3, "observation", "obs1")
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["by_tier"]["hot"] == 3
        assert stats["by_type"]["task"] == 2
        assert stats["by_type"]["observation"] == 1


class TestCosine:
    """Cosine similarity helper."""

    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == 0.0

    def test_opposite_vectors(self):
        assert abs(_cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 1e-9

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0


class TestEmbedding:
    """Embedding storage and vector search."""

    def test_add_with_embedding(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add_with_embedding(1, "task", "test", embedding=[0.1, 0.2, 0.3])
        entries = store.get_recent(1)
        assert entries[0]["content"] == "test"

    def test_search_similar(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add_with_embedding(1, "task", "python coding", embedding=[1.0, 0.0, 0.0])
        store.add_with_embedding(2, "task", "stock market", embedding=[0.0, 1.0, 0.0])
        store.add_with_embedding(3, "task", "python debug", embedding=[0.9, 0.1, 0.0])

        results = store.search_similar([1.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0]["content"] == "python coding"  # Most similar
        assert results[0]["similarity"] == 1.0

    def test_search_similar_min_threshold(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add_with_embedding(1, "task", "a", embedding=[1.0, 0.0])
        store.add_with_embedding(2, "task", "b", embedding=[0.0, 1.0])

        results = store.search_similar([1.0, 0.0], min_similarity=0.9)
        assert len(results) == 1

    def test_add_without_embedding(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add_with_embedding(1, "task", "no embedding")
        results = store.search_similar([1.0, 0.0])
        assert len(results) == 0


class TestHybridSearch:
    """Mixed keyword + vector search."""

    def test_keyword_only(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis")
        store.add(2, "task", "stock market data")

        results = store.hybrid_search(["python", "code"])
        assert len(results) >= 1
        assert "python" in results[0]["content"]

    def test_with_embedding(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add_with_embedding(1, "task", "python coding", embedding=[1.0, 0.0])
        store.add_with_embedding(2, "task", "stock market", embedding=[0.0, 1.0])

        results = store.hybrid_search(["python"], query_embedding=[1.0, 0.0])
        assert len(results) >= 1
        assert results[0]["content"] == "python coding"

    def test_empty_search(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        assert store.hybrid_search([], None) == []


class TestCompact:
    """Tiered compaction."""

    def test_hot_to_warm_basic(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        # Add old entries (generation 1) with low importance
        for i in range(5):
            store.add(1, "observation", f"old entry {i}", importance=0.5)
        # Add recent entries (generation 50)
        store.add(50, "observation", "recent", importance=0.5)

        result = store.compact(current_generation=50)
        assert result["hot_to_warm"] > 0
        # Recent entry should still be hot
        hot = store.get_by_tier("hot")
        assert any(e["content"] == "recent" for e in hot)

    def test_high_importance_not_demoted(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "directive", "important directive", importance=0.9)
        result = store.compact(current_generation=50)
        # Directives have importance 0.9 >= 0.7, should stay hot
        hot = store.get_by_tier("hot")
        assert len(hot) == 1

    def test_cold_cleanup(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        # Manually insert a cold entry with low importance
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'observation', 'ancient', '{}', 0.1, 'cold', '')",
        )
        con.commit()
        con.close()

        result = store.compact(current_generation=400)
        assert result["deleted"] == 1

    def test_compact_returns_correct_keys(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        result = store.compact(current_generation=0)
        assert "hot_to_warm" in result
        assert "warm_to_cold" in result
        assert "deleted" in result
        assert "llm_curated" in result

    def test_compact_with_curator(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        # Add warm entries old enough for cold transition
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'task', 'old warm task', '{}', 0.5, 'warm', 'old warm task')",
        )
        con.commit()
        con.close()

        # Mock curator
        class MockCurator:
            def curate(self, candidates):
                return [{"id": c["id"], "action": "keep"} for c in candidates]

        result = store.compact(current_generation=100, curator=MockCurator())
        assert result["llm_curated"] > 0

    def test_compact_curator_failure_fallback(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'task', 'warm entry', '{}', 0.5, 'warm', '')",
        )
        con.commit()
        con.close()

        class FailingCurator:
            def curate(self, candidates):
                raise RuntimeError("LLM failed")

        result = store.compact(current_generation=100, curator=FailingCurator())
        # Should fall back to rule-based
        assert result["llm_curated"] == 0
        assert result["warm_to_cold"] > 0


class TestMigration:
    """Database migration adds new columns to existing tables."""

    def test_migration_adds_columns(self, tmp_path):
        import sqlite3

        db = tmp_path / "legacy.db"
        # Create a legacy table without new columns.
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE memory ("
            "id INTEGER PRIMARY KEY, generation INTEGER, entry_type TEXT, "
            "content TEXT, metadata TEXT DEFAULT '{}', timestamp TEXT)"
        )
        con.execute(
            "INSERT INTO memory (generation, entry_type, content) VALUES (1, 'task', 'old')"
        )
        con.commit()
        con.close()

        # Opening MemoryStore should migrate.
        store = MemoryStore(db)
        entries = store.get_recent(1)
        assert entries[0]["content"] == "old"
        # New columns should have defaults.
        assert entries[0]["importance"] == 0.5
        assert entries[0]["tier"] == "hot"

    def test_migration_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        store1 = MemoryStore(db)
        store1.add(1, "task", "test")
        # Re-open should not fail.
        store2 = MemoryStore(db)
        assert store2.count() == 1
