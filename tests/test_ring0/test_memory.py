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
        assert _compute_importance("crash_log", "short") == 0.4

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
        assert _compute_importance("observation", long_content) == 0.35

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
        assert entries[0]["importance"] == 0.3  # observation base, no boost


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

    def test_cold_cleanup_archives(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        # Manually insert a cold entry with low importance
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'observation', 'ancient', '{}', 0.1, 'cold', 'ancient')",
        )
        con.commit()
        con.close()

        result = store.compact(current_generation=400)
        assert result["deleted"] == 1
        # Entry is archived, not deleted.
        archived = store.get_by_tier("archive")
        assert len(archived) == 1
        assert archived[0]["content"] == "ancient"
        # Importance decayed by 0.3x.
        assert abs(archived[0]["importance"] - 0.03) < 1e-6

    def test_compact_returns_correct_keys(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        result = store.compact(current_generation=0)
        assert "hot_to_warm" in result
        assert "warm_to_cold" in result
        assert "deleted" in result
        assert "llm_curated" in result
        assert "deduped" in result

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


class TestArchiveTier:
    """Archive tier: memories are preserved, not deleted."""

    def test_compact_hot_to_warm_archives_originals(self, tmp_path):
        """Merged originals go to archive tier with archived_ids in summary."""
        store = MemoryStore(tmp_path / "mem.db")
        # Add 5 old low-importance entries of same type → 3 kept, 2 merged+archived
        ids = []
        for i in range(5):
            rid = store.add(1, "observation", f"old obs {i}", importance=0.5)
            ids.append(rid)

        result = store.compact(current_generation=50)
        assert result["hot_to_warm"] == 5

        # The 2 merged originals should be in archive tier.
        archived = store.get_by_tier("archive")
        assert len(archived) == 2
        archived_ids = {e["id"] for e in archived}
        # They should be the last 2 entries (ids[3], ids[4]).
        assert ids[3] in archived_ids
        assert ids[4] in archived_ids

        # The compacted summary should contain archived_ids metadata.
        warm = store.get_by_tier("warm")
        summaries = [e for e in warm if "Compacted" in e["content"]]
        assert len(summaries) == 1
        assert set(summaries[0]["metadata"]["archived_ids"]) == {ids[3], ids[4]}

    def test_get_recent_excludes_archive(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "active entry")
        # Manually insert an archived entry.
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'observation', 'archived entry', '{}', 0.1, 'archive', 'archived')",
        )
        con.commit()
        con.close()

        recent = store.get_recent(10)
        assert len(recent) == 1
        assert recent[0]["content"] == "active entry"

    def test_get_by_type_excludes_archive(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "active task")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'task', 'archived task', '{}', 0.1, 'archive', 'archived')",
        )
        con.commit()
        con.close()

        tasks = store.get_by_type("task")
        assert len(tasks) == 1
        assert tasks[0]["content"] == "active task"

    def test_get_relevant_excludes_archive(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "python code analysis")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'task', 'python old stuff', '{}', 0.1, 'archive', 'python old stuff')",
        )
        con.commit()
        con.close()

        results = store.get_relevant(["python"])
        assert len(results) == 1
        assert results[0]["content"] == "python code analysis"

    def test_recall_keyword_match(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        # Add an archived entry with keywords.
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, embedding) "
            "VALUES (42, 'task', 'websocket realtime push research', '{}', 0.05, 'archive', 'websocket realtime push research', '')",
        )
        con.commit()
        con.close()

        results = store.recall(["websocket", "realtime"])
        assert len(results) == 1
        assert results[0]["recalled"] is True
        assert results[0]["generation"] == 42
        assert "websocket" in results[0]["content"]

    def test_recall_no_match(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        # Add an archived entry.
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords, embedding) "
            "VALUES (1, 'task', 'stock market data', '{}', 0.05, 'archive', 'stock market data', '')",
        )
        con.commit()
        con.close()

        results = store.recall(["quantum", "computing"])
        assert results == []

    def test_recall_ignores_non_archive(self, tmp_path):
        """recall() should only search archive tier."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "websocket hot entry")
        results = store.recall(["websocket"])
        assert results == []

    def test_recall_empty_keywords(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        assert store.recall([]) == []

    def test_get_stats_includes_archive(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "hot entry")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'observation', 'old entry', '{}', 0.05, 'archive', '')",
        )
        con.commit()
        con.close()

        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["by_tier"].get("archive") == 1
        assert stats["by_tier"].get("hot") == 1


class TestIsRecentDuplicate:
    """_is_recent_duplicate() rejects duplicate content at write time."""

    def test_exact_duplicate_rejected(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "server crashed due to OOM")
        assert store._is_recent_duplicate("reflection", "server crashed due to OOM")

    def test_case_insensitive_match(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "Server Crashed")
        assert store._is_recent_duplicate("reflection", "server crashed")

    def test_fuzzy_match_for_non_task(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(
            1, "reflection",
            "tasks.jsonl not found in the working directory, this caused generation failure",
        )
        # Similar but not identical (>50 chars to trigger fuzzy)
        assert store._is_recent_duplicate(
            "reflection",
            "tasks.jsonl not found in the current working directory, this caused generation failure",
        )

    def test_no_fuzzy_match_for_task(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "帮我分析一下 memory 的内容", importance=0.7)
        # Similar phrasing but tasks should only exact-match
        assert not store._is_recent_duplicate(
            "task", "帮我分析一下 memory 里面的内容"
        )

    def test_exact_task_duplicate_rejected(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "task", "commit and push", importance=0.7)
        assert store._is_recent_duplicate("task", "commit and push")

    def test_short_content_no_fuzzy(self, tmp_path):
        """Short non-task content uses exact match only (no fuzzy)."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "entry-0")
        # Very similar but not exact — should NOT be flagged as duplicate
        assert not store._is_recent_duplicate("observation", "entry-1")

    def test_different_type_not_duplicate(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "OOM crash on gen 100")
        # Same content but different type → not a duplicate
        assert not store._is_recent_duplicate("crash_log", "OOM crash on gen 100")

    def test_no_entries_not_duplicate(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        assert not store._is_recent_duplicate("reflection", "anything")

    def test_add_rejects_duplicate(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        rid1 = store.add(1, "reflection", "ring2 timed out after 600s")
        rid2 = store.add(2, "reflection", "ring2 timed out after 600s")
        assert rid1 > 0
        assert rid2 == -1  # Rejected

    def test_add_with_embedding_rejects_duplicate(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        rid1 = store.add_with_embedding(1, "reflection", "ring2 timed out", embedding=[0.1, 0.2])
        rid2 = store.add_with_embedding(2, "reflection", "ring2 timed out", embedding=[0.3, 0.4])
        assert rid1 > 0
        assert rid2 == -1


class TestDeduplicate:
    """deduplicate() archives exact-duplicate entries in the database."""

    def test_removes_exact_duplicates(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        # Insert duplicates directly via SQL (bypassing add() dedup)
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        for i in range(3):
            con.execute(
                "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
                "VALUES (?, 'task', 'duplicate content', '{}', 0.5, 'hot', '')",
                (i,),
            )
        con.commit()
        con.close()

        archived = store.deduplicate()
        assert archived == 2  # 3 copies → keep 1, archive 2
        hot = store.get_by_tier("hot")
        assert sum(1 for e in hot if e["content"] == "duplicate content") == 1

    def test_no_duplicates_returns_zero(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "unique entry one")
        store.add(2, "reflection", "unique entry two is different")
        assert store.deduplicate() == 0

    def test_keeps_newest(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        for gen in [10, 20, 30]:
            con.execute(
                "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
                "VALUES (?, 'reflection', 'same reflection', '{}', 0.5, 'hot', '')",
                (gen,),
            )
        con.commit()
        con.close()

        store.deduplicate()
        hot = [e for e in store.get_by_tier("hot") if e["content"] == "same reflection"]
        assert len(hot) == 1
        assert hot[0]["generation"] == 30  # Newest kept

    def test_compact_includes_dedup(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        for i in range(2):
            con.execute(
                "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
                "VALUES (1, 'observation', 'dup obs', '{}', 0.5, 'hot', '')",
            )
        con.commit()
        con.close()

        result = store.compact(current_generation=50)
        assert result["deduped"] == 1
        assert "deduped" in result


class TestCompactThresholds:
    """Verify tightened hot→warm compaction thresholds."""

    def test_importance_059_gets_demoted(self, tmp_path):
        """Entries with importance < 0.6 and old enough should be demoted."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "observation", "low importance entry", importance=0.59)
        result = store.compact(current_generation=10)
        assert result["hot_to_warm"] > 0

    def test_importance_060_stays_hot(self, tmp_path):
        """Entries with importance >= 0.6 should remain hot."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "reflection", "moderate importance", importance=0.6)
        result = store.compact(current_generation=10)
        hot = store.get_by_tier("hot")
        assert any(e["content"] == "moderate importance" for e in hot)

    def test_generation_gap_5(self, tmp_path):
        """Entries within 5 generations should not be compacted."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(6, "observation", "recent low importance", importance=0.3)
        result = store.compact(current_generation=10)
        hot = store.get_by_tier("hot")
        assert any(e["content"] == "recent low importance" for e in hot)

    def test_generation_gap_6_gets_compacted(self, tmp_path):
        """Entries older than 5 generations with low importance get compacted."""
        store = MemoryStore(tmp_path / "mem.db")
        store.add(4, "observation", "old low importance entry", importance=0.3)
        result = store.compact(current_generation=10)
        assert result["hot_to_warm"] > 0


class TestSemanticRule:
    """Semantic rule importance, get_semantic_rules, and _apply_curation extract_rule."""

    def test_semantic_rule_importance(self):
        assert _compute_importance("semantic_rule", "Always use heartbeat") == 0.8

    def test_get_semantic_rules_returns_correct(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "semantic_rule", "Rule A", importance=0.8)
        store.add(2, "semantic_rule", "Rule B", importance=0.9)
        store.add(3, "observation", "Not a rule")

        rules = store.get_semantic_rules()
        assert len(rules) == 2
        # Ordered by importance DESC
        assert rules[0]["content"] == "Rule B"
        assert rules[1]["content"] == "Rule A"

    def test_get_semantic_rules_excludes_archive(self, tmp_path):
        import sqlite3
        store = MemoryStore(tmp_path / "mem.db")
        store.add(1, "semantic_rule", "Active rule", importance=0.8)
        con = sqlite3.connect(str(tmp_path / "mem.db"))
        con.execute(
            "INSERT INTO memory (generation, entry_type, content, metadata, importance, tier, keywords) "
            "VALUES (1, 'semantic_rule', 'Archived rule', '{}', 0.8, 'archive', '')",
        )
        con.commit()
        con.close()

        rules = store.get_semantic_rules()
        assert len(rules) == 1
        assert rules[0]["content"] == "Active rule"

    def test_get_semantic_rules_respects_limit(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        for i in range(5):
            store.add(i, "semantic_rule", f"Rule {i}", importance=0.8)
        rules = store.get_semantic_rules(limit=3)
        assert len(rules) == 3

    def test_apply_curation_extract_rule(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.db")
        id1 = store.add(10, "reflection", "CA patterns survive")
        id2 = store.add(12, "reflection", "CA with heartbeat is robust")
        id3 = store.add(15, "observation", "unrelated observation")

        decisions = [
            {"id": [id1, id2], "action": "extract_rule", "rule": "CA patterns are the most robust approach"},
        ]
        store._apply_curation(decisions, current_generation=20)

        # Source entries demoted to cold.
        src1 = [e for e in store.get_by_tier("cold") if e["id"] == id1]
        src2 = [e for e in store.get_by_tier("cold") if e["id"] == id2]
        assert len(src1) == 1
        assert len(src2) == 1

        # New semantic_rule entry created.
        rules = store.get_semantic_rules()
        assert len(rules) == 1
        assert rules[0]["content"] == "CA patterns are the most robust approach"
        assert rules[0]["entry_type"] == "semantic_rule"
        assert rules[0]["importance"] == 0.8
        assert rules[0]["tier"] == "hot"
        assert rules[0]["metadata"]["source_ids"] == [id1, id2]
        # Generation should be max of sources (12), not current_generation
        assert rules[0]["generation"] == 12

        # Unrelated entry unchanged.
        hot = store.get_by_tier("hot")
        assert any(e["id"] == id3 for e in hot)
