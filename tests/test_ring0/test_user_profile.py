"""Tests for ring0.user_profile — UserProfiler."""

from __future__ import annotations

from ring0.user_profile import UserProfiler, _tokenize, _extract_bigrams


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello world python code")
        assert "python" in tokens
        assert "code" in tokens

    def test_removes_stop_words(self):
        tokens = _tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "very" not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("I am a do it")
        assert "am" not in tokens
        assert "do" not in tokens

    def test_lowercases(self):
        tokens = _tokenize("Python JavaScript CLASS")
        assert "python" in tokens
        assert "javascript" in tokens
        assert "class" in tokens


class TestExtractBigrams:
    def test_basic_bigrams(self):
        tokens = ["machine", "learning", "model"]
        bigrams = _extract_bigrams(tokens)
        assert "machine_learning" in bigrams
        assert "learning_model" in bigrams

    def test_empty_tokens(self):
        assert _extract_bigrams([]) == []

    def test_single_token(self):
        assert _extract_bigrams(["hello"]) == []


class TestUpdateFromTask:
    def test_extracts_known_topics(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("Help me debug my python code")
        topics = profiler.get_top_topics()
        topic_names = [t["topic"] for t in topics]
        assert "python" in topic_names
        assert "code" in topic_names
        assert "debug" in topic_names

    def test_categorizes_correctly(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python function class")
        topics = profiler.get_top_topics()
        coding_topics = [t for t in topics if t["category"] == "coding"]
        assert len(coding_topics) >= 2

    def test_repeated_topics_increase_weight(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code")
        profiler.update_from_task("python debug")
        topics = profiler.get_top_topics()
        python_topic = [t for t in topics if t["topic"] == "python"][0]
        assert python_topic["weight"] == 2.0
        assert python_topic["hit_count"] == 2

    def test_unmatched_words_go_to_general(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("xylophone something")
        topics = profiler.get_top_topics()
        general = [t for t in topics if t["category"] == "general"]
        assert len(general) >= 1

    def test_empty_text_no_crash(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("")
        assert profiler.get_top_topics() == []

    def test_response_summary_included(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("hello", "python analysis code")
        topics = profiler.get_top_topics()
        topic_names = [t["topic"] for t in topics]
        assert "python" in topic_names

    def test_interaction_count_increments(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code")
        profiler.update_from_task("javascript debug")
        stats = profiler.get_stats()
        assert stats["interaction_count"] == 2


class TestApplyDecay:
    def test_reduces_weights(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code debug")
        profiler.apply_decay(0.5)
        topics = profiler.get_top_topics()
        for t in topics:
            assert t["weight"] <= 0.5

    def test_removes_low_weight_topics(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code")
        # Decay aggressively until topics disappear
        for _ in range(50):
            profiler.apply_decay(0.8)
        topics = profiler.get_top_topics()
        assert len(topics) == 0

    def test_returns_removed_count(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code debug function class")
        # One aggressive decay to bring weights below 0.1
        removed = profiler.apply_decay(0.01)
        assert removed > 0


class TestGetTopTopics:
    def test_ordered_by_weight(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python python python")
        profiler.update_from_task("code")
        topics = profiler.get_top_topics()
        if len(topics) >= 2:
            assert topics[0]["weight"] >= topics[1]["weight"]

    def test_respects_limit(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task(
            "python code debug function class api git algorithm loop variable"
        )
        topics = profiler.get_top_topics(limit=3)
        assert len(topics) <= 3


class TestGetCategoryDistribution:
    def test_returns_categories(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code debug")
        profiler.update_from_task("stock market trade")
        dist = profiler.get_category_distribution()
        assert "coding" in dist
        assert "finance" in dist

    def test_empty_profiler(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        assert profiler.get_category_distribution() == {}


class TestGetStats:
    def test_empty_stats(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        stats = profiler.get_stats()
        assert stats["interaction_count"] == 0
        assert stats["topic_count"] == 0

    def test_after_interactions(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code")
        stats = profiler.get_stats()
        assert stats["interaction_count"] == 1
        assert stats["topic_count"] > 0


class TestGetProfileSummary:
    def test_empty_profiler_returns_empty(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        assert profiler.get_profile_summary() == ""

    def test_contains_interests(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code debug function class")
        summary = profiler.get_profile_summary()
        assert "User interests:" in summary
        assert "coding" in summary

    def test_contains_top_topics(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code debug")
        summary = profiler.get_profile_summary()
        assert "Top topics:" in summary
        assert "python" in summary

    def test_contains_interaction_count(self, tmp_path):
        profiler = UserProfiler(tmp_path / "test.db")
        profiler.update_from_task("python code")
        summary = profiler.get_profile_summary()
        assert "Total interactions:" in summary


class TestSharedDatabase:
    def test_coexists_with_memory(self, tmp_path):
        from ring0.memory import MemoryStore

        db = tmp_path / "shared.db"
        memory = MemoryStore(db)
        profiler = UserProfiler(db)

        memory.add(1, "observation", "test")
        profiler.update_from_task("python code")

        assert memory.count() == 1
        assert len(profiler.get_top_topics()) > 0
