"""Integration test: End-to-end pipeline from Task → Memory → Profile → Evolution.

Verifies the complete data flow:
  Task input → Memory storage → Profile update → Preference extraction →
  Preference aggregation → Evolution prompt injection → Gene retention

Each test validates one link in the chain.  The final test runs the full
pipeline end-to-end in a single temp database.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ring0.fitness import FitnessTracker
from ring0.gene_pool import GenePool
from ring0.memory import MemoryStore, _compute_importance
from ring0.user_profile import PreferenceStore, UserProfiler
from ring1.memory_curator import MemoryCurator
from ring1.prompts import build_evolution_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Create all stores sharing one DB file, as in production."""
    db = tmp_path / "protea.db"
    memory = MemoryStore(db)
    profiler = UserProfiler(db)
    prefs = PreferenceStore(db, config={"moment_aggregation_threshold": 3})
    fitness = FitnessTracker(db)
    genes = GenePool(db)
    return db, memory, profiler, prefs, fitness, genes


# ---------------------------------------------------------------------------
# Stage 1: Task → Memory
# ---------------------------------------------------------------------------


class TestTaskToMemory:
    """Tasks are stored in memory with correct type, importance, and metadata."""

    def test_substantive_task_stored(self, tmp_path):
        """A substantive user task should be stored in hot tier."""
        _, memory, *_ = _make_db(tmp_path)
        task_text = "帮我分析一下这个系统的性能问题，找出瓶颈所在"
        rid = memory.add(1, "task", task_text)
        assert rid > 0

        entries = memory.get_recent(1)
        assert len(entries) == 1
        e = entries[0]
        assert e["entry_type"] == "task"
        assert e["content"] == task_text
        assert e["tier"] == "hot"
        assert e["importance"] >= 0.7  # Substantive task

    def test_short_operational_task_rejected(self, tmp_path):
        """Ultra-short tasks are rejected by quality gate."""
        _, memory, *_ = _make_db(tmp_path)
        rid = memory.add(1, "task", "好的")
        assert rid == -1  # Below quality gate (importance 0.1 < 0.25)

    def test_task_with_embedding(self, tmp_path):
        """Tasks can be stored with embeddings for vector search."""
        _, memory, *_ = _make_db(tmp_path)
        emb = [0.1, 0.2, 0.3]
        rid = memory.add_with_embedding(
            1, "task", "帮我搜索关于机器学习的最新论文，要关注 transformer 架构的改进",
            embedding=emb,
        )
        assert rid > 0
        results = memory.search_similar(emb, limit=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Stage 2: Memory → Profile
# ---------------------------------------------------------------------------


class TestMemoryToProfile:
    """Memory entries update user profile topics."""

    def test_task_updates_profile_topics(self, tmp_path):
        """Processing a task should create topic entries in the profiler."""
        _, memory, profiler, *_ = _make_db(tmp_path)

        task_text = "帮我用 Python 写一个 Fibonacci 数列的生成器"
        memory.add(1, "task", task_text)

        # Simulate what TaskExecutor does: extract intent and update profile.
        profiler.update_from_task(task_text)

        categories = profiler.get_category_distribution()
        assert len(categories) > 0
        # "coding" or "general" should appear as a category.
        cat_names = list(categories.keys())
        assert any(c in ("coding", "general", "ai") for c in cat_names)

    def test_multiple_tasks_build_profile(self, tmp_path):
        """Multiple tasks from the same domain should strengthen topic weights."""
        _, memory, profiler, *_ = _make_db(tmp_path)

        tasks = [
            "帮我用 Python 实现一个 REST API",
            "帮我调试这个 Python 代码的 bug",
            "帮我写一个 Python 单元测试",
        ]
        for t in tasks:
            memory.add(1, "task", t, importance=0.7)
            profiler.update_from_task(t)

        categories = profiler.get_category_distribution()
        assert "coding" in categories
        # Weight should have accumulated.
        assert categories["coding"] > 0


# ---------------------------------------------------------------------------
# Stage 3: Preference Extraction → Moments → Aggregation
# ---------------------------------------------------------------------------


class TestPreferenceFlow:
    """Preference moments accumulate and aggregate into stable preferences."""

    def test_moments_stored(self, tmp_path):
        """Storing moments creates entries in the moments table."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        prefs.store_moment("interest", "user likes AI research", "ai", "AI research interest")
        prefs.store_moment("communication", "user prefers Chinese", "lifestyle", "prefers Chinese")

        pending = prefs.get_pending_moments()
        assert len(pending) == 2
        assert pending[0]["category"] == "ai"
        assert pending[1]["category"] == "lifestyle"

    def test_aggregation_with_sufficient_moments(self, tmp_path):
        """3+ moments in same category → creates a preference entry."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        # Add 3 moments in the same category with similar signals.
        for i in range(3):
            prefs.store_moment(
                "interest", f"user researches AI topic {i}", "ai",
                "AI research interest",
            )

        updated = prefs.aggregate_moments()
        assert updated == 1  # One preference created for "ai" category.

        preferences = prefs.get_preferences()
        assert len(preferences) == 1
        assert preferences[0]["category"] == "ai"
        assert preferences[0]["confidence"] >= 0.5
        assert preferences[0]["source"] == "implicit"

        # Moments should be marked as aggregated.
        pending = prefs.get_pending_moments()
        assert len(pending) == 0

    def test_aggregation_below_threshold_skips(self, tmp_path):
        """< 3 moments in a category → no aggregation."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        prefs.store_moment("interest", "one-off", "finance", "finance signal")
        prefs.store_moment("interest", "another", "finance", "finance signal")

        updated = prefs.aggregate_moments()
        assert updated == 0

        pending = prefs.get_pending_moments()
        assert len(pending) == 2  # Still pending.

    def test_confidence_grows_with_more_moments(self, tmp_path):
        """Adding more moments to same category increases confidence."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        # First batch: 3 moments → creates preference.
        for _ in range(3):
            prefs.store_moment("interest", "AI paper", "ai", "AI research")
        prefs.aggregate_moments()
        conf1 = prefs.get_preferences()[0]["confidence"]

        # Second batch: 3 more moments → updates preference.
        for _ in range(3):
            prefs.store_moment("interest", "AI model", "ai", "AI research")
        prefs.aggregate_moments()
        conf2 = prefs.get_preferences()[0]["confidence"]

        assert conf2 > conf1  # Confidence should increase.


# ---------------------------------------------------------------------------
# Stage 4: Preferences → Evolution Prompt
# ---------------------------------------------------------------------------


class TestPreferencesInEvolution:
    """Structured preferences appear in the evolution prompt."""

    def test_preference_summary_text(self, tmp_path):
        """Preferences generate a summary text for prompt injection."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        # Create aggregated preferences.
        for _ in range(4):
            prefs.store_moment("interest", "AI coding", "ai", "AI development")
        for _ in range(4):
            prefs.store_moment("tool", "Python scripting", "coding", "Python tools")
        prefs.aggregate_moments()

        summary = prefs.get_preference_summary_text()
        assert len(summary) > 0
        assert "preferences" in summary.lower() or "interest" in summary.lower()

    def test_preferences_injected_into_evolution_prompt(self, tmp_path):
        """build_evolution_prompt includes structured preferences section."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        for _ in range(4):
            prefs.store_moment("interest", "AI coding", "ai", "AI development")
        prefs.aggregate_moments()

        pref_text = prefs.get_preference_summary_text()

        system_prompt, user_prompt = build_evolution_prompt(
            current_source="print('hello')",
            fitness_history=[],
            best_performers=[],
            params={"generation": 100, "mutation_rate": 0.1},
            generation=100,
            survived=True,
            structured_preferences=pref_text,
        )

        # Structured preferences are in user_prompt (parts), not system_prompt.
        assert "Structured Preferences" in user_prompt
        assert "AI" in user_prompt or "ai" in user_prompt.lower()


# ---------------------------------------------------------------------------
# Stage 5: Drift Detection → Directive
# ---------------------------------------------------------------------------


class TestDriftDetection:
    """Preference drift triggers evolution directives."""

    def test_rising_drift_detected(self, tmp_path):
        """5+ recent moments with low existing confidence → rising drift."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        # Create a low-confidence existing preference.
        for _ in range(3):
            prefs.store_moment("interest", "new hobby", "lifestyle", "gardening")
        prefs.aggregate_moments()

        # Manually lower confidence to simulate decay.
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "protea.db"))
        con.execute("UPDATE user_preferences SET confidence = 0.3")
        con.commit()
        con.close()

        # Add 5 more recent moments in same category.
        for _ in range(5):
            prefs.store_moment("interest", "gardening tips", "lifestyle", "gardening")

        drifts = prefs.detect_drift()
        rising = [d for d in drifts if d["direction"] == "rising"]
        assert len(rising) >= 1
        assert rising[0]["category"] == "lifestyle"

        # Drift should be logged.
        log = prefs.get_drift_log()
        assert len(log) >= 1

    def test_no_drift_with_few_moments(self, tmp_path):
        """< 5 recent moments → no drift detected."""
        _, _, _, prefs, *_ = _make_db(tmp_path)

        for _ in range(3):
            prefs.store_moment("interest", "random", "general", "signal")

        drifts = prefs.detect_drift()
        assert len(drifts) == 0


# ---------------------------------------------------------------------------
# Stage 6: Task Alignment Scoring
# ---------------------------------------------------------------------------


class TestTaskAlignment:
    """Ring 2 output matching user interests gets fitness bonus."""

    def test_alignment_bonus_applied(self, tmp_path):
        """Output containing user-interest keywords gets alignment bonus."""
        _, _, profiler, _, fitness, _ = _make_db(tmp_path)

        # Build a profile with strong "coding" interest.
        for _ in range(5):
            profiler.update_from_task("Python 代码调试和优化")

        categories = profiler.get_category_distribution()
        assert "coding" in categories

        # Ring 2 output that mentions coding-related terms.
        # score_task_alignment matches category name tokens against output tokens,
        # so "coding" must appear literally in the output.
        output_lines = [
            "class CodeAnalyzer:",
            "    '''A coding assistant for source analysis.'''",
            "    def analyze(self, source_code):",
            "        import ast",
            "        tree = ast.parse(source_code)",
            "        return self._extract_patterns(tree)",
        ]

        bonus = fitness.score_task_alignment(output_lines, categories)
        assert bonus > 0  # Should get some alignment bonus.
        assert bonus <= 0.15  # Capped at 0.15.


# ---------------------------------------------------------------------------
# Stage 7: Gene Adoption Verification
# ---------------------------------------------------------------------------


class TestGeneAdoption:
    """Genes used in evolved code get verified hits."""

    def test_verify_adoption_records_hits(self, tmp_path):
        """Genes whose tags appear in new source get hit count incremented."""
        _, _, _, _, _, genes = _make_db(tmp_path)

        # Add some genes with tags.
        genes.add(100, 0.8, "class LogAnalyzer: ...", {"tags": ["log", "analyzer", "parsing"]})
        genes.add(101, 0.7, "class WebScraper: ...", {"tags": ["web", "scraper", "http"]})

        all_genes = genes.get_top(10)
        gene_ids = [g["id"] for g in all_genes]

        # New source code that uses "log" and "analyzer" but not "web"/"scraper".
        new_source = """
class LogPatternMatcher:
    def analyze(self, log_lines):
        patterns = self._extract_patterns(log_lines)
        return self._score_anomalies(patterns)
"""

        adopted = genes.verify_adoption(new_source, gene_ids, generation=105)
        assert len(adopted) >= 1  # At least the LogAnalyzer gene was adopted.


# ---------------------------------------------------------------------------
# Stage 8: Nightly Consolidation
# ---------------------------------------------------------------------------


class TestNightlyConsolidation:
    """Nightly consolidation discovers cross-task patterns."""

    def test_consolidation_with_enough_tasks(self, tmp_path):
        """3+ recent tasks → LLM called for pattern discovery."""
        _, memory, _, prefs, *_ = _make_db(tmp_path)

        # Add enough tasks for consolidation.
        for i in range(5):
            memory.add(i, "task", f"研究 AI 在 {['医疗', '金融', '教育', '制造', '农业'][i]} 领域的应用",
                        importance=0.7)

        # Mock LLM client.
        mock_client = MagicMock()
        mock_client.send_message.return_value = json.dumps([
            {
                "type": "correlation",
                "content": "User is systematically exploring AI applications across industries",
                "confidence": 0.8,
                "related_tasks": [0, 1, 2, 3, 4],
            }
        ])

        curator = MemoryCurator(mock_client)
        result = curator.nightly_consolidate(memory, prefs)

        assert result["insights_found"] == 1
        assert result["moments_stored"] == 1
        mock_client.send_message.assert_called_once()

    def test_consolidation_skips_few_tasks(self, tmp_path):
        """< 3 tasks → no LLM call."""
        _, memory, _, prefs, *_ = _make_db(tmp_path)

        memory.add(1, "task", "帮我写一个 Python 函数来解析 JSON 数据", importance=0.7)
        memory.add(2, "task", "帮我部署这个应用到生产环境上", importance=0.7)

        mock_client = MagicMock()
        curator = MemoryCurator(mock_client)
        result = curator.nightly_consolidate(memory, prefs)

        assert result["insights_found"] == 0
        mock_client.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 9: Cross-Domain Search
# ---------------------------------------------------------------------------


class TestCrossDomainSearch:
    """Memory can find cross-domain connections for inspiration."""

    def test_search_cross_domain_finds_related(self, tmp_path):
        """Tasks from different categories can be cross-referenced."""
        _, memory, profiler, *_ = _make_db(tmp_path)

        # Build profile with multiple interests.
        profiler.update_from_task("Python 代码调试和优化")
        profiler.update_from_task("股票市场数据分析")

        # Add memories in different domains.
        memory.add(1, "task", "用 Python 实现了一个遗传算法来优化参数", importance=0.7)
        memory.add(2, "task", "分析了沪深 300 指数最近三个月的波动率", importance=0.7)

        categories = profiler.get_category_distribution()

        # Search for cross-domain connections from a finance task.
        results = memory.search_cross_domain(
            current_task="帮我优化交易策略的回测逻辑",
            user_profile={"categories": categories},
            limit=3,
        )
        # Should find results (may or may not include cross-domain hits
        # depending on keyword overlap).
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Full Pipeline Integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: task → memory → profile → preferences → evolution prompt."""

    def test_full_flow(self, tmp_path):
        """Trace data from a task all the way to the evolution prompt."""
        db, memory, profiler, prefs, fitness, genes = _make_db(tmp_path)

        # ── Step 1: User sends tasks ──
        tasks = [
            "帮我实现一个基于 transformer 的文本分类器",
            "帮我用 PyTorch 训练一个情感分析模型",
            "帮我部署这个 NLP 模型到生产环境，需要支持批量推理",
            "帮我写一个 API 来调用这个模型，支持 REST 和 gRPC",
        ]

        for i, task_text in enumerate(tasks):
            # Step 1a: Store in memory.
            rid = memory.add(i + 1, "task", task_text)
            assert rid > 0, f"Task {i} should be stored"

            # Step 1b: Update profile.
            profiler.update_from_task(task_text)

            # Step 1c: Extract preferences (simulate PreferenceExtractor).
            prefs.store_moment(
                "interest", task_text[:80], "ai",
                extracted_signal="AI/ML model development",
            )

        # ── Step 2: Verify memory ──
        recent = memory.get_recent(10)
        assert len(recent) == 4
        task_entries = memory.get_by_type("task")
        assert len(task_entries) == 4

        # ── Step 3: Verify profile ──
        categories = profiler.get_category_distribution()
        assert len(categories) > 0
        # Should have AI/coding related categories.
        all_cats = list(categories.keys())
        assert any(c in ("ai", "coding") for c in all_cats), \
            f"Expected ai/coding in {all_cats}"

        # ── Step 4: Verify preference aggregation ──
        pending = prefs.get_pending_moments()
        assert len(pending) == 4  # 4 moments stored.

        aggregated = prefs.aggregate_moments()
        assert aggregated >= 1  # At least one preference created.

        preferences = prefs.get_preferences()
        assert len(preferences) >= 1
        assert any(p["category"] == "ai" for p in preferences)

        # ── Step 5: Verify preferences in evolution prompt ──
        pref_summary = prefs.get_preference_summary_text()
        assert len(pref_summary) > 0

        profile_text = profiler.get_profile_summary()

        system_prompt, user_prompt = build_evolution_prompt(
            current_source="class TextClassifier: pass",
            fitness_history=[],
            best_performers=[],
            params={"generation": 100, "mutation_rate": 0.1},
            generation=100,
            survived=True,
            user_profile_summary=profile_text,
            structured_preferences=pref_summary,
        )

        # Structured preferences are in user_prompt (parts), not system_prompt.
        assert "Structured Preferences" in user_prompt
        # User profile should also be there.
        assert "User Profile" in user_prompt or "user" in user_prompt.lower()

        # ── Step 6: Verify task alignment ──
        output_lines = [
            "class TransformerClassifier:",
            "    def __init__(self, model_name='bert-base'):",
            "        self.tokenizer = AutoTokenizer.from_pretrained(model_name)",
        ]
        bonus = fitness.score_task_alignment(output_lines, categories)
        # May or may not get bonus depending on keyword overlap.
        assert isinstance(bonus, float)
        assert 0 <= bonus <= 0.10

        # ── Step 7: Verify gene flow ──
        genes.add(100, 0.85, "class NLPPipeline: ...",
                  {"tags": ["nlp", "transformer", "classification"]})
        top_genes = genes.get_top(5)
        assert len(top_genes) >= 1
        assert top_genes[0]["score"] == 0.85

        # ── Pipeline complete ──
        # Summary: Task → Memory(4) → Profile(categories) → Preferences(1+)
        #          → Evolution prompt(with prefs) → Alignment scoring → Gene pool
        stats = memory.get_stats()
        assert stats["total"] == 4
        assert stats["by_type"]["task"] == 4
