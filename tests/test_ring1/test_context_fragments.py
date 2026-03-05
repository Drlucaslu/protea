"""Tests for ring1.context_fragments — fragment-based context selection."""

from __future__ import annotations

import pytest

from ring1.context_fragments import (
    Fragment,
    FragmentRegistry,
    _classify_text,
    _cosine_similarity,
    _estimate_tokens,
    _task_category_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEmbeddingProvider:
    """Deterministic embedding provider for testing."""

    def __init__(self, dim=8):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return simple hash-based vectors for deterministic testing."""
        results = []
        for text in texts:
            vec = [0.0] * self._dim
            for i, ch in enumerate(text[:self._dim]):
                vec[i % self._dim] = float(ord(ch)) / 128.0
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results

    def dimension(self) -> int:
        return self._dim


class NoOpProvider:
    """Zero-vector provider (mimics NoOpEmbedding)."""

    def embed(self, texts):
        return [[0.0] * 8] * len(texts)

    def dimension(self):
        return 8


def _make_state() -> dict:
    return {"generation": 5, "alive": True, "paused": False, "last_score": 8.5, "last_survived": True}


def _make_registry(token_budget=3000, provider=None) -> FragmentRegistry:
    return FragmentRegistry(provider or FakeEmbeddingProvider(), token_budget=token_budget)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("hello world") > 0

    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_long_text(self):
        text = "a" * 4000
        assert _estimate_tokens(text) == 1000


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0


class TestClassifyText:
    def test_coding(self):
        assert _classify_text("debug the python function") == "coding"

    def test_general(self):
        # Text with no category keyword matches returns "general"
        assert _classify_text("tell me a joke please") == "general"


class TestTaskCategorySet:
    def test_multi_category(self):
        cats = _task_category_set("write python code to fetch json from url")
        assert "coding" in cats
        # "json" is in "data" category, "url" is in "web"
        assert len(cats) >= 2


# ---------------------------------------------------------------------------
# FragmentRegistry
# ---------------------------------------------------------------------------

class TestCollect:
    def test_always_has_state(self):
        reg = _make_registry()
        frags = reg.collect(_make_state(), "")
        assert any(f.section == "state" for f in frags)

    def test_ring2_included(self):
        reg = _make_registry()
        frags = reg.collect(_make_state(), "print('hello')")
        assert any(f.section == "ring2" for f in frags)

    def test_ring2_excluded_when_empty(self):
        reg = _make_registry()
        frags = reg.collect(_make_state(), "")
        assert not any(f.section == "ring2" for f in frags)

    def test_all_sections(self):
        reg = _make_registry()
        frags = reg.collect(
            _make_state(),
            "print('hello')",
            memories=[{"generation": 1, "content": "learned something"}],
            recommended_skills=[{"name": "web_search", "description": "search the web"}],
            other_skills=[{"name": "calc", "description": "calculator"}],
            chat_history=[("hi", "hello")],
            recalled=[{"generation": 1, "content": "old memory"}],
            semantic_rules=[{"content": "always check errors"}],
            strategies=[{"content": "use caching"}],
            reflections=[{"content": "reflection lesson"}],
        )
        sections = {f.section for f in frags}
        assert "state" in sections
        assert "ring2" in sections
        assert "memories" in sections
        assert "skills" in sections
        assert "skills_list" in sections
        assert "history" in sections
        assert "recalled" in sections
        assert "rules" in sections
        assert "strategies" in sections
        assert "reflections" in sections


class TestRank:
    def test_global_always_score_1(self):
        reg = _make_registry()
        frags = reg.collect(_make_state(), "print('hello')")
        ranked = reg.rank(frags, "write python code")
        for frag, score in ranked:
            if frag.tag == "global":
                assert score == 1.0

    def test_scores_sorted_descending(self):
        reg = _make_registry()
        frags = reg.collect(
            _make_state(), "print('hello')",
            memories=[{"generation": 1, "content": "debug python bug"}],
            strategies=[{"content": "stock market analysis"}],
        )
        ranked = reg.rank(frags, "fix python bug in code")
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


class TestSelect:
    def test_respects_budget(self):
        reg = _make_registry(token_budget=100)
        frags = [
            Fragment(tag="global", section="state", content="state " * 10, importance=1.0),
            Fragment(tag="coding", section="ring2", content="code " * 200, importance=0.4),
        ]
        ranked = [(f, 1.0 if f.tag == "global" else 0.5) for f in frags]
        selected = reg.select(ranked)
        total = sum(f.token_est for f in selected)
        assert total <= 100

    def test_globals_always_included(self):
        reg = _make_registry(token_budget=5000)
        frags = [
            Fragment(tag="global", section="state", content="state info", importance=1.0),
            Fragment(tag="coding", section="ring2", content="code", importance=0.4),
        ]
        ranked = [(f, 1.0 if f.tag == "global" else 0.5) for f in frags]
        selected = reg.select(ranked)
        assert any(f.tag == "global" for f in selected)

    def test_empty_input(self):
        reg = _make_registry()
        assert reg.select([]) == []


class TestAssemble:
    def test_section_ordering(self):
        reg = _make_registry()
        frags = [
            Fragment(tag="coding", section="ring2", content="## Ring 2", importance=0.4),
            Fragment(tag="global", section="state", content="## State", importance=1.0),
            Fragment(tag="global", section="history", content="## History", importance=0.8),
        ]
        result = reg.assemble(frags)
        state_pos = result.index("## State")
        history_pos = result.index("## History")
        ring2_pos = result.index("## Ring 2")
        assert state_pos < history_pos < ring2_pos


class TestKeywordFallback:
    """Test that ranking works without embeddings (NoOp provider)."""

    def test_keyword_match_boosts_score(self):
        reg = _make_registry(provider=NoOpProvider())
        coding_frag = Fragment(tag="coding", section="ring2", content="python code", importance=0.5)
        finance_frag = Fragment(tag="finance", section="strategies", content="stock market", importance=0.5)
        # Task is about coding — coding frag should score higher
        ranked = reg.rank([coding_frag, finance_frag], "debug python function")
        # Find scores
        scores = {f.tag: s for f, s in ranked}
        assert scores["coding"] > scores["finance"]


class TestEndToEnd:
    """Full collect → rank → select → assemble pipeline."""

    def test_full_pipeline(self):
        reg = _make_registry(token_budget=3000)
        frags = reg.collect(
            _make_state(),
            "import os\nprint(os.getcwd())",
            memories=[{"generation": 1, "content": "learned caching"}],
            chat_history=[("what time is it", "it is 3pm")],
        )
        ranked = reg.rank(frags, "write a python script")
        selected = reg.select(ranked)
        result = reg.assemble(selected)
        assert "Protea State" in result
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_no_embedding(self):
        """Without embedding provider, keyword fallback still works."""
        reg = _make_registry(provider=NoOpProvider())
        frags = reg.collect(
            _make_state(), "",
            memories=[{"generation": 1, "content": "test memory"}],
        )
        ranked = reg.rank(frags, "hello")
        selected = reg.select(ranked)
        result = reg.assemble(selected)
        assert "Protea State" in result
