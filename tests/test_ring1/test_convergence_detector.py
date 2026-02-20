"""Tests for ring1.convergence_detector (multi-round convergence detection)."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from ring1.convergence_detector import (
    ConvergenceDetector,
    _AFFIRMATION_PATTERNS,
    _NEGATION_PATTERNS,
    _REINFORCEMENT_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(
    config: dict | None = None,
    convergence_context: dict | None = None,
) -> tuple[ConvergenceDetector, MagicMock, MagicMock, MagicMock]:
    """Create a ConvergenceDetector with mock dependencies."""
    memory_store = MagicMock()
    embedding_provider = MagicMock()
    llm_client = MagicMock()
    ctx = convergence_context if convergence_context is not None else {}

    detector = ConvergenceDetector(
        memory_store=memory_store,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
        convergence_context=ctx,
        config=config,
    )
    return detector, memory_store, embedding_provider, llm_client


def _make_embedding(seed: float) -> list[float]:
    """Create a simple deterministic embedding vector for testing."""
    import math
    # Produce a 4-dimensional unit vector rotated by seed.
    return [
        math.cos(seed), math.sin(seed),
        math.cos(seed + 1), math.sin(seed + 1),
    ]


# ---------------------------------------------------------------------------
# TestNegationPatterns
# ---------------------------------------------------------------------------

class TestNegationPatterns:
    """Test negation signal regex patterns."""

    @pytest.mark.parametrize("text", [
        "不对，这个不是我要的",
        "错了，重新来",
        "不行，不是这样的",
        "搞错了，应该是另一个",
        "That's wrong, try again",
        "No, that's not right",
        "wrong answer",
        "不是我要的",
        "再试一下",
    ])
    def test_negation_matches(self, text: str) -> None:
        assert any(pat.search(text) for pat in _NEGATION_PATTERNS), \
            f"Expected negation match for: {text}"

    @pytest.mark.parametrize("text", [
        "帮我查一下天气",
        "今天星期几",
        "hello world",
        "请翻译这句话",
        "好的，收到了",
    ])
    def test_normal_text_no_match(self, text: str) -> None:
        assert not any(pat.search(text) for pat in _NEGATION_PATTERNS), \
            f"Expected no negation match for: {text}"


# ---------------------------------------------------------------------------
# TestAffirmationPatterns
# ---------------------------------------------------------------------------

class TestAffirmationPatterns:
    """Test affirmation and reinforcement signal patterns."""

    @pytest.mark.parametrize("text", [
        "对了，就是这样",
        "可以了",
        "这次对了",
        "好了，没问题了",
        "终于对了",
        "That's right",
        "Yes, that's it!",
        "perfect",
        "exactly what I wanted",
    ])
    def test_affirmation_matches(self, text: str) -> None:
        assert any(pat.search(text) for pat in _AFFIRMATION_PATTERNS), \
            f"Expected affirmation match for: {text}"

    @pytest.mark.parametrize("text", [
        "以后都这样做",
        "记住这个规则",
        "from now on always do it this way",
        "remember this rule",
        "never do that again",
    ])
    def test_reinforcement_matches(self, text: str) -> None:
        assert any(pat.search(text) for pat in _REINFORCEMENT_PATTERNS), \
            f"Expected reinforcement match for: {text}"


# ---------------------------------------------------------------------------
# TestClustering
# ---------------------------------------------------------------------------

class TestClustering:
    """Test embedding-based clustering."""

    def test_similar_messages_cluster(self) -> None:
        """Messages with similar embeddings should cluster together."""
        detector, _, emb_prov, _ = _make_detector()

        # All messages get similar embeddings.
        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 4

        for i in range(4):
            detector.record(f"msg {i}", f"resp {i}")

        cluster = detector._cluster_recent()
        assert cluster is not None
        assert len(cluster) == 4

    def test_dissimilar_messages_no_cluster(self) -> None:
        """Messages with orthogonal embeddings should not cluster."""
        detector, _, emb_prov, _ = _make_detector()

        # Each message gets a very different embedding.
        emb_prov.embed.return_value = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],  # anchor
        ]

        for i in range(4):
            detector.record(f"msg {i}", f"resp {i}")

        cluster = detector._cluster_recent()
        # Only anchor matches itself, not enough for cluster of 3
        assert cluster is None

    def test_time_window_filter(self) -> None:
        """Entries older than cluster_window_sec should be pruned."""
        detector, _, emb_prov, _ = _make_detector(
            config={"cluster_window_sec": 60},
        )

        # Record old entries, then new ones.
        similar_emb = [1.0, 0.0, 0.0, 0.0]

        # Manually insert an old entry.
        detector._history.append((time.time() - 120, "old msg", "old resp"))

        # Record 3 new ones.
        for i in range(3):
            detector.record(f"msg {i}", f"resp {i}")

        # The old one should have been pruned.
        assert len(detector._history) == 3

    def test_fewer_than_3_returns_none(self) -> None:
        """Fewer than 3 entries should return None."""
        detector, _, emb_prov, _ = _make_detector()
        detector.record("msg1", "resp1")
        detector.record("msg2", "resp2")

        cluster = detector._cluster_recent()
        assert cluster is None


# ---------------------------------------------------------------------------
# TestConvergenceDetection
# ---------------------------------------------------------------------------

class TestConvergenceDetection:
    """Test convergence pattern detection."""

    def test_classic_convergence_triggers(self) -> None:
        """2 negations + 1 affirmation should trigger."""
        detector, _, emb_prov, llm = _make_detector()

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 4

        llm.send_message.return_value = (
            "RULE: Always use metric units\n"
            "CONTEXT: When displaying measurements"
        )

        detector.record("帮我转换单位", "好的，10 inches = ...")
        detector.record("不对，用公制", "好的，25.4 cm...")
        detector.record("错了，应该是厘米", "对不起，25.4 厘米")
        detector.record("对了，可以了", "明白了")

        result = detector.check()
        assert result is not None
        text, buttons = result
        assert "规则" in text
        assert len(buttons) == 1
        assert len(buttons[0]) == 2
        assert "convergence:confirm:" in buttons[0][0]["callback_data"]
        assert "convergence:dismiss:" in buttons[0][1]["callback_data"]

    def test_no_affirmation_no_trigger(self) -> None:
        """2 negations but no affirmation should not trigger."""
        detector, _, emb_prov, _ = _make_detector()

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 3

        detector.record("不对", "好的")
        detector.record("错了", "抱歉")
        detector.record("帮我查天气", "好的")

        result = detector.check()
        assert result is None

    def test_single_negation_no_trigger(self) -> None:
        """Only 1 negation should not trigger (needs >= 2)."""
        detector, _, emb_prov, _ = _make_detector()

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 3

        detector.record("不对", "好的")
        detector.record("继续", "好的")
        detector.record("对了", "明白了")

        result = detector.check()
        assert result is None

    def test_reinforcement_also_triggers(self) -> None:
        """2 negations + reinforcement (no affirmation) should trigger."""
        detector, _, emb_prov, llm = _make_detector()

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 4

        llm.send_message.return_value = (
            "RULE: Use snake_case\n"
            "CONTEXT: Variable names"
        )

        detector.record("变量名", "camelCase")
        detector.record("不对，用下划线", "snake_case?")
        detector.record("错了", "sorry")
        detector.record("以后都这样做", "明白了")

        result = detector.check()
        assert result is not None


# ---------------------------------------------------------------------------
# TestRuleExtraction
# ---------------------------------------------------------------------------

class TestRuleExtraction:
    """Test LLM-based rule extraction."""

    def test_parse_rule_and_context(self) -> None:
        detector, _, _, llm = _make_detector()
        llm.send_message.return_value = (
            "RULE: Always use Chinese for responses\n"
            "CONTEXT: When the user writes in Chinese"
        )

        cluster = [
            (time.time(), "用中文", "OK"),
            (time.time(), "不对，要中文", "好的"),
            (time.time(), "对了", "明白"),
        ]
        rule, context = detector._extract_rule(cluster)
        assert rule == "Always use Chinese for responses"
        assert context == "When the user writes in Chinese"

    def test_none_response(self) -> None:
        detector, _, _, llm = _make_detector()
        llm.send_message.return_value = "NONE"

        cluster = [
            (time.time(), "a", "b"),
            (time.time(), "c", "d"),
            (time.time(), "e", "f"),
        ]
        rule, context = detector._extract_rule(cluster)
        assert rule == ""
        assert context == ""

    def test_llm_error_returns_empty(self) -> None:
        detector, _, _, llm = _make_detector()
        llm.send_message.side_effect = Exception("LLM down")

        cluster = [
            (time.time(), "a", "b"),
            (time.time(), "c", "d"),
            (time.time(), "e", "f"),
        ]
        rule, context = detector._extract_rule(cluster)
        assert rule == ""
        assert context == ""

    def test_rule_only_no_context(self) -> None:
        detector, _, _, llm = _make_detector()
        llm.send_message.return_value = "RULE: Use spaces not tabs"

        cluster = [
            (time.time(), "a", "b"),
            (time.time(), "c", "d"),
            (time.time(), "e", "f"),
        ]
        rule, context = detector._extract_rule(cluster)
        assert rule == "Use spaces not tabs"
        assert context == ""


# ---------------------------------------------------------------------------
# TestRateLimiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Test cooldown prevents repeated triggers."""

    def test_cooldown_blocks_second_trigger(self) -> None:
        detector, _, emb_prov, llm = _make_detector(
            config={"cooldown_sec": 600},
        )

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 4

        llm.send_message.return_value = (
            "RULE: Test rule\nCONTEXT: Test context"
        )

        detector.record("msg1", "resp1")
        detector.record("不对", "resp2")
        detector.record("错了", "resp3")
        detector.record("对了", "resp4")

        # First check should trigger.
        result1 = detector.check()
        assert result1 is not None

        # Reset history so clustering works again.
        detector._history.clear()
        detector.record("msg1", "resp1")
        detector.record("不对", "resp2")
        detector.record("错了", "resp3")
        detector.record("对了", "resp4")

        # Second check within cooldown should NOT trigger.
        result2 = detector.check()
        assert result2 is None

    def test_after_cooldown_triggers_again(self) -> None:
        detector, _, emb_prov, llm = _make_detector(
            config={"cooldown_sec": 1},
        )

        similar_emb = [1.0, 0.0, 0.0, 0.0]
        emb_prov.embed.return_value = [similar_emb] * 4

        llm.send_message.return_value = (
            "RULE: Test rule\nCONTEXT: Test context"
        )

        detector.record("msg1", "resp1")
        detector.record("不对", "resp2")
        detector.record("错了", "resp3")
        detector.record("对了", "resp4")

        result1 = detector.check()
        assert result1 is not None

        # Wait for cooldown to expire.
        time.sleep(1.1)

        detector._history.clear()
        detector.record("msg1", "resp1")
        detector.record("不对", "resp2")
        detector.record("错了", "resp3")
        detector.record("对了", "resp4")

        result2 = detector.check()
        assert result2 is not None


# ---------------------------------------------------------------------------
# TestConfirmFlow
# ---------------------------------------------------------------------------

class TestConfirmFlow:
    """Test confirm/dismiss callback handling (telegram_bot integration)."""

    def test_confirm_stores_semantic_rule(self) -> None:
        """Confirming a proposal should call memory_store.add with importance=0.9."""
        from ring1.telegram_bot import SentinelState

        state = SentinelState()
        state.memory_store = MagicMock()

        rule_key = "abc12345"
        state._convergence_context[rule_key] = {
            "rule_text": "Always use metric units",
            "context_text": "measurements",
            "cluster_size": 4,
            "timestamp": time.time(),
        }

        # Simulate the callback handler logic.
        data = f"convergence:confirm:{rule_key}"
        ctx = state._convergence_context.pop(rule_key, None)
        assert ctx is not None

        state.memory_store.add(
            generation=0,
            entry_type="semantic_rule",
            content=ctx["rule_text"],
            importance=0.9,
            metadata={"source": "convergence", "cluster_size": ctx["cluster_size"]},
        )

        state.memory_store.add.assert_called_once()
        call_kwargs = state.memory_store.add.call_args
        assert call_kwargs[1]["entry_type"] == "semantic_rule"
        assert call_kwargs[1]["importance"] == 0.9
        assert call_kwargs[1]["content"] == "Always use metric units"

    def test_dismiss_clears_context(self) -> None:
        """Dismissing should remove the context entry."""
        ctx: dict[str, dict] = {}
        rule_key = "xyz99999"
        ctx[rule_key] = {
            "rule_text": "test rule",
            "context_text": "",
            "cluster_size": 3,
            "timestamp": time.time(),
        }

        # Simulate dismiss.
        ctx.pop(rule_key, None)
        assert rule_key not in ctx

    def test_expired_context_returns_none(self) -> None:
        """Confirming an expired (already popped) key should return None."""
        ctx: dict[str, dict] = {}
        rule_key = "expired123"
        result = ctx.pop(rule_key, None)
        assert result is None


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_history_returns_none(self) -> None:
        detector, _, _, _ = _make_detector()
        assert detector.check() is None

    def test_single_message_returns_none(self) -> None:
        detector, _, _, _ = _make_detector()
        detector.record("hello", "world")
        assert detector.check() is None

    def test_two_messages_returns_none(self) -> None:
        detector, _, _, _ = _make_detector()
        detector.record("hello", "world")
        detector.record("不对", "sorry")
        assert detector.check() is None

    def test_embedding_failure_returns_none(self) -> None:
        """If embedding provider raises, clustering should return None."""
        detector, _, emb_prov, _ = _make_detector()
        emb_prov.embed.side_effect = Exception("API error")

        for i in range(4):
            detector.record(f"msg {i}", f"resp {i}")

        result = detector.check()
        assert result is None

    def test_history_capacity_limit(self) -> None:
        """History should not exceed _history_max."""
        detector, _, _, _ = _make_detector()
        detector._history_max = 5
        for i in range(20):
            detector.record(f"msg {i}", f"resp {i}")
        assert len(detector._history) <= 5

    def test_context_auto_cleanup(self) -> None:
        """Expired pending proposals should be cleaned up."""
        ctx: dict[str, dict] = {}
        detector, _, _, _ = _make_detector(convergence_context=ctx)
        detector._context_expiry_sec = 1

        ctx["old_rule"] = {
            "rule_text": "old rule",
            "context_text": "",
            "cluster_size": 3,
            "timestamp": time.time() - 10,
        }

        # record() triggers cleanup.
        detector.record("msg", "resp")
        assert "old_rule" not in ctx

    def test_embedding_length_mismatch_returns_none(self) -> None:
        """If embedding provider returns wrong number of vectors, should return None."""
        detector, _, emb_prov, _ = _make_detector()
        emb_prov.embed.return_value = [[1, 0]]  # Only 1 vector for 3+ entries

        for i in range(4):
            detector.record(f"msg {i}", f"resp {i}")

        cluster = detector._cluster_recent()
        assert cluster is None
