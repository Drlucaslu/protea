"""Tests for ring0.fitness — FitnessTracker and evaluate_output."""

from __future__ import annotations

import json

from ring0.fitness import (
    FitnessTracker,
    _is_real_error_line,
    _output_fingerprint,
    compute_novelty,
    evaluate_output,
)


class TestRecord:
    """record() should insert rows and return their rowid."""

    def test_insert_returns_rowid(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        rid = tracker.record(
            generation=1,
            commit_hash="abc123",
            score=0.85,
            runtime_sec=1.2,
            survived=True,
        )
        assert rid == 1

    def test_successive_inserts_increment_rowid(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        r1 = tracker.record(1, "aaa", 0.5, 1.0, True)
        r2 = tracker.record(1, "bbb", 0.6, 1.1, False)
        assert r2 == r1 + 1


class TestGetBest:
    """get_best() should return entries sorted by score descending."""

    def test_returns_sorted_by_score(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "low", 0.1, 1.0, True)
        tracker.record(1, "high", 0.9, 1.0, True)
        tracker.record(1, "mid", 0.5, 1.0, True)

        best = tracker.get_best(n=3)
        scores = [entry["score"] for entry in best]
        assert scores == [0.9, 0.5, 0.1]

    def test_limits_to_n(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(10):
            tracker.record(1, f"hash{i}", float(i), 1.0, True)

        best = tracker.get_best(n=3)
        assert len(best) == 3
        assert best[0]["score"] == 9.0

    def test_empty_database_returns_empty_list(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_best() == []


class TestGetGenerationStats:
    """get_generation_stats() should compute correct aggregates."""

    def test_computes_correct_stats(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "a", 0.2, 1.0, True)
        tracker.record(1, "b", 0.8, 1.0, True)
        tracker.record(1, "c", 0.5, 1.0, False)

        stats = tracker.get_generation_stats(generation=1)
        assert stats is not None
        assert stats["count"] == 3
        assert stats["max_score"] == 0.8
        assert stats["min_score"] == 0.2
        assert abs(stats["avg_score"] - 0.5) < 1e-9

    def test_returns_none_for_missing_generation(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_generation_stats(generation=99) is None

    def test_ignores_other_generations(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "a", 1.0, 1.0, True)
        tracker.record(2, "b", 0.0, 1.0, True)

        stats = tracker.get_generation_stats(generation=1)
        assert stats is not None
        assert stats["count"] == 1
        assert stats["avg_score"] == 1.0


class TestGetHistory:
    """get_history() should return entries in reverse chronological order."""

    def test_returns_reverse_order(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "first", 0.1, 1.0, True)
        tracker.record(2, "second", 0.2, 1.0, True)
        tracker.record(3, "third", 0.3, 1.0, True)

        history = tracker.get_history(limit=10)
        hashes = [entry["commit_hash"] for entry in history]
        assert hashes == ["third", "second", "first"]

    def test_respects_limit(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(10):
            tracker.record(i, f"hash{i}", float(i), 1.0, True)

        history = tracker.get_history(limit=3)
        assert len(history) == 3
        # Most recent first.
        assert history[0]["commit_hash"] == "hash9"

    def test_empty_database_returns_empty_list(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_history() == []


class TestEvaluateOutput:
    """evaluate_output() multi-factor scoring."""

    def test_empty_output_survivor_gets_base_plus_novelty(self):
        # Empty output: base=0.50 + novelty=0.05 (no history = full novelty)
        score, detail = evaluate_output([], survived=True, elapsed=60, max_runtime=60)
        assert score == 0.55
        assert detail["basis"] == "survived"
        assert detail["volume"] == 0.0
        assert detail["novelty"] == 0.05

    def test_rich_diverse_output_scores_high(self):
        lines = [f"result_{i}: {i * 3.14:.4f}" for i in range(60)]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert score > 0.80
        assert detail["volume"] == 0.10  # saturated at 50+ (max 0.10)
        assert detail["diversity"] > 0.05

    def test_duplicate_output_low_diversity(self):
        lines = ["same line"] * 50
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        # 1 unique / 50 total → diversity near 0
        assert detail["diversity"] < 0.01

    def test_structured_output_bonus(self):
        lines = [
            '{"key": "value"}',
            '| col1 | col2 |',
            'status: running',
            'normal line',
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["structure"] > 0.0

    def test_error_penalty_applied(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "main.py", line 1',
            "ZeroDivisionError: division by zero",
            "Another error occurred",
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["error_penalty"] > 0.0
        # Score should still be >= 0.50 (floor for survivors)
        assert score >= 0.50

    def test_failure_capped_below_survivor(self):
        # Best possible failure: elapsed == max_runtime → ratio 0.99
        score, detail = evaluate_output(
            ["some output"], survived=False, elapsed=59.4, max_runtime=60,
        )
        assert detail["basis"] == "failure"
        assert score < 0.50

    def test_failure_zero_runtime(self):
        score, detail = evaluate_output([], survived=False, elapsed=0, max_runtime=60)
        assert score == 0.0

    def test_failure_zero_max_runtime(self):
        score, detail = evaluate_output([], survived=False, elapsed=10, max_runtime=0)
        assert score == 0.0

    def test_whitespace_lines_ignored(self):
        lines = ["  ", "\t", "", "actual output"]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["meaningful_lines"] == 1


class TestDetailColumn:
    """Schema migration adds detail column, record() stores it."""

    def test_detail_stored_and_retrievable(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        detail = {"basis": "survived", "volume": 0.15}
        tracker.record(1, "abc", 0.75, 60.0, True, detail=detail)

        rows = tracker.get_history(limit=1)
        assert len(rows) == 1
        stored = json.loads(rows[0]["detail"])
        assert stored["volume"] == 0.15

    def test_detail_none_by_default(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "abc", 0.5, 60.0, True)

        rows = tracker.get_history(limit=1)
        assert rows[0]["detail"] is None

    def test_migration_idempotent(self, tmp_path):
        """Creating FitnessTracker twice should not error."""
        db = tmp_path / "fitness.db"
        FitnessTracker(db)
        FitnessTracker(db)  # second init — ALTER TABLE should be no-op


class TestNovelty:
    """Novelty scoring via Jaccard distance."""

    def test_no_history_returns_full_novelty(self):
        tokens = {"hello", "world", "test"}
        assert compute_novelty(tokens, []) == 1.0

    def test_identical_output_zero_novelty(self):
        tokens = {"hello", "world"}
        assert compute_novelty(tokens, [tokens]) == 0.0

    def test_completely_different_output_full_novelty(self):
        current = {"aaa", "bbb", "ccc"}
        previous = [{"xxx", "yyy", "zzz"}]
        assert compute_novelty(current, previous) == 1.0

    def test_partial_overlap(self):
        current = {"hello", "world", "new"}
        previous = [{"hello", "world", "old"}]
        novelty = compute_novelty(current, previous)
        # 2 shared out of 4 union → jaccard=0.5, distance=0.5
        assert 0.4 < novelty < 0.6

    def test_multiple_previous_outputs(self):
        current = {"aaa", "bbb"}
        prev1 = {"aaa", "bbb"}      # identical → distance 0
        prev2 = {"xxx", "yyy"}      # disjoint → distance 1.0
        novelty = compute_novelty(current, [prev1, prev2])
        assert 0.4 < novelty < 0.6  # average of 0.0 and 1.0

    def test_empty_current_returns_full_novelty(self):
        assert compute_novelty(set(), [{"aaa"}]) == 1.0

    def test_fingerprint_extracts_tokens(self):
        lines = ["result_42: 3.14", "status: running", "x"]
        fp = _output_fingerprint(lines)
        assert "result_42" in fp
        assert "status" in fp
        assert "running" in fp
        # Short tokens and pure numbers excluded.
        assert "x" not in fp
        assert "314" not in fp


class TestFunctionalScoring:
    """Functional bonus for I/O-related output."""

    def test_http_url_triggers_functional(self):
        lines = ["Fetched https://api.example.com/data"]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["functional"] > 0.0

    def test_server_listening_triggers_functional(self):
        lines = ["Server started on port 8080"] * 5
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["functional"] > 0.0

    def test_pure_computation_no_functional(self):
        lines = [f"pi digit {i}: {i}" for i in range(50)]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["functional"] == 0.0


class TestNoveltyInScoring:
    """Novelty component in evaluate_output with fingerprints."""

    def test_novel_output_gets_bonus(self):
        lines = [f"unique_concept_{i}" for i in range(50)]
        score, detail = evaluate_output(
            lines, survived=True, elapsed=60, max_runtime=60,
            recent_fingerprints=[],
        )
        assert detail["novelty"] == 0.05  # full novelty (no history)

    def test_repeated_output_loses_novelty(self):
        lines = [f"result_{i}: value" for i in range(50)]
        fp = _output_fingerprint(lines)
        score, detail = evaluate_output(
            lines, survived=True, elapsed=60, max_runtime=60,
            recent_fingerprints=[fp],
        )
        assert detail["novelty"] == 0.0  # identical to history

    def test_fingerprint_stored_in_detail(self):
        lines = ["hello world test"]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert "fingerprint" in detail
        assert isinstance(detail["fingerprint"], list)

    def test_error_signatures_stored_in_detail(self):
        lines = [
            "normal output",
            "AttributeError: 'list' has no attribute 'get'",
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert "error_signatures" in detail
        assert len(detail["error_signatures"]) > 0


class TestRecentFingerprints:
    """FitnessTracker.get_recent_fingerprints retrieves stored data."""

    def test_returns_fingerprints_from_detail(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        detail = {"fingerprint": ["hello", "world"], "basis": "survived"}
        tracker.record(1, "abc", 0.8, 60.0, True, detail=detail)

        fps = tracker.get_recent_fingerprints(limit=5)
        assert len(fps) == 1
        assert fps[0] == {"hello", "world"}

    def test_skips_entries_without_fingerprint(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "abc", 0.8, 60.0, True, detail={"basis": "survived"})
        tracker.record(2, "def", 0.8, 60.0, True, detail={"fingerprint": ["foo"]})

        fps = tracker.get_recent_fingerprints(limit=5)
        assert len(fps) == 1
        assert fps[0] == {"foo"}

    def test_ignores_failed_entries(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        detail = {"fingerprint": ["fail_token"]}
        tracker.record(1, "abc", 0.2, 10.0, False, detail=detail)

        fps = tracker.get_recent_fingerprints(limit=5)
        assert len(fps) == 0


class TestPersistentErrors:
    """FitnessTracker.get_recent_error_signatures tracks recurring bugs."""

    def test_returns_errors_in_multiple_generations(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        sig = "attributeerror: 'list' has no attribute 'get'"
        for i in range(3):
            detail = {"error_signatures": [sig]}
            tracker.record(i, f"h{i}", 0.8, 60.0, True, detail=detail)

        errors = tracker.get_recent_error_signatures(limit=5)
        assert len(errors) == 1
        assert sig in errors[0]

    def test_ignores_one_off_errors(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        # Each generation has a different error — not persistent.
        for i in range(3):
            detail = {"error_signatures": [f"error_{i}"]}
            tracker.record(i, f"h{i}", 0.8, 60.0, True, detail=detail)

        errors = tracker.get_recent_error_signatures(limit=5)
        assert len(errors) == 0

    def test_empty_db_returns_empty(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_recent_error_signatures() == []


class TestLLMUsage:
    """FitnessTracker LLM token usage tracking."""

    def test_record_and_retrieve(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        rid = tracker.record_llm_usage(1, "evolution", 500, 200)
        assert rid == 1
        rows = tracker.get_llm_usage(limit=10)
        assert len(rows) == 1
        assert rows[0]["caller"] == "evolution"
        assert rows[0]["input_tokens"] == 500
        assert rows[0]["output_tokens"] == 200

    def test_get_llm_usage_ordering(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record_llm_usage(1, "evolution", 100, 50)
        tracker.record_llm_usage(2, "crystallization", 200, 75)
        rows = tracker.get_llm_usage(limit=10)
        assert rows[0]["caller"] == "crystallization"  # most recent first
        assert rows[1]["caller"] == "evolution"

    def test_get_llm_usage_respects_limit(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(10):
            tracker.record_llm_usage(i, "evolution", 100, 50)
        rows = tracker.get_llm_usage(limit=3)
        assert len(rows) == 3

    def test_summary_totals(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record_llm_usage(1, "evolution", 500, 200)
        tracker.record_llm_usage(2, "crystallization", 300, 100)
        tracker.record_llm_usage(3, "evolution", 400, 150)
        summary = tracker.get_llm_usage_summary()
        assert summary["total_input"] == 1200
        assert summary["total_output"] == 450
        assert summary["total_calls"] == 3

    def test_summary_by_caller(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record_llm_usage(1, "evolution", 500, 200)
        tracker.record_llm_usage(2, "crystallization", 300, 100)
        tracker.record_llm_usage(3, "evolution", 400, 150)
        summary = tracker.get_llm_usage_summary()
        by_caller = summary["by_caller"]
        assert by_caller["evolution"]["input_tokens"] == 900
        assert by_caller["evolution"]["output_tokens"] == 350
        assert by_caller["evolution"]["calls"] == 2
        assert by_caller["crystallization"]["calls"] == 1

    def test_summary_empty_db(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        summary = tracker.get_llm_usage_summary()
        assert summary["total_input"] == 0
        assert summary["total_output"] == 0
        assert summary["total_calls"] == 0
        assert summary["by_caller"] == {}

    def test_migration_creates_table(self, tmp_path):
        """Creating FitnessTracker should also create the llm_usage table."""
        db = tmp_path / "fitness.db"
        tracker = FitnessTracker(db)
        # Should not raise — table exists.
        tracker.record_llm_usage(1, "test", 100, 50)
        assert tracker.get_llm_usage(limit=1)[0]["caller"] == "test"


class TestPlateau:
    """FitnessTracker.is_plateaued detects stagnant scores."""

    def test_plateaued_when_scores_close(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(5):
            tracker.record(i, f"h{i}", 0.80 + i * 0.005, 60.0, True)
        # Range: 0.80 to 0.82 = 0.02, within epsilon=0.03
        assert tracker.is_plateaued(window=5, epsilon=0.03) is True

    def test_not_plateaued_when_scores_vary(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        scores = [0.60, 0.70, 0.80, 0.75, 0.85]
        for i, s in enumerate(scores):
            tracker.record(i, f"h{i}", s, 60.0, True)
        # Range: 0.60 to 0.85 = 0.25, exceeds epsilon=0.03
        assert tracker.is_plateaued(window=5, epsilon=0.03) is False

    def test_not_enough_data(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "h1", 0.80, 60.0, True)
        assert tracker.is_plateaued(window=5) is False

    def test_only_counts_survived(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        # 3 survived (not enough for window=5) + 2 failed (ignored).
        for i in range(3):
            tracker.record(i, f"s{i}", 0.80, 60.0, True)
        for i in range(2):
            tracker.record(i + 3, f"f{i}", 0.10, 10.0, False)
        assert tracker.is_plateaued(window=5) is False


class TestRealErrorDetection:
    """_is_real_error_line distinguishes actual errors from reported errors."""

    def test_traceback_header(self):
        assert _is_real_error_line("Traceback (most recent call last):")

    def test_traceback_frame(self):
        assert _is_real_error_line('  File "main.py", line 10, in <module>')

    def test_python_exception(self):
        assert _is_real_error_line("TypeError: foo() missing 1 required argument")
        assert _is_real_error_line("ValueError: invalid literal")
        assert _is_real_error_line("FileNotFoundError: [Errno 2] No such file")
        assert _is_real_error_line("ZeroDivisionError: division by zero")

    def test_generic_exception(self):
        assert _is_real_error_line("RuntimeException: something went wrong")

    def test_log_analyzer_error_count_not_matched(self):
        """Lines reporting error counts should NOT be flagged."""
        assert not _is_real_error_line("cycle 53: 52 new lines, 15 errors, 1 anomalies")
        assert not _is_real_error_line("error rate: 100.00%")
        assert not _is_real_error_line("recent errors (2):")
        assert not _is_real_error_line("error_count = 5")

    def test_json_error_key_not_matched(self):
        """JSON data mentioning 'error' should NOT be flagged."""
        assert not _is_real_error_line('{"severity_counts": {"error": 15, "info": 37}}')
        assert not _is_real_error_line('"error": 2')

    def test_report_text_not_matched(self):
        """Analytical/report text should NOT be flagged."""
        assert not _is_real_error_line("[critical] high_error_rate: 100%")
        assert not _is_real_error_line("Another error occurred")
        assert not _is_real_error_line("No errors found in the log")
        assert not _is_real_error_line("error_penalty: 0.03")

    def test_normal_output_not_matched(self):
        assert not _is_real_error_line("result_42: 3.14")
        assert not _is_real_error_line("status: running")
        assert not _is_real_error_line("")


class TestScoreTaskAlignment:
    """score_task_alignment with optional topic_keywords."""

    def test_basic_category_match(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        output = ["coding analysis running"]
        cats = {"coding": 5.0}
        score = tracker.score_task_alignment(output, cats)
        assert score > 0.0

    def test_empty_output_returns_zero(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        score = tracker.score_task_alignment([], {"coding": 5.0})
        assert score == 0.0

    def test_empty_categories_returns_zero(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        score = tracker.score_task_alignment(["hello world"], {})
        assert score == 0.0

    def test_topic_keywords_boost_score(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        output = ["stock price analysis report for portfolio"]
        cats = {"finance": 5.0}
        # Without topic keywords — category name "finance" doesn't appear in output.
        score_no_topics = tracker.score_task_alignment(output, cats)
        # With topic keywords — "stock", "price", "portfolio" match.
        score_with_topics = tracker.score_task_alignment(
            output, cats, topic_keywords=["stock", "price", "portfolio"],
        )
        assert score_with_topics > score_no_topics

    def test_topic_keywords_only_no_category_match(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        output = ["python debugging session complete"]
        cats = {"coding": 5.0}
        score = tracker.score_task_alignment(
            output, cats, topic_keywords=["python", "debugging"],
        )
        assert score > 0.0

    def test_max_score_capped_at_015(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        output = ["coding coding coding coding coding"]
        cats = {"coding": 100.0}
        score = tracker.score_task_alignment(
            output, cats, topic_keywords=["coding"],
        )
        assert score <= 0.15

    def test_none_topic_keywords_works(self, tmp_path):
        """Passing None for topic_keywords should behave like no keywords."""
        tracker = FitnessTracker(tmp_path / "fitness.db")
        output = ["hello world"]
        cats = {"general": 1.0}
        score = tracker.score_task_alignment(output, cats, topic_keywords=None)
        assert isinstance(score, float)


class TestErrorPenaltyFalsePositive:
    """evaluate_output should not penalize programs that report on errors."""

    def test_log_analyzer_output_no_penalty(self):
        """Log analyzer output mentioning 'error' should have zero penalty."""
        lines = [
            "[22:26:01] cycle 53: 52 new lines, 15 errors, 1 anomalies",
            'json_summary: {"cycle": 53, "severity_counts": {"error": 15, "info": 37}}',
            "[22:26:11] cycle 54: 2 new lines, 2 errors, 1 anomalies",
            "[critical] high_error_rate: error rate: 100.00% (2 errors in 2 lines)",
            "recent errors (2):",
            "normal analysis output line",
        ] * 10  # 60 lines total
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["error_penalty"] == 0.0
        assert detail["error_lines"] == 0

    def test_real_traceback_still_penalized(self):
        """Actual Python tracebacks should still be penalized."""
        lines = [
            "Traceback (most recent call last):",
            '  File "main.py", line 10, in <module>',
            "    foo()",
            "TypeError: foo() missing 1 required argument",
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["error_penalty"] > 0.0
        assert detail["error_lines"] == 3  # traceback header + frame + TypeError

    def test_mixed_real_and_reported_errors(self):
        """Only real errors should count, not reported ones."""
        lines = [
            "Analyzing log file...",
            "Found 15 errors in log",
            "error rate: 50%",
            "ValueError: invalid config",  # only this is a real error
            "Analysis complete",
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["error_lines"] == 1
