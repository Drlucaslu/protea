"""Tests for ring0.evolution_intent."""

import pytest

from ring0.evolution_intent import classify_intent, compute_blast_radius


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntent:
    """Priority: directive > crashed > persistent_errors > plateau > default."""

    def test_directive_returns_adapt(self):
        result = classify_intent(
            survived=True,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[],
            directive="make a snake game",
        )
        assert result["intent"] == "adapt"
        assert any("directive:" in s for s in result["signals"])

    def test_directive_truncated_at_80_chars(self):
        long_directive = "x" * 200
        result = classify_intent(
            survived=True,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[],
            directive=long_directive,
        )
        assert result["intent"] == "adapt"
        # Signal text should contain at most 80 chars of directive.
        sig = result["signals"][0]
        assert len(sig) <= len("directive: ") + 80

    def test_directive_overrides_crash(self):
        """Directive takes priority even when code crashed."""
        result = classify_intent(
            survived=False,
            is_plateaued=True,
            persistent_errors=["SomeError"],
            crash_logs=[{"content": "TypeError: bad"}],
            directive="build a calculator",
        )
        assert result["intent"] == "adapt"

    def test_crashed_returns_repair(self):
        result = classify_intent(
            survived=False,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "repair"
        assert "crashed" in result["signals"]

    def test_crashed_with_crash_logs_extracts_errors(self):
        result = classify_intent(
            survived=False,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[
                {"content": "Traceback:\n  File main.py\nTypeError: bad arg"},
            ],
            directive="",
        )
        assert result["intent"] == "repair"
        assert "TypeError" in result["signals"]

    def test_crashed_with_persistent_errors(self):
        result = classify_intent(
            survived=False,
            is_plateaued=False,
            persistent_errors=["attributeerror: 'list' has no attr"],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "repair"
        assert any("attributeerror" in s for s in result["signals"])

    def test_persistent_errors_survived_returns_repair(self):
        """Survived but with persistent errors -> repair."""
        result = classify_intent(
            survived=True,
            is_plateaued=False,
            persistent_errors=["TimeoutError in line 45"],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "repair"
        assert any("TimeoutError" in s for s in result["signals"])

    def test_plateaued_returns_explore(self):
        result = classify_intent(
            survived=True,
            is_plateaued=True,
            persistent_errors=[],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "explore"
        assert "plateau" in result["signals"]

    def test_default_returns_optimize(self):
        result = classify_intent(
            survived=True,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "optimize"
        assert "survived" in result["signals"]

    def test_persistent_errors_override_plateau(self):
        """Persistent errors take priority over plateau."""
        result = classify_intent(
            survived=True,
            is_plateaued=True,
            persistent_errors=["KeyError: 'foo'"],
            crash_logs=[],
            directive="",
        )
        assert result["intent"] == "repair"

    def test_crash_logs_dedup_error_names(self):
        """Same error class from multiple logs should appear once."""
        result = classify_intent(
            survived=False,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[
                {"content": "TypeError: bad\nTypeError: also bad"},
                {"content": "TypeError: yet another"},
            ],
            directive="",
        )
        assert result["signals"].count("TypeError") == 1

    def test_multiple_error_classes_extracted(self):
        result = classify_intent(
            survived=False,
            is_plateaued=False,
            persistent_errors=[],
            crash_logs=[
                {"content": "ValueError: x\nKeyError: y\nRuntimeError: z"},
            ],
            directive="",
        )
        assert result["intent"] == "repair"
        assert "ValueError" in result["signals"]
        assert "KeyError" in result["signals"]
        assert "RuntimeError" in result["signals"]


# ---------------------------------------------------------------------------
# compute_blast_radius
# ---------------------------------------------------------------------------


class TestComputeBlastRadius:
    def test_identical_sources(self):
        source = "line1\nline2\nline3\n"
        result = compute_blast_radius(source, source)
        assert result["lines_changed"] == 0
        assert result["lines_added"] == 0
        assert result["lines_removed"] == 0
        assert result["scope"] == "minor"

    def test_minor_change(self):
        old = "\n".join(f"line {i}" for i in range(100)) + "\n"
        # Change 1 line.
        lines = [f"line {i}" for i in range(100)]
        lines[50] = "CHANGED line 50"
        new = "\n".join(lines) + "\n"
        result = compute_blast_radius(old, new)
        assert result["lines_added"] >= 1
        assert result["lines_removed"] >= 1
        assert result["scope"] == "minor"

    def test_moderate_change(self):
        old = "\n".join(f"line {i}" for i in range(20)) + "\n"
        # Change 3 lines out of 20 -> ratio ~ 0.3 (6/20)
        lines = [f"line {i}" for i in range(20)]
        lines[5] = "CHANGED"
        lines[10] = "CHANGED"
        lines[15] = "CHANGED"
        new = "\n".join(lines) + "\n"
        result = compute_blast_radius(old, new)
        assert result["scope"] in ("moderate", "major")

    def test_full_rewrite(self):
        old = "line1\nline2\nline3\n"
        new = "completely\ndifferent\ncontent\nwith\nextra\nlines\n"
        result = compute_blast_radius(old, new)
        assert result["scope"] == "full_rewrite"

    def test_empty_old_source(self):
        result = compute_blast_radius("", "line1\nline2\n")
        assert result["lines_added"] == 2
        assert result["lines_removed"] == 0
        assert result["scope"] == "full_rewrite"

    def test_empty_new_source(self):
        result = compute_blast_radius("line1\nline2\n", "")
        assert result["lines_removed"] == 2
        assert result["lines_added"] == 0
        assert result["scope"] == "full_rewrite"

    def test_both_empty(self):
        result = compute_blast_radius("", "")
        assert result["lines_changed"] == 0
        assert result["scope"] == "minor"

    def test_major_change(self):
        old = "\n".join(f"line {i}" for i in range(10)) + "\n"
        # Replace 5 out of 10 lines -> ratio ~1.0 (10/10)
        lines = [f"line {i}" for i in range(10)]
        for i in range(5):
            lines[i] = f"NEW {i}"
        new = "\n".join(lines) + "\n"
        result = compute_blast_radius(old, new)
        assert result["scope"] in ("major", "full_rewrite")

    def test_lines_added_only(self):
        old = "line1\n"
        new = "line1\nline2\nline3\n"
        result = compute_blast_radius(old, new)
        assert result["lines_added"] == 2
        assert result["lines_removed"] == 0
