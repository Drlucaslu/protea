"""Tests for ring0.cron — cron expression parser."""

from __future__ import annotations

from datetime import datetime

import pytest

from ring0.cron import _parse_field, next_run, describe


# ---------------------------------------------------------------------------
# _parse_field
# ---------------------------------------------------------------------------

class TestParseField:
    def test_star(self):
        assert _parse_field("*", 0, 59) == set(range(0, 60))

    def test_exact_value(self):
        assert _parse_field("5", 0, 59) == {5}

    def test_range(self):
        assert _parse_field("1-5", 0, 59) == {1, 2, 3, 4, 5}

    def test_step(self):
        assert _parse_field("*/15", 0, 59) == {0, 15, 30, 45}

    def test_list(self):
        assert _parse_field("1,3,5", 0, 59) == {1, 3, 5}

    def test_range_with_step(self):
        assert _parse_field("0-30/10", 0, 59) == {0, 10, 20, 30}

    def test_mixed_list(self):
        assert _parse_field("1,10-12,30", 0, 59) == {1, 10, 11, 12, 30}

    def test_clamps_to_bounds(self):
        # Values outside lo-hi are excluded
        assert _parse_field("0", 1, 31) == set()

    def test_weekday_range(self):
        assert _parse_field("1-5", 0, 6) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# next_run
# ---------------------------------------------------------------------------

class TestNextRun:
    def test_every_minute(self):
        after = datetime(2026, 2, 19, 10, 30, 0)
        result = next_run("* * * * *", after)
        assert result == datetime(2026, 2, 19, 10, 31, 0)

    def test_specific_time(self):
        after = datetime(2026, 2, 19, 8, 0, 0)
        result = next_run("30 9 * * *", after)
        assert result == datetime(2026, 2, 19, 9, 30, 0)

    def test_past_time_rolls_to_next_day(self):
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("30 9 * * *", after)
        assert result == datetime(2026, 2, 20, 9, 30, 0)

    def test_every_15_minutes(self):
        after = datetime(2026, 2, 19, 10, 14, 0)
        result = next_run("*/15 * * * *", after)
        assert result == datetime(2026, 2, 19, 10, 15, 0)

    def test_weekday_monday(self):
        # 2026-02-19 is Thursday, next Monday is 2026-02-23
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("0 9 * * 1", after)  # Mon=1 in cron
        assert result == datetime(2026, 2, 23, 9, 0, 0)
        assert result.weekday() == 0  # Python Monday

    def test_specific_day_of_month(self):
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("0 8 1 * *", after)
        assert result == datetime(2026, 3, 1, 8, 0, 0)

    def test_specific_month(self):
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("0 0 1 6 *", after)
        assert result == datetime(2026, 6, 1, 0, 0, 0)

    def test_hourly(self):
        after = datetime(2026, 2, 19, 10, 45, 0)
        result = next_run("0 * * * *", after)
        assert result == datetime(2026, 2, 19, 11, 0, 0)

    def test_invalid_field_count_raises(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            next_run("* * *")

    def test_seconds_are_zeroed(self):
        after = datetime(2026, 2, 19, 10, 30, 45)
        result = next_run("* * * * *", after)
        assert result.second == 0
        assert result.microsecond == 0

    def test_strictly_after(self):
        # Even if after is exactly on a match, advance to next
        after = datetime(2026, 2, 19, 9, 30, 0)
        result = next_run("30 9 * * *", after)
        assert result == datetime(2026, 2, 20, 9, 30, 0)

    def test_none_after_uses_now(self):
        result = next_run("* * * * *")
        assert result > datetime.now()

    def test_weekend_only(self):
        # 2026-02-19 is Thursday; next Sat is 2026-02-21
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("0 12 * * 0,6", after)  # Sun=0, Sat=6
        assert result == datetime(2026, 2, 21, 12, 0, 0)
        assert result.weekday() == 5  # Saturday

    def test_every_5_hours(self):
        after = datetime(2026, 2, 19, 3, 0, 0)
        result = next_run("0 */5 * * *", after)
        assert result == datetime(2026, 2, 19, 5, 0, 0)

    def test_list_hours(self):
        after = datetime(2026, 2, 19, 10, 0, 0)
        result = next_run("0 8,12,18 * * *", after)
        assert result == datetime(2026, 2, 19, 12, 0, 0)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_every_minute(self):
        assert describe("* * * * *") == "every minute"

    def test_daily_time(self):
        desc = describe("30 9 * * *")
        assert "09:30" in desc

    def test_step_minutes(self):
        desc = describe("*/15 * * * *")
        assert "15" in desc and "min" in desc

    def test_weekday_mon_fri(self):
        desc = describe("0 8 * * 1-5")
        assert "Mon-Fri" in desc

    def test_specific_days(self):
        desc = describe("0 0 1,15 * *")
        assert "1" in desc and "15" in desc

    def test_invalid_returns_raw(self):
        assert describe("bad") == "bad"

    def test_hourly(self):
        desc = describe("30 * * * *")
        assert "hour" in desc or "30" in desc

    def test_step_hours(self):
        desc = describe("0 */2 * * *")
        assert "2" in desc and "hour" in desc
