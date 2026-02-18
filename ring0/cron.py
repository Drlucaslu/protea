"""Cron expression parser — pure stdlib, no external dependencies.

Supports standard 5-field cron (minute hour day month weekday):
  - ``*``       any value
  - ``5``       exact value
  - ``1-5``     range
  - ``*/15``    step
  - ``1,3,5``   list
  - combinations like ``1-5/2`` (range with step)

Public API:
  - ``next_run(cron_expr, after)`` — next matching datetime after *after*
  - ``describe(cron_expr)``        — human-readable description
"""

from __future__ import annotations

from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    """Parse a single cron field into a set of valid integers."""
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if "/" in part:
            base, step_str = part.split("/", 1)
            step = int(step_str)
            if base == "*":
                start, end = lo, hi
            elif "-" in base:
                a, b = base.split("-", 1)
                start, end = int(a), int(b)
            else:
                start, end = int(base), hi
            for v in range(start, end + 1, step):
                if lo <= v <= hi:
                    values.add(v)
        elif "-" in part:
            a, b = part.split("-", 1)
            for v in range(int(a), int(b) + 1):
                if lo <= v <= hi:
                    values.add(v)
        elif part == "*":
            values.update(range(lo, hi + 1))
        else:
            v = int(part)
            if lo <= v <= hi:
                values.add(v)
    return values


def _parse_cron(cron_expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Parse a 5-field cron expression into (minutes, hours, days, months, weekdays)."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields in cron expression, got {len(fields)}: {cron_expr!r}")
    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    days = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    weekdays = _parse_field(fields[4], 0, 6)  # 0=Sunday
    return minutes, hours, days, months, weekdays


# ---------------------------------------------------------------------------
# Core: next_run
# ---------------------------------------------------------------------------

def next_run(cron_expr: str, after: datetime | None = None) -> datetime:
    """Compute the next datetime matching *cron_expr*, strictly after *after*.

    Raises ``ValueError`` if no match is found within ~4 years (safety limit).
    """
    if after is None:
        after = datetime.now()

    minutes, hours, days, months, weekdays = _parse_cron(cron_expr)

    # Start from the next minute
    dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Safety: cap iterations (~4 years of minutes)
    max_iterations = 366 * 24 * 60 * 4
    for _ in range(max_iterations):
        if dt.month not in months:
            # Advance to next valid month
            dt = dt.replace(day=1, hour=0, minute=0) + timedelta(days=32)
            dt = dt.replace(day=1, hour=0, minute=0)
            continue
        if dt.day not in days:
            dt = dt.replace(hour=0, minute=0) + timedelta(days=1)
            continue
        if dt.weekday() != 6:
            cron_dow = (dt.weekday() + 1) % 7  # Python: Mon=0; cron: Sun=0
        else:
            cron_dow = 0
        if cron_dow not in weekdays:
            dt = dt.replace(hour=0, minute=0) + timedelta(days=1)
            continue
        if dt.hour not in hours:
            dt = dt.replace(minute=0) + timedelta(hours=1)
            continue
        if dt.minute not in minutes:
            dt += timedelta(minutes=1)
            continue
        return dt

    raise ValueError(f"No matching time found within 4 years for: {cron_expr!r}")


# ---------------------------------------------------------------------------
# Human-readable description
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
_WEEKDAY_NAMES_ZH = {0: "日", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}


def describe(cron_expr: str) -> str:
    """Return a human-readable description of *cron_expr*."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return cron_expr

    f_min, f_hour, f_day, f_month, f_dow = fields

    # Every minute
    if all(f == "*" for f in fields):
        return "every minute"

    parts: list[str] = []

    # Time
    if f_min != "*" and f_hour != "*" and f_min.isdigit() and f_hour.isdigit():
        parts.append(f"{int(f_hour):02d}:{int(f_min):02d}")
    elif f_min.startswith("*/"):
        parts.append(f"every {f_min[2:]} min")
    elif f_hour.startswith("*/"):
        parts.append(f"every {f_hour[2:]} hours")
        if f_min.isdigit():
            parts[-1] += f" at :{int(f_min):02d}"
    elif f_hour != "*" and f_min.isdigit():
        hours = _parse_field(f_hour, 0, 23)
        parts.append(f"at :{int(f_min):02d} on hours {','.join(str(h) for h in sorted(hours))}")
    elif f_hour == "*" and f_min.isdigit():
        parts.append(f"every hour at :{int(f_min):02d}")

    # Weekday
    if f_dow != "*":
        weekdays = _parse_field(f_dow, 0, 6)
        if weekdays == {1, 2, 3, 4, 5}:
            parts.append("Mon-Fri")
        elif weekdays == {0, 6}:
            parts.append("weekends")
        else:
            names = [_WEEKDAY_NAMES.get(d, str(d)) for d in sorted(weekdays)]
            parts.append(",".join(names))

    # Day of month
    if f_day != "*":
        days = _parse_field(f_day, 1, 31)
        if len(days) <= 5:
            parts.append(f"day {','.join(str(d) for d in sorted(days))}")
        else:
            parts.append(f"days {min(days)}-{max(days)}")

    # Month
    if f_month != "*":
        months = _parse_field(f_month, 1, 12)
        parts.append(f"month {','.join(str(m) for m in sorted(months))}")

    return " ".join(parts) if parts else cron_expr
