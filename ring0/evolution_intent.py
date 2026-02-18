"""Evolution intent classification and blast radius computation.

Pure-function module — classifies why an evolution was triggered and
measures how much code changed.
"""

from __future__ import annotations

import difflib
import re

_ERROR_PATTERN = re.compile(r"\b(\w*Error|\w*Exception)\b")


def classify_intent(
    survived: bool,
    is_plateaued: bool,
    persistent_errors: list[str],
    crash_logs: list[dict],
    directive: str,
) -> dict:
    """Classify evolution intent from available signals.

    Returns {"intent": str, "signals": list[str]}

    Classification rules (priority order):
    1. directive non-empty -> intent="adapt"
    2. not survived -> intent="repair"
    3. persistent_errors non-empty (even if survived) -> intent="repair"
    4. is_plateaued -> intent="explore"
    5. default (survived, no issues) -> intent="optimize"
    """
    signals: list[str] = []

    # 1. Directive overrides everything.
    if directive:
        signals.append(f"directive: {directive[:80]}")
        return {"intent": "adapt", "signals": signals}

    # 2. Crashed — repair.
    if not survived:
        signals.extend(_extract_error_signals(crash_logs))
        for err in persistent_errors[:3]:
            signals.append(err[:120])
        if not signals:
            signals.append("crashed")
        return {"intent": "repair", "signals": signals}

    # 3. Persistent errors even though survived.
    if persistent_errors:
        for err in persistent_errors[:3]:
            signals.append(err[:120])
        return {"intent": "repair", "signals": signals}

    # 4. Plateau.
    if is_plateaued:
        signals.append("plateau")
        return {"intent": "explore", "signals": signals}

    # 5. Default — survived, no issues.
    signals.append("survived")
    return {"intent": "optimize", "signals": signals}


def _extract_error_signals(crash_logs: list[dict]) -> list[str]:
    """Extract error class names from crash log content."""
    seen: set[str] = set()
    signals: list[str] = []
    for log_entry in crash_logs[:3]:
        content = log_entry.get("content", "")
        for match in _ERROR_PATTERN.finditer(content):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                signals.append(name)
    return signals


def compute_blast_radius(old_source: str, new_source: str) -> dict:
    """Compute scope of code changes via line diff.

    Returns {"lines_changed": int, "lines_added": int,
             "lines_removed": int, "scope": str}
    """
    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, n=0))

    lines_added = 0
    lines_removed = 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    lines_changed = lines_added + lines_removed
    total_lines = max(len(old_lines), len(new_lines), 1)
    ratio = lines_changed / total_lines

    if ratio > 0.7:
        scope = "full_rewrite"
    elif ratio > 0.3:
        scope = "major"
    elif ratio > 0.1:
        scope = "moderate"
    else:
        scope = "minor"

    return {
        "lines_changed": lines_changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "scope": scope,
    }
