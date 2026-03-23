"""Trajectory saving and outcome labeling for distillation.

Saves complete LLM interaction trajectories (system prompt, messages,
tool calls, tool results, responses) as JSONL files for downstream
fine-tuning / distillation.

Each trajectory is labeled with an outcome (success/failure) based on
multiple signals: fabrication detection, quality score, tool errors, etc.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Any

log = logging.getLogger("protea.trajectory")

# Default directory for trajectory storage
_DEFAULT_DIR = pathlib.Path("trajectories")

# Outcome labels
OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_PARTIAL = "partial"


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def determine_outcome(
    fab_signals: list[str],
    tool_sequence: list[str],
    response: str,
    duration: float,
    usage: dict,
) -> tuple[str, list[str]]:
    """Determine the outcome label for a task trajectory.

    Returns (outcome, reasons) where reasons explain the classification.
    """
    reasons: list[str] = []

    # Fabrication = definite failure
    if fab_signals:
        reasons.append(f"fabrication: {', '.join(fab_signals)}")
        return OUTCOME_FAILURE, reasons

    # Empty response = failure
    if not response or not response.strip():
        reasons.append("empty_response")
        return OUTCOME_FAILURE, reasons

    # Apology / failure messages in response
    failure_markers = [
        "没有完成", "无法完成", "无法处理", "出现了错误",
        "couldn't process", "failed to", "I ran out of tool-call budget",
    ]
    for marker in failure_markers:
        if marker in response:
            reasons.append(f"failure_marker: {marker}")
            return OUTCOME_FAILURE, reasons

    # Excessive duration (>5 min for a single task is suspicious)
    if duration > 300:
        reasons.append(f"long_duration: {duration:.0f}s")

    # Very high token usage (>50K input) might indicate context issues
    input_tokens = usage.get("input_tokens", 0)
    if input_tokens > 50000:
        reasons.append(f"high_tokens: {input_tokens}")

    # Tools used but no data tools = might be shallow
    data_tools = {"web_fetch", "web_search", "exec", "run_skill", "read_file"}
    if tool_sequence and not set(tool_sequence) & data_tools:
        # Only used non-data tools (message, write_file, etc.) — could be partial
        if len(tool_sequence) == 1 and tool_sequence[0] == "message":
            pass  # Single message tool is fine
        else:
            reasons.append("no_data_tools")

    if reasons:
        return OUTCOME_PARTIAL, reasons

    return OUTCOME_SUCCESS, ["clean"]


def save_trajectory(
    task_id: str,
    task_text: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    response: str,
    tool_sequence: list[str],
    skills_used: list[str],
    outcome: str,
    outcome_reasons: list[str],
    duration: float,
    usage: dict,
    trajectory_dir: pathlib.Path | None = None,
) -> str | None:
    """Save a complete task trajectory as a JSONL entry.

    Returns the path to the trajectory file, or None on failure.
    """
    traj_dir = trajectory_dir or _DEFAULT_DIR
    _ensure_dir(traj_dir)

    # Convert Claude API messages to a serializable format
    serializable_messages = _serialize_messages(messages)

    # Build the trajectory entry
    entry = {
        "id": task_id,
        "timestamp": time.time(),
        "task": task_text[:500],
        "outcome": outcome,
        "outcome_reasons": outcome_reasons,
        "system_prompt": system_prompt,
        "messages": serializable_messages,
        "tools": tools,
        "response": response[:2000],
        "tool_sequence": tool_sequence,
        "skills_used": skills_used,
        "duration_sec": round(duration, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }

    # Write to date-partitioned file
    date_str = time.strftime("%Y-%m-%d")
    filepath = traj_dir / f"{date_str}.jsonl"

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        log.info(
            "Trajectory saved: task=%s outcome=%s tools=%d file=%s",
            task_id, outcome, len(tool_sequence), filepath.name,
        )
        return str(filepath)
    except Exception:
        log.error("Failed to save trajectory", exc_info=True)
        return None


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Convert Claude API message format to a clean serializable format.

    Handles content blocks (text, tool_use, tool_result) gracefully.
    """
    result = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Content blocks — flatten to a simpler format
            entry: dict[str, Any] = {"role": role}
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    })
                elif btype == "tool_result":
                    # Truncate long tool results for storage
                    content_val = block.get("content", "")
                    if isinstance(content_val, str) and len(content_val) > 1000:
                        content_val = content_val[:1000] + "...(truncated)"
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": content_val,
                    })

            if text_parts:
                entry["content"] = "\n".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if tool_results:
                entry["tool_results"] = tool_results

            result.append(entry)
        else:
            result.append({"role": role, "content": str(content)})

    return result


def load_trajectories(
    trajectory_dir: pathlib.Path | None = None,
    outcome_filter: str | None = None,
    days: int = 30,
) -> list[dict]:
    """Load trajectories from JSONL files.

    Args:
        trajectory_dir: Directory containing trajectory files.
        outcome_filter: Filter by outcome (success/failure/partial).
        days: Only load files from the last N days.

    Returns:
        List of trajectory entries.
    """
    traj_dir = trajectory_dir or _DEFAULT_DIR
    if not traj_dir.exists():
        return []

    cutoff = time.time() - (days * 86400)
    entries = []

    for filepath in sorted(traj_dir.glob("*.jsonl")):
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("timestamp", 0) < cutoff:
                        continue
                    if outcome_filter and entry.get("outcome") != outcome_filter:
                        continue
                    entries.append(entry)
        except Exception:
            log.warning("Failed to read trajectory file: %s", filepath, exc_info=True)

    return entries


def get_trajectory_stats(trajectory_dir: pathlib.Path | None = None, days: int = 7) -> dict:
    """Get summary statistics of saved trajectories."""
    entries = load_trajectories(trajectory_dir, days=days)

    if not entries:
        return {"total": 0, "success": 0, "failure": 0, "partial": 0}

    stats = {
        "total": len(entries),
        "success": sum(1 for e in entries if e.get("outcome") == OUTCOME_SUCCESS),
        "failure": sum(1 for e in entries if e.get("outcome") == OUTCOME_FAILURE),
        "partial": sum(1 for e in entries if e.get("outcome") == OUTCOME_PARTIAL),
        "total_tokens": sum(
            e.get("input_tokens", 0) + e.get("output_tokens", 0) for e in entries
        ),
        "avg_tools_per_task": (
            sum(len(e.get("tool_sequence", [])) for e in entries) / len(entries)
        ),
    }

    # Tool frequency
    tool_freq: dict[str, int] = {}
    for e in entries:
        for t in e.get("tool_sequence", []):
            tool_freq[t] = tool_freq.get(t, 0) + 1
    stats["tool_frequency"] = dict(
        sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    return stats
