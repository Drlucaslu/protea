"""Trajectory compressor — prepares saved trajectories for distillation.

Compresses trajectories by:
1. Shrinking system prompts (remove Soul Profile verbosity, Ring2 code)
2. Truncating long tool results
3. Keeping only the essential interaction pattern
4. Converting to OpenAI fine-tuning format (messages array)

Output: JSONL files ready for LoRA fine-tuning via louter's distill pipeline.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import time

from ring1.trajectory import load_trajectories, OUTCOME_SUCCESS, OUTCOME_PARTIAL

log = logging.getLogger("protea.trajectory_compressor")

# Max chars for compressed system prompt
_MAX_SYSTEM_CHARS = 600
# Max chars per tool result in compressed trajectory
_MAX_TOOL_RESULT_CHARS = 500
# Max chars for user message
_MAX_USER_MSG_CHARS = 1500

# Patterns to strip from system prompts
_SOUL_VERBOSE_SECTIONS = re.compile(
    r"## (Cognitive Style|Work Context|Communication|Autonomy|Response Style)"
    r"[\s\S]*?(?=\n## |\Z)",
    re.MULTILINE,
)
_RING2_CODE_BLOCK = re.compile(
    r"## Ring 2 Code\s*```python[\s\S]*?```",
    re.MULTILINE,
)
_PROTEA_STATE_BLOCK = re.compile(
    r"## Protea State[\s\S]*?(?=\n## |\Z)",
    re.MULTILINE,
)


def compress_system_prompt(prompt: str) -> str:
    """Compress a system prompt for distillation.

    Keeps identity and core instructions, removes verbose sections
    like Soul Profile details and Ring2 code.
    """
    compressed = prompt

    # Remove Ring2 code block (often 3-5K chars)
    compressed = _RING2_CODE_BLOCK.sub(
        "## Ring 2 Code\n[ring2 code omitted for brevity]\n",
        compressed,
    )

    # Remove Protea State block (cycle info, etc.)
    compressed = _PROTEA_STATE_BLOCK.sub("", compressed)

    # Remove verbose Soul Profile sections
    compressed = _SOUL_VERBOSE_SECTIONS.sub("", compressed)

    # Collapse whitespace
    compressed = re.sub(r"\n{3,}", "\n\n", compressed).strip()

    # Final truncation if still too long
    if len(compressed) > _MAX_SYSTEM_CHARS:
        compressed = compressed[:_MAX_SYSTEM_CHARS] + "\n...(compressed)"

    return compressed


def compress_message(msg: dict) -> dict | None:
    """Compress a single message for training data.

    Returns None if the message should be skipped entirely.
    """
    role = msg.get("role", "")
    content = msg.get("content", "")

    if role == "user":
        # Truncate very long user messages but keep tool results
        if msg.get("tool_results"):
            # Tool result message — compress individual results
            compressed_results = []
            for tr in msg["tool_results"]:
                result_content = tr.get("content", "")
                if isinstance(result_content, str) and len(result_content) > _MAX_TOOL_RESULT_CHARS:
                    result_content = result_content[:_MAX_TOOL_RESULT_CHARS] + "...(truncated)"
                compressed_results.append({
                    "tool_use_id": tr.get("tool_use_id", ""),
                    "content": result_content,
                })
            return {"role": role, "tool_results": compressed_results}
        elif isinstance(content, str) and len(content) > _MAX_USER_MSG_CHARS:
            return {"role": role, "content": content[:_MAX_USER_MSG_CHARS] + "...(truncated)"}
        return msg

    if role == "assistant":
        # Keep assistant messages as-is (they're the training target)
        return msg

    return msg


def trajectory_to_training_sample(
    entry: dict,
    format: str = "openai",
) -> dict | None:
    """Convert a trajectory entry to a training sample.

    Args:
        entry: A trajectory entry from load_trajectories().
        format: Output format — "openai" or "sharegpt".

    Returns:
        A training sample dict, or None if the entry can't be converted.
    """
    messages = entry.get("messages", [])
    if not messages:
        return None

    # Compress system prompt
    system_prompt = entry.get("system_prompt", "")
    compressed_system = compress_system_prompt(system_prompt)

    # Build compressed message sequence
    compressed_messages: list[dict] = []

    # Add system message
    if compressed_system:
        compressed_messages.append({
            "role": "system",
            "content": compressed_system,
        })

    # Process conversation messages
    for msg in messages:
        compressed = compress_message(msg)
        if compressed:
            compressed_messages.append(compressed)

    if len(compressed_messages) < 2:
        return None  # Need at least system + one exchange

    if format == "sharegpt":
        return _to_sharegpt(compressed_messages, entry)
    else:
        return _to_openai(compressed_messages, entry)


def _to_openai(messages: list[dict], entry: dict) -> dict:
    """Convert to OpenAI fine-tuning format."""
    # Convert tool_calls/tool_results to OpenAI function calling format
    openai_messages = []
    for msg in messages:
        if msg.get("tool_calls"):
            # Assistant message with tool calls
            openai_msg: dict = {"role": "assistant", "content": msg.get("content", "")}
            openai_tool_calls = []
            for tc in msg["tool_calls"]:
                openai_tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("input", {}), ensure_ascii=False),
                    },
                })
            openai_msg["tool_calls"] = openai_tool_calls
            openai_messages.append(openai_msg)
        elif msg.get("tool_results"):
            # Tool result messages — one per result
            for tr in msg["tool_results"]:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": tr.get("content", ""),
                })
        else:
            openai_messages.append(msg)

    result: dict = {"messages": openai_messages}

    # Include tools if present
    tools = entry.get("tools", [])
    if tools:
        # Convert Claude tool format to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        result["tools"] = openai_tools

    return result


def _to_sharegpt(messages: list[dict], entry: dict) -> dict:
    """Convert to ShareGPT format."""
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}
    conversations = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        from_role = role_map.get(role, role)

        if msg.get("tool_calls"):
            # Include tool calls in content
            tc_text = json.dumps(msg["tool_calls"], ensure_ascii=False, indent=2)
            content = f"{content}\n\n[Tool Calls]\n{tc_text}" if content else f"[Tool Calls]\n{tc_text}"
        elif msg.get("tool_results"):
            # Tool results
            for tr in msg["tool_results"]:
                conversations.append({
                    "from": "tool",
                    "value": tr.get("content", ""),
                })
            continue

        conversations.append({"from": from_role, "value": content or ""})

    return {"conversations": conversations}


def compress_and_export(
    output_path: str | pathlib.Path,
    trajectory_dir: pathlib.Path | None = None,
    format: str = "openai",
    outcome_filter: str | None = None,
    days: int = 30,
    max_samples: int = 0,
) -> int:
    """Load trajectories, compress, and export as training data.

    Args:
        output_path: Path to write the JSONL output file.
        trajectory_dir: Directory containing trajectory JSONL files.
        format: Output format ("openai" or "sharegpt").
        outcome_filter: Filter by outcome (default: success + partial).
        days: Load trajectories from last N days.
        max_samples: Maximum samples to export (0 = all).

    Returns:
        Number of samples exported.
    """
    # Load trajectories — default to success + partial
    entries = []
    if outcome_filter:
        entries = load_trajectories(trajectory_dir, outcome_filter=outcome_filter, days=days)
    else:
        entries = load_trajectories(trajectory_dir, outcome_filter=OUTCOME_SUCCESS, days=days)
        entries += load_trajectories(trajectory_dir, outcome_filter=OUTCOME_PARTIAL, days=days)

    if not entries:
        log.info("No trajectories found for export")
        return 0

    # Sort by timestamp
    entries.sort(key=lambda e: e.get("timestamp", 0))

    if max_samples > 0:
        entries = entries[:max_samples]

    # Convert and write
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            sample = trajectory_to_training_sample(entry, format=format)
            if sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

    log.info("Exported %d training samples to %s (format=%s)", count, output_path, format)
    return count
