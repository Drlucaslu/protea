"""Prompt templates for Ring 1.

Memory curation prompts. Evolution and crystallization prompts have been
removed — see ring1/reflector.py for the reflection system prompts.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Memory Curation
# ---------------------------------------------------------------------------

MEMORY_CURATION_SYSTEM_PROMPT = """\
You are the memory curator for Protea, a self-evolving AI system.
Your task: review memory entries and decide which to keep, discard, summarize, extract_rule, or conflict.

## Decision criteria
- keep: Unique insights, user preferences, important lessons, recurring patterns
- summarize: Valuable but verbose — condense to 1-2 sentences
- discard: Redundant, outdated, trivial, or superseded by newer memories
- extract_rule: When you see 2+ related entries forming a pattern, distill into a reusable rule/preference. Return: {"id": [1, 2, 3], "action": "extract_rule", "rule": "Concise rule text"}
- conflict: Two entries contain contradictory information. Include "conflict_with": <other_entry_id>. Use when the same topic has conflicting claims (e.g., "user prefers A" vs "user prefers B").

## Response format
Respond with a JSON array (no markdown fences):
[{"id": 1, "action": "keep"}, {"id": 2, "action": "summarize", "summary": "..."}, ...]
"""


def build_memory_curation_prompt(candidates: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_message) for memory curation.

    Args:
        candidates: List of dicts with id, entry_type/type, content, importance.

    Returns:
        (system_prompt, user_message) tuple.
    """
    parts = ["## Memory Entries to Review\n"]
    for c in candidates:
        entry_id = c.get("id", "?")
        entry_type = c.get("entry_type", c.get("type", "unknown"))
        content = c.get("content", "")
        importance = c.get("importance", 0.5)
        # Truncate long content.
        if len(content) > 200:
            content = content[:200] + "..."
        parts.append(
            f"- **ID {entry_id}** [{entry_type}] (importance: {importance:.2f}): {content}"
        )
    parts.append("")
    parts.append(f"Total: {len(candidates)} entries. Review each and decide: keep, discard, summarize, extract_rule, or conflict.")

    return MEMORY_CURATION_SYSTEM_PROMPT, "\n".join(parts)
