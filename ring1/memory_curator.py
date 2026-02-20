"""LLM-assisted memory curation.

Reviews candidate memory entries and decides which to keep, discard, or
summarize.  Used during the warm→cold compaction phase.
Nightly consolidation: cross-task correlation and behavior insight extraction.
"""

from __future__ import annotations

import json
import logging
import re

from ring1.llm_base import LLMClient, LLMError
from ring1.prompts import build_memory_curation_prompt

log = logging.getLogger("protea.memory_curator")

_CONSOLIDATION_SYSTEM_PROMPT = """\
You are the nightly memory consolidation engine for Protea, a personal AI assistant.
Analyze the user's recent tasks and discover hidden patterns.

Your job:
1. Cross-task correlation: find connections between seemingly unrelated tasks
2. Behavior insights: detect work patterns, time preferences, recurring themes
3. Interest evolution: note which topics are growing or declining

Output format — JSON array of insight objects:
[
  {
    "type": "correlation" | "behavior" | "interest_shift",
    "content": "Concise insight (1-2 sentences)",
    "confidence": 0.0-1.0,
    "related_tasks": [task indices that support this insight]
  }
]

Rules:
- Max 5 insights per consolidation
- Only report genuinely useful patterns (not obvious observations)
- confidence >= 0.6 to be worth reporting
- If no meaningful patterns found, return []
"""


class MemoryCurator:
    """Curate memory entries using an LLM for the warm→cold transition."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def curate(self, candidates: list[dict]) -> list[dict]:
        """Review candidate memories and return curation decisions.

        Args:
            candidates: List of dicts with keys: id, entry_type (aliased as type),
                        content, importance.

        Returns:
            List of dicts with keys: id, action (keep|discard|summarize),
            and optional summary.  Empty list on failure (caller should
            fall back to rule-based curation).
        """
        if not candidates:
            return []

        system_prompt, user_message = build_memory_curation_prompt(candidates)

        try:
            response = self._client.send_message(system_prompt, user_message)
        except LLMError as exc:
            log.warning("Memory curation LLM call failed: %s", exc)
            return []

        return self._parse_response(response, candidates)

    @staticmethod
    def _parse_response(response: str, candidates: list[dict]) -> list[dict]:
        """Parse the LLM JSON response into curation decisions."""
        text = response.strip()

        # Strip markdown code fences if present.
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        try:
            decisions = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            log.warning("Failed to parse curation response as JSON")
            return []

        if not isinstance(decisions, list):
            return []

        valid_ids = {c["id"] for c in candidates}
        valid_actions = {"keep", "discard", "summarize", "extract_rule"}

        result = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            action = d.get("action", "keep")
            if action not in valid_actions:
                continue

            if action == "extract_rule":
                # id can be a list of ints or a single int.
                raw_id = d.get("id")
                if isinstance(raw_id, int):
                    ids = [raw_id]
                elif isinstance(raw_id, list):
                    ids = [i for i in raw_id if isinstance(i, int)]
                else:
                    continue
                # All ids must be valid.
                if not ids or not all(i in valid_ids for i in ids):
                    continue
                rule = d.get("rule")
                if not rule or not isinstance(rule, str) or not rule.strip():
                    continue
                result.append({"id": ids, "action": "extract_rule", "rule": rule.strip()})
            else:
                entry_id = d.get("id")
                if entry_id not in valid_ids:
                    continue
                entry = {"id": entry_id, "action": action}
                if action == "summarize" and d.get("summary"):
                    entry["summary"] = d["summary"]
                result.append(entry)

        return result

    def nightly_consolidate(
        self,
        memory_store,
        preference_store=None,
    ) -> dict:
        """Nightly deep consolidation — discover cross-task patterns and insights.

        Runs once per night (caller controls scheduling). Analyzes recent tasks
        to find correlations, behavior patterns, and interest shifts.
        Stores discovered insights as preference_moments.

        Args:
            memory_store: MemoryStore to query recent tasks from.
            preference_store: Optional PreferenceStore to write insights into.

        Returns:
            Dict with counts: insights_found, moments_stored.
        """
        result = {"insights_found": 0, "moments_stored": 0}

        # Get tasks from the last 7 days.
        try:
            recent_tasks = memory_store.get_by_type("task", limit=30)
        except Exception:
            log.debug("Nightly consolidation: failed to get tasks", exc_info=True)
            return result

        if len(recent_tasks) < 3:
            return result  # Not enough data for meaningful patterns.

        # Build the user message with task summaries.
        parts = ["## Recent Tasks (last 7 days)\n"]
        for i, task in enumerate(recent_tasks):
            content = task.get("content", "")[:150]
            ts = task.get("timestamp", "")[:16]
            parts.append(f"{i}. [{ts}] {content}")

        user_message = "\n".join(parts)

        try:
            response = self._client.send_message(
                _CONSOLIDATION_SYSTEM_PROMPT, user_message,
            )
        except LLMError as exc:
            log.debug("Nightly consolidation LLM call failed: %s", exc)
            return result

        # Parse response.
        insights = self._parse_consolidation_response(response)
        result["insights_found"] = len(insights)

        # Store insights as preference moments.
        if preference_store and insights:
            for insight in insights:
                insight_type = insight.get("type", "behavior")
                content = insight.get("content", "")
                confidence = insight.get("confidence", 0.5)

                if not content or confidence < 0.6:
                    continue

                category_map = {
                    "correlation": "research",
                    "behavior": "lifestyle",
                    "interest_shift": "lifestyle",
                }

                try:
                    preference_store.store_moment(
                        moment_type=insight_type,
                        content=content[:200],
                        category=category_map.get(insight_type, "general"),
                        extracted_signal=content[:100],
                    )
                    result["moments_stored"] += 1
                except Exception:
                    log.debug("Failed to store consolidation insight", exc_info=True)

        if result["moments_stored"]:
            log.info(
                "Nightly consolidation: %d insights found, %d moments stored",
                result["insights_found"], result["moments_stored"],
            )

        return result

    @staticmethod
    def _parse_consolidation_response(response: str) -> list[dict]:
        """Parse the JSON response from nightly consolidation."""
        text = response.strip()
        # Strip markdown code fences if present.
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            log.debug("Failed to parse consolidation response as JSON")
            return []

        if not isinstance(data, list):
            return []

        valid_types = {"correlation", "behavior", "interest_shift"}
        results = []
        for item in data[:5]:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in valid_types:
                continue
            if not item.get("content"):
                continue
            results.append(item)

        return results
