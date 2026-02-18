"""LLM-assisted memory curation.

Reviews candidate memory entries and decides which to keep, discard, or
summarize.  Used during the warm→cold compaction phase.
"""

from __future__ import annotations

import json
import logging
import re

from ring1.llm_base import LLMClient, LLMError
from ring1.prompts import build_memory_curation_prompt

log = logging.getLogger("protea.memory_curator")


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
        valid_actions = {"keep", "discard", "summarize"}

        result = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            entry_id = d.get("id")
            action = d.get("action", "keep")
            if entry_id not in valid_ids or action not in valid_actions:
                continue
            entry = {"id": entry_id, "action": action}
            if action == "summarize" and d.get("summary"):
                entry["summary"] = d["summary"]
            result.append(entry)

        return result
