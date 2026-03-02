"""LLM-based implicit preference extraction from user task interactions.

Analyzes task text to extract implicit preferences (communication style,
topic interests, behavior patterns) and stores them as preference moments
in the PreferenceStore.

Cost controls:
- Skip short texts (< 15 characters)
- Rate limit: one extraction per extraction_rate_limit_sec (default 300s)
- Only extract from substantive tasks

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import logging
import time

from ring1.llm_base import LLMClient, LLMError

log = logging.getLogger("protea.preference_extractor")

_SYSTEM_PROMPT_BASE = """\
You are a preference extractor. Given a user's task request and the response,
identify implicit preferences and behavioral signals.

Focus on:
1. Communication style (language preference, formality, detail level)
2. Domain interests (what topics they care about)
3. Work patterns (time-sensitive, iterative, exploratory)
4. Tool preferences (prefers code, prefers explanations, prefers files)

Output format — ONE line per signal, max 3 signals:
CATEGORY: signal description

Categories: communication, interest, workflow, tool_preference, behavior

Example input: "帮我用python分析这个csv，重点看revenue趋势"
Example output:
communication: prefers Chinese for technical discussions
interest: data analysis and revenue tracking
tool_preference: prefers Python for data processing

If no meaningful preference can be extracted, output: NONE
"""


def _build_system_prompt() -> str:
    """Build system prompt with soul rule constraints injected."""
    try:
        from ring1.soul import get_rules
        rules = get_rules()
    except Exception:
        rules = []
    base = _SYSTEM_PROMPT_BASE
    if rules:
        constraint = "\n\nIMPORTANT: The following are FIXED user rules. Do NOT extract signals that contradict them:\n"
        for r in rules:
            constraint += f"- {r}\n"
        base += constraint
    return base

# Minimum text length to attempt extraction.
_MIN_TEXT_LENGTH = 15


class PreferenceExtractor:
    """Extract implicit preferences from task interactions via LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        preference_store=None,
        rate_limit_sec: int = 300,
    ) -> None:
        self._client = llm_client
        self._preference_store = preference_store
        self._rate_limit_sec = rate_limit_sec
        self._last_extraction_time: float = 0.0

    def extract_and_store(
        self,
        task_text: str,
        response_text: str = "",
        category_hint: str = "",
    ) -> int:
        """Extract preferences from a task and store as moments.

        Args:
            task_text: The user's task text.
            response_text: The assistant's response (optional context).
            category_hint: Category from UserProfiler (optional).

        Returns:
            Number of moments stored (0 if skipped or failed).
        """
        # Cost control: skip short texts.
        if len(task_text.strip()) < _MIN_TEXT_LENGTH:
            return 0

        # Cost control: rate limiting.
        now = time.time()
        if now - self._last_extraction_time < self._rate_limit_sec:
            return 0

        self._last_extraction_time = now

        if not self._preference_store:
            return 0

        # Build user message.
        user_msg = f"Task: {task_text[:500]}"
        if response_text:
            user_msg += f"\nResponse summary: {response_text[:200]}"

        try:
            result = self._client.send_message(_build_system_prompt(), user_msg)
        except LLMError as exc:
            log.debug("Preference extraction LLM call failed: %s", exc)
            return 0

        return self._parse_and_store(result, category_hint)

    def _parse_and_store(self, response: str, category_hint: str) -> int:
        """Parse LLM response and store as preference moments."""
        response = response.strip()
        if not response or response.upper() == "NONE":
            return 0

        stored = 0
        for line in response.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue

            # Parse "CATEGORY: signal" format.
            if ":" not in line:
                continue

            parts = line.split(":", 1)
            moment_type = parts[0].strip().lower()
            signal = parts[1].strip()

            if not signal or len(signal) < 5:
                continue

            # Map to categories.
            category = category_hint or self._map_moment_category(moment_type)

            try:
                self._preference_store.store_moment(
                    moment_type=moment_type,
                    content=signal[:200],
                    category=category,
                    extracted_signal=signal[:100],
                )
                stored += 1
            except Exception:
                log.debug("Failed to store preference moment", exc_info=True)

            if stored >= 3:
                break

        if stored:
            log.info("Extracted %d preference moments", stored)

        return stored

    @staticmethod
    def _map_moment_category(moment_type: str) -> str:
        """Map moment type to UserProfiler category taxonomy."""
        mapping = {
            "communication": "lifestyle",
            "interest": "lifestyle",
            "workflow": "system",
            "tool_preference": "coding",
            "behavior": "lifestyle",
        }
        return mapping.get(moment_type, "lifestyle")
