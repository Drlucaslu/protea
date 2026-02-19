"""Automatic directive generation from recent task history + user profile.

When evolution scores plateau and no user directive is pending, this module
generates a focused directive by summarizing recent task patterns via LLM.
"""

from __future__ import annotations

import logging

from ring1.llm_base import LLMClient

log = logging.getLogger("protea.directive_generator")

_SYSTEM_PROMPT = """\
You are a strategic advisor for Protea, a self-evolving AI system.
Your task: analyse the user's recent task history and profile to identify
the most impactful direction for the system's next evolution.

Output ONLY a single directive line (no markdown, no explanation):
- Must be < 100 characters
- Must be specific and actionable (not vague like "improve performance")
- Must address a concrete gap between what the user needs and what exists
- Must be phrased as an imperative ("Add ...", "Improve ...", "Build ...")

Example good directives:
- "Add calendar event parsing from natural language input"
- "Build a file search skill that handles Chinese filenames"
- "Improve error recovery when Telegram API times out"

Example bad directives (too vague):
- "Make the system better"
- "Explore new capabilities"
- "Improve performance"
"""

MIN_TASKS = 3
MAX_DIRECTIVE_LEN = 100


class DirectiveGenerator:
    """Generate evolution directives from task history + user profile."""

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client

    def generate(
        self,
        task_history: list[dict],
        user_profile_summary: str = "",
    ) -> str | None:
        """Summarize recent tasks + user profile into an evolution directive.

        Returns a directive string or None if insufficient data or failure.
        """
        if len(task_history) < MIN_TASKS:
            return None

        parts = []
        if user_profile_summary:
            parts.append(f"## User Profile\n{user_profile_summary}\n")

        parts.append("## Recent Tasks (newest first)")
        for task in task_history[:30]:
            content = task.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"- {content}")

        user_message = "\n".join(parts)

        try:
            response = self._client.send_message(_SYSTEM_PROMPT, user_message)
        except Exception as exc:
            log.debug("Directive generation LLM call failed: %s", exc)
            return None

        return self._parse(response)

    @staticmethod
    def _parse(response: str) -> str | None:
        """Extract a valid directive from the LLM response."""
        # Take the first non-empty line.
        for line in response.strip().splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if not line:
                continue
            # Skip markdown headers or meta-text.
            if line.startswith("#") or line.startswith("```"):
                continue
            # Enforce length limit.
            if len(line) > MAX_DIRECTIVE_LEN:
                line = line[:MAX_DIRECTIVE_LEN]
            return line
        return None
