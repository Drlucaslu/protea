"""Proactive thinking loop — generates morning briefings, evening summaries,
and periodic proactive checks for user-facing push notifications.

Triggered by the Sentinel main loop at configured intervals.
Sends notifications via the existing notifier.send() mechanism.

Pure stdlib — no external dependencies (LLM calls via ring1.llm_base).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from ring1.llm_base import LLMClient, LLMError

log = logging.getLogger("protea.proactive_loop")

_MORNING_SYSTEM_PROMPT_BASE = """\
You are Protea, a personal AI assistant generating a morning briefing.
Based on the user's interest profile, recent task history, and preferences,
create a concise, personalized morning briefing.

Format (use emojis, keep it under 1500 chars):

☀️ 早安，今日简报：

📚 基于兴趣（list top 3 interest areas with %）：
• 1-3 personalized recommendations based on their interests

📋 昨日回顾：
• Summary of yesterday's tasks (count + highlights)
• Any new preferences discovered

🔮 今日建议：
• 1-2 actionable suggestions based on patterns

If there's not enough data for a section, skip it gracefully.
Keep the tone warm and helpful. Mix Chinese and English naturally.
"""

_EVENING_SYSTEM_PROMPT_BASE = """\
You are Protea, a personal AI assistant generating an evening summary.
Summarize today's interactions, insights discovered, and preview tomorrow.

Format (use emojis, keep it under 1000 chars):

🌙 今日总结：

📊 今日工作：
• Task count and categories
• Notable accomplishments

💡 今日发现：
• Any new preference or behavior patterns observed
• Cross-domain connections noticed

📝 待办提醒：
• Any deferred/pending items from today

Keep it brief and actionable.
"""

_PERIODIC_SYSTEM_PROMPT_BASE = """\
You are Protea, deciding whether to proactively notify the user.
Based on the current context (time, recent activity, preferences),
decide if there's something worth proactively sharing right now.

Rules:
- Only notify if genuinely useful (not just to fill silence)
- Max 1 notification per check cycle
- Good reasons: found relevant info, detected useful pattern, deadline reminder
- Bad reasons: generic tips, obvious observations, already-known info

Respond in exactly this format:
ACTION: none | notify
REASON: brief explanation
CONTENT: (if notify) the notification text (max 500 chars, use emojis)
"""


def _morning_prompt() -> str:
    from ring1.soul import inject
    return inject(_MORNING_SYSTEM_PROMPT_BASE)


def _evening_prompt() -> str:
    from ring1.soul import inject
    return inject(_EVENING_SYSTEM_PROMPT_BASE)


def _periodic_prompt() -> str:
    from ring1.soul import inject
    return inject(_PERIODIC_SYSTEM_PROMPT_BASE)


class ProactiveLoop:
    """Proactive thinking loop — generates periodic user-facing content.

    Designed to be called by the Sentinel main loop at appropriate intervals.
    All methods are best-effort: failures are logged but never propagated.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_store=None,
        preference_store=None,
        user_profiler=None,
        notifier=None,
        config: dict | None = None,
        project_root=None,
        state=None,
    ) -> None:
        self._client = llm_client
        self._memory_store = memory_store
        self._preference_store = preference_store
        self._user_profiler = user_profiler
        self._notifier = notifier
        self._project_root = project_root
        self._state = state

        cfg = config or {}
        self.morning_hour = cfg.get("morning_hour", 9)
        self.evening_hour = cfg.get("evening_hour", 21)
        self.check_interval_sec = cfg.get("check_interval_sec", 1800)
        self.max_proactive_actions_per_day = cfg.get("max_proactive_actions_per_day", 5)

        self._last_morning_date: str = ""
        self._last_evening_date: str = ""
        self._last_periodic_check: float = 0.0
        self._daily_action_count: int = 0
        self._daily_action_date: str = ""

    def _reset_daily_counter(self) -> None:
        """Reset the daily action counter if the date has changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._daily_action_date:
            self._daily_action_count = 0
            self._daily_action_date = today

    def _can_act(self) -> bool:
        """Check if we can perform another proactive action today."""
        self._reset_daily_counter()
        return self._daily_action_count < self.max_proactive_actions_per_day

    def _record_action(self) -> None:
        """Record that a proactive action was taken."""
        self._reset_daily_counter()
        self._daily_action_count += 1

    def _build_context(self, include_tasks: bool = True) -> str:
        """Build context string from available data sources."""
        parts = []

        # User profile (keyword-based).
        if self._user_profiler:
            try:
                summary = self._user_profiler.get_profile_summary()
                if summary:
                    parts.append(f"## User Profile\n{summary}")
            except Exception:
                pass

        # Structured preferences.
        if self._preference_store:
            try:
                pref_summary = self._preference_store.get_preference_summary_text()
                if pref_summary:
                    parts.append(f"## Structured Preferences\n{pref_summary}")
            except Exception:
                pass

        # Recent tasks.
        if include_tasks and self._memory_store:
            try:
                tasks = self._memory_store.get_by_type("task", limit=10)
                if tasks:
                    parts.append("## Recent Tasks")
                    for t in tasks:
                        content = t.get("content", "")[:100]
                        ts = t.get("timestamp", "")
                        parts.append(f"- [{ts[:10]}] {content}")
            except Exception:
                pass

        # Current time context.
        now = datetime.now()
        parts.append(f"\n## Current Time: {now.strftime('%Y-%m-%d %H:%M %A')}")

        return "\n".join(parts)

    def _check_onboarding(self) -> str | None:
        """Ask the next onboarding question if profile is incomplete.

        Returns the question text if sent, None otherwise.
        """
        if not self._state or not self._notifier:
            return None
        # Don't interrupt if there's already a pending question.
        if getattr(self._state, '_pending_soul_question', None):
            return None
        try:
            from ring1 import soul
            if not soul.is_section_empty("Owner"):
                return None  # Owner section already has content
            if not soul.should_ask_today():
                return None
            nq = soul.get_next_question()
            if not nq:
                return None
            field, question = nq
            self._send(question)
            soul.record_asked()
            self._state._pending_soul_question = {"field": field}
            log.info("Soul onboarding: asked '%s'", field)
            return question
        except Exception:
            log.debug("Onboarding check failed", exc_info=True)
            return None

    def check_and_send(self) -> str | None:
        """Main entry point — check time and decide what to send.

        Called by the Sentinel main loop. Returns the content that was sent
        (or None if nothing was sent).
        """
        # Soul onboarding (max 1 question/day, before other proactive content).
        onboarding = self._check_onboarding()
        if onboarding:
            return onboarding

        if not self._can_act():
            return None

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Morning briefing check.
        if (now.hour == self.morning_hour
                and self._last_morning_date != today):
            content = self.morning_briefing()
            if content:
                self._last_morning_date = today
                self._record_action()
                self._send(content)
                return content

        # Evening summary check.
        if (now.hour == self.evening_hour
                and self._last_evening_date != today):
            content = self.evening_summary()
            if content:
                self._last_evening_date = today
                self._record_action()
                self._send(content)
                return content

        # Periodic check.
        if time.time() - self._last_periodic_check >= self.check_interval_sec:
            self._last_periodic_check = time.time()
            result = self.periodic_check()
            if result and result.get("action") == "notify":
                content = result.get("content", "")
                if content:
                    self._record_action()
                    self._send(content)
                    return content

        return None

    def morning_briefing(self) -> str | None:
        """Generate a morning briefing.

        Data sources: user_profile, memory (yesterday's tasks), preferences.
        Returns the briefing text or None on failure.
        """
        context = self._build_context(include_tasks=True)
        user_msg = f"{context}\n\nGenerate a morning briefing for today."

        try:
            response = self._client.send_message(
                _morning_prompt(), user_msg,
            )
            return response.strip() if response.strip() else None
        except LLMError as exc:
            log.debug("Morning briefing LLM call failed: %s", exc)
            return None

    def evening_summary(self) -> str | None:
        """Generate an evening summary.

        Data sources: memory (today's tasks), preferences (new discoveries).
        Returns the summary text or None on failure.
        """
        context = self._build_context(include_tasks=True)
        user_msg = f"{context}\n\nGenerate an evening summary for today."

        try:
            response = self._client.send_message(
                _evening_prompt(), user_msg,
            )
            return response.strip() if response.strip() else None
        except LLMError as exc:
            log.debug("Evening summary LLM call failed: %s", exc)
            return None

    def periodic_check(self) -> dict | None:
        """Periodic proactive thinking: decide if user should be notified.

        Returns {"action": "none"|"notify", "content": "..."} or None on failure.
        """
        context = self._build_context(include_tasks=True)
        user_msg = f"{context}\n\nShould we notify the user about anything right now?"

        try:
            response = self._client.send_message(
                _periodic_prompt(), user_msg,
            )
        except LLMError as exc:
            log.debug("Periodic check LLM call failed: %s", exc)
            return None

        return self._parse_periodic_response(response)

    @staticmethod
    def _parse_periodic_response(response: str) -> dict | None:
        """Parse the periodic check LLM response."""
        result = {"action": "none", "content": ""}
        for line in response.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().lower()
                if action in ("none", "notify"):
                    result["action"] = action
            elif line.upper().startswith("CONTENT:"):
                result["content"] = line.split(":", 1)[1].strip()
        return result

    def _send(self, content: str) -> bool:
        """Send notification via the configured notifier."""
        if not self._notifier:
            log.debug("ProactiveLoop: no notifier configured, skipping send")
            return False
        try:
            return self._notifier.send(content)
        except Exception:
            log.debug("ProactiveLoop: notification send failed", exc_info=True)
            return False
