"""Nudge Engine — contextual suggestions after tasks + proactive engagement.

Generates LLM-driven nudges:
  - schedule: suggest scheduling a recurring-looking query
  - expand: offer to elaborate or create a detailed plan
  - tip: teach the user about unused Protea features
  - proactive: weather, news, follow-ups during active usage
"""

from __future__ import annotations

import hashlib
import logging
import time

from ring1.llm_base import LLMClient

log = logging.getLogger("protea.nudge")

# All known Protea features for tip nudges.
_PROTEA_FEATURES = [
    "/schedule — 定期任务",
    "/calendar — 日历管理",
    "/background — 后台任务",
    "/status — 系统状态",
    "进化指令 — 用 /evolve 引导进化方向",
    "技能系统 — Protea 自动学习可复用技能",
    "记忆系统 — Protea 记住你的偏好和历史",
    "多轮对话 — 直接回复 Protea 消息可继续对话",
]

_POST_TASK_SYSTEM = """\
You are Protea's nudge engine.  After a user task completes, decide whether
to offer a contextual suggestion.

Available nudge types:
- schedule: the task looks recurring — suggest creating a scheduled task
- expand: the response was brief — offer to elaborate or create a plan
- tip: teach the user about a Protea feature they haven't used
- proactive: suggest a useful follow-up action

Rules:
- Only nudge when genuinely useful.  If the task is simple acknowledgement
  or the response is already comprehensive, output ACTION: none.
- Keep the nudge text SHORT (1-2 sentences, Chinese).
- BUTTON_YES and BUTTON_NO should be short Chinese labels (2-4 chars).
- If type is "schedule", include TASK: with the recurring task description.
- If type is "expand", include TASK: with what to expand on.
- If type is "proactive", include TASK: with the follow-up task.

Output format (exactly):
ACTION: none | nudge
TYPE: schedule | expand | tip | proactive
TEXT: <nudge text in Chinese>
BUTTON_YES: <accept button label>
BUTTON_NO: <dismiss button label>
TASK: <optional task text for execute/schedule actions>
"""

_PROACTIVE_SYSTEM = """\
You are Protea's proactive engagement engine.  During active usage periods,
suggest something useful to the user.

Nudge types:
- tip: teach about an unused Protea feature
- proactive: weather check, news summary, follow-up on recent tasks

Rules:
- Only suggest when genuinely useful.  Output ACTION: none if nothing is timely.
- Keep text SHORT (1-2 sentences, Chinese).
- Consider the time of day and recent activity.

Output format (exactly):
ACTION: none | nudge
TYPE: tip | proactive
TEXT: <nudge text in Chinese>
BUTTON_YES: <accept button label>
BUTTON_NO: <dismiss button label>
TASK: <task to execute if user accepts>
"""


def _task_hash(text: str) -> str:
    """Short hash for callback_data (Telegram limit: 64 bytes)."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


class NudgeEngine:
    """Generate contextual nudges after tasks and proactive engagement."""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_store,
        scheduled_store,
        user_profiler,
        preference_store,
        config: dict,
    ) -> None:
        self._client = llm_client
        self._memory_store = memory_store
        self._scheduled_store = scheduled_store
        self._user_profiler = user_profiler
        self._preference_store = preference_store
        # Config
        self._interval_sec = config.get("interval_sec", 600)
        self._max_daily = config.get("max_daily_nudges", 20)
        # State
        self._last_nudge_time: float = 0.0
        self._daily_count: int = 0
        self._daily_date: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post_task_nudge(
        self,
        task_text: str,
        response: str,
        context: dict,
    ) -> tuple[str, list, dict] | None:
        """Generate a nudge after task completion.

        Returns (text, buttons, nudge_meta) or None.
        nudge_meta contains keys like task_hash, suggested_task for callback handling.
        """
        if not self._can_nudge():
            return None

        user_msg = self._build_post_task_message(task_text, response, context)
        try:
            llm_response = self._client.send_message(
                _POST_TASK_SYSTEM, user_msg,
            )
        except Exception:
            log.debug("Nudge LLM call failed", exc_info=True)
            return None

        return self._parse_nudge_response(llm_response, task_text)

    def proactive_nudge(self) -> tuple[str, list, dict] | None:
        """Generate a proactive nudge during active usage.

        Returns (text, buttons, nudge_meta) or None.
        """
        if not self._can_nudge():
            return None

        user_msg = self._build_proactive_message()
        try:
            llm_response = self._client.send_message(
                _PROACTIVE_SYSTEM, user_msg,
            )
        except Exception:
            log.debug("Proactive nudge LLM call failed", exc_info=True)
            return None

        return self._parse_nudge_response(llm_response, "proactive")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _can_nudge(self) -> bool:
        """Check daily cap and minimum interval."""
        now = time.time()
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_count = 0
        if self._daily_count >= self._max_daily:
            return False
        if now - self._last_nudge_time < self._interval_sec:
            return False
        return True

    def _record_nudge(self) -> None:
        """Record that a nudge was sent."""
        self._last_nudge_time = time.time()
        self._daily_count += 1

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_post_task_message(
        self, task_text: str, response: str, context: dict,
    ) -> str:
        parts: list[str] = []
        parts.append(f"Task: {task_text[:200]}")
        parts.append(f"Response summary: {response[:300]}")

        # User profile
        if self._user_profiler:
            try:
                summary = self._user_profiler.get_profile_summary()
                if summary:
                    parts.append(f"User profile: {summary[:200]}")
            except Exception:
                pass

        # Recent task history
        if self._memory_store:
            try:
                recent = self._memory_store.get_by_type("task", limit=5)
                if recent:
                    history = "; ".join(
                        t.get("content", "")[:60] for t in recent
                    )
                    parts.append(f"Recent tasks: {history}")
            except Exception:
                pass

        # Feature usage hints
        parts.append(f"Available features: {', '.join(_PROTEA_FEATURES)}")

        # Time context
        parts.append(f"Current time: {time.strftime('%Y-%m-%d %H:%M %A')}")

        return "\n".join(parts)

    def _build_proactive_message(self) -> str:
        parts: list[str] = []
        parts.append(f"Time: {time.strftime('%Y-%m-%d %H:%M %A')}")

        if self._user_profiler:
            try:
                summary = self._user_profiler.get_profile_summary()
                if summary:
                    parts.append(f"User profile: {summary[:200]}")
            except Exception:
                pass

        if self._memory_store:
            try:
                recent = self._memory_store.get_by_type("task", limit=5)
                if recent:
                    history = "; ".join(
                        t.get("content", "")[:60] for t in recent
                    )
                    parts.append(f"Recent tasks: {history}")
            except Exception:
                pass

        parts.append(f"Available features: {', '.join(_PROTEA_FEATURES)}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_nudge_response(
        self, response: str, source_text: str,
    ) -> tuple[str, list, dict] | None:
        """Parse LLM response into (text, buttons, nudge_meta) or None."""
        lines = response.strip().splitlines()
        fields: dict[str, str] = {}
        for line in lines:
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip().upper()] = val.strip()

        action = fields.get("ACTION", "none").lower()
        if action != "nudge":
            return None

        text = fields.get("TEXT", "")
        if not text:
            return None

        nudge_type = fields.get("TYPE", "tip").lower()
        btn_yes = fields.get("BUTTON_YES", "好的")
        btn_no = fields.get("BUTTON_NO", "不了")
        suggested_task = fields.get("TASK", "")

        h = _task_hash(source_text)

        # Determine callback action based on nudge type.
        if nudge_type == "schedule":
            yes_callback = f"nudge:schedule:{h}"
        elif nudge_type == "expand":
            yes_callback = f"nudge:expand:{h}"
        else:
            yes_callback = f"nudge:execute:{h}"

        buttons = [[
            {"text": btn_yes, "callback_data": yes_callback},
            {"text": btn_no, "callback_data": "nudge:dismiss"},
        ]]

        nudge_meta = {
            "task_hash": h,
            "nudge_type": nudge_type,
            "source_text": source_text[:500],
            "suggested_task": suggested_task,
            "created_at": time.time(),
        }

        self._record_nudge()
        return (text, buttons, nudge_meta)
