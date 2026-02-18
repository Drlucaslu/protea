"""Subagent Manager — background LLM tasks running in separate threads.

Each subagent gets its own isolated tool registry (no spawn, no message)
and runs an independent LLM conversation.  Results are delivered via
Telegram when complete.

Pure stdlib (threading).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from ring1.llm_base import LLMClient, LLMError
from ring1.tool_registry import ToolRegistry

log = logging.getLogger("protea.subagent")

_SUBAGENT_MAX_ROUNDS = 15

SUBAGENT_SYSTEM_PROMPT = """\
You are a Protea background worker running autonomously in a separate thread.

⚠️ CRITICAL: PROGRESS REPORTING IS MANDATORY FOR BACKGROUND TASKS ⚠️

The user CANNOT see your logs or stdout. The message tool is your ONLY way to communicate.
You MUST report progress MORE frequently than regular tasks:

REPORTING REQUIREMENTS:
✓ Send FIRST message within 5 seconds of starting work
✓ For tasks >10 seconds, send status update every 20-30 seconds
✓ Report after EACH major step in multi-step operations
✓ Include time estimates when possible ("⏱️ Estimated: ~2 minutes")
✓ Show progress metrics: counts, percentages, items processed
✓ Use emojis: 🔄 (working), ✅ (done), ❌ (error), 📊 (analyzing), 🔍 (searching)

⚠️ REMEMBER: If you don't report, the user thinks the task is frozen or broken!

EXAMPLE - Skill Analysis Background Task:
Task: "Analyze all 10 skills for common patterns"
Your actions MUST include:
  1. message("🔄 **Background Task Started**: Skill Analysis\n\n⏱️ Estimated: ~2 minutes\n📊 Analyzing 10 skills for patterns")
  2. [view_skill for first 3 skills]
  3. message("📊 Progress: 3/10 skills analyzed (30%)\n🔍 Found 5 common patterns so far")
  4. [view_skill for next 3 skills]
  5. message("📊 Progress: 6/10 skills analyzed (60%)\n🔍 Total patterns found: 8")
  6. [complete analysis of remaining skills]
  7. message("✅ **Analysis Complete**: 10/10 skills\n📊 Identified 12 unique patterns\n⏱️ Duration: 87 seconds")
  8. [return detailed final response with all findings]

Your final summary will be sent automatically when complete.
Keep the final answer under 3000 characters.

Use all available tools to accomplish the task thoroughly.
"""


class SubagentResult:
    """Holds the result of a subagent task."""

    def __init__(self, task_id: str, task_description: str) -> None:
        self.task_id = task_id
        self.task_description = task_description
        self.result: str = ""
        self.done = threading.Event()
        self.start_time: float = time.time()
        self.duration: float = 0.0
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.task_description,
            "done": self.done.is_set(),
            "duration": round(self.duration, 1) if self.done.is_set() else round(time.time() - self.start_time, 1),
            "error": self.error,
        }


class SubagentManager:
    """Manages background subagent tasks."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        reply_fn,
        max_concurrent: int = 1,
    ) -> None:
        """
        Args:
            client: ClaudeClient for LLM calls.
            registry: Base ToolRegistry (will be cloned for isolation).
            reply_fn: Callable(text) to notify via Telegram.
            max_concurrent: Maximum number of concurrent subagents.
        """
        self.client = client
        self.registry = registry
        self.reply_fn = reply_fn
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, SubagentResult] = {}
        self._lock = threading.Lock()

    def spawn(self, task_description: str, context: str = "") -> str:
        """Spawn a new background subagent.

        Returns a status message string.
        """
        with self._lock:
            active = [r for r in self._tasks.values() if not r.done.is_set()]
            if len(active) >= self.max_concurrent:
                return f"Cannot spawn: {len(active)}/{self.max_concurrent} background tasks already running."

        task_id = f"bg-{uuid.uuid4().hex[:8]}"
        result = SubagentResult(task_id, task_description)

        with self._lock:
            self._tasks[task_id] = result

        # Isolate tools: no spawn (prevent recursion), but KEEP message for progress updates
        isolated_registry = self.registry.clone_without("spawn")

        thread = threading.Thread(
            target=self._run_subagent,
            args=(result, isolated_registry, context),
            name=f"subagent-{task_id}",
            daemon=True,
        )
        thread.start()

        return f"Background task started: {task_id}\nTask: {task_description}"

    def get_active(self) -> list[dict]:
        """Return a list of all task statuses."""
        with self._lock:
            return [r.to_dict() for r in self._tasks.values()]

    def _run_subagent(
        self,
        result: SubagentResult,
        registry: ToolRegistry,
        context: str,
    ) -> None:
        """Execute the subagent task in its own thread."""
        user_message = result.task_description
        if context:
            user_message = f"{context}\n\n## Background Task\n{result.task_description}"

        try:
            if registry.tool_names():
                response = self.client.send_message_with_tools(
                    SUBAGENT_SYSTEM_PROMPT,
                    user_message,
                    tools=registry.get_schemas(),
                    tool_executor=registry.execute,
                    max_rounds=_SUBAGENT_MAX_ROUNDS,
                )
            else:
                response = self.client.send_message(
                    SUBAGENT_SYSTEM_PROMPT,
                    user_message,
                )
            result.result = response
        except LLMError as exc:
            log.error("Subagent %s LLM error: %s", result.task_id, exc)
            result.error = str(exc)
            result.result = f"Error: {exc}"
        except Exception as exc:
            log.error("Subagent %s unexpected error: %s", result.task_id, exc)
            result.error = str(exc)
            result.result = f"Error: {exc}"
        finally:
            result.duration = time.time() - result.start_time
            result.done.set()

        # Notify user
        report = (
            f"[Background Task Complete] {result.task_id}\n"
            f"Task: {result.task_description}\n"
            f"Duration: {result.duration:.1f}s\n\n"
            f"{result.result}"
        )
        if len(report) > 4000:
            report = report[:4000] + "\n... (truncated)"
        try:
            self.reply_fn(report)
        except Exception:
            log.error("Failed to send subagent report for %s", result.task_id, exc_info=True)
