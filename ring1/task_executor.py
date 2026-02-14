"""Task Executor — processes P0 user tasks and P1 autonomous tasks via Claude API.

Runs in a daemon thread.  Pulls tasks from state.task_queue, calls the LLM,
and replies via the bot's _send_reply.  Sets/clears state.p0_active so the
Sentinel can skip evolution while a user task is in flight.

P1 autonomous tasks: When idle for a configurable threshold, the executor
queries the LLM to decide if there's useful proactive work to do based on
the user's task history.

Pure stdlib (threading, queue, logging).
"""

from __future__ import annotations

import json
import logging
import pathlib
import queue
import threading
import time

from ring1.llm_client import ClaudeClient, LLMError
from ring1.web_tools import web_fetch, web_search

log = logging.getLogger("protea.task_executor")

_MAX_REPLY_LEN = 4000  # Telegram message limit safety margin

TASK_SYSTEM_PROMPT = """\
You are Protea, a self-evolving artificial life agent running on a host machine.
You are helpful and concise.  Answer the user's question or perform the requested
analysis.  You have context about your current state (generation, survival, code).
Keep responses under 3500 characters so they fit in a Telegram message.

You have access to web tools:
- web_search: Search the web using DuckDuckGo. Use this when the user asks you to
  research, look up, or find information about something on the internet.
- web_fetch: Fetch and read the content of a specific URL. Use this to read a page
  found via web_search for more detail.

Use these tools when the user's request requires current information from the web.
Do NOT use them for questions you can answer from your training data alone.
"""

# ---------------------------------------------------------------------------
# Web tool definitions for Claude API tool_use
# ---------------------------------------------------------------------------

WEB_TOOLS: list[dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the web via DuckDuckGo.  Returns a list of results, each "
            "with title, url, and snippet.  Use for research or lookup tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and extract its text content.  Use to read a specific "
            "web page for detailed information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 5000).",
                },
            },
            "required": ["url"],
        },
    },
]


def _execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the appropriate web_tools function."""
    if name == "web_search":
        results = web_search(
            query=tool_input["query"],
            max_results=tool_input.get("max_results", 5),
        )
        return json.dumps(results, ensure_ascii=False)
    elif name == "web_fetch":
        return web_fetch(
            url=tool_input["url"],
            max_chars=tool_input.get("max_chars", 5000),
        )
    else:
        return f"Unknown tool: {name}"

P1_SYSTEM_PROMPT = """\
You are Protea, a self-evolving artificial life agent.  Your owner has been
interacting with you through tasks.  Based on the task history and any standing
directives, decide whether there is useful proactive work you can do right now.

Rules:
- Only suggest work that is clearly valuable based on observed patterns.
- Do NOT suggest work if the history is empty or too sparse to infer needs.
- Do NOT repeat tasks that have already been completed recently.
- Keep the task description concise and actionable.

Respond in EXACTLY this format:
## Decision
YES or NO

## Task
(If YES) A concise description of the proactive work to do.
(If NO) Brief reason why not.
"""


def _build_task_context(
    state_snapshot: dict,
    ring2_source: str,
    memories: list[dict] | None = None,
    skills: list[dict] | None = None,
) -> str:
    """Build context string from current Protea state for LLM task calls."""
    parts = ["## Protea State"]
    parts.append(f"Generation: {state_snapshot.get('generation', '?')}")
    parts.append(f"Alive: {state_snapshot.get('alive', '?')}")
    parts.append(f"Paused: {state_snapshot.get('paused', '?')}")
    parts.append(f"Last score: {state_snapshot.get('last_score', '?')}")
    parts.append(f"Last survived: {state_snapshot.get('last_survived', '?')}")
    parts.append("")

    if ring2_source:
        truncated = ring2_source[:2000]
        if len(ring2_source) > 2000:
            truncated += "\n... (truncated)"
        parts.append("## Current Ring 2 Code (first 2000 chars)")
        parts.append("```python")
        parts.append(truncated)
        parts.append("```")

    if memories:
        parts.append("")
        parts.append("## Recent Learnings")
        for mem in memories:
            gen = mem.get("generation", "?")
            content = mem.get("content", "")
            parts.append(f"- [Gen {gen}] {content}")

    if skills:
        parts.append("")
        parts.append("## Available Skills")
        for skill in skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            parts.append(f"- {name}: {desc}")

    return "\n".join(parts)


class TaskExecutor:
    """Processes user tasks from the queue, one at a time."""

    def __init__(
        self,
        state,
        client: ClaudeClient,
        ring2_path: pathlib.Path,
        reply_fn,
        memory_store=None,
        skill_store=None,
        p1_enabled: bool = False,
        p1_idle_threshold_sec: int = 600,
        p1_check_interval_sec: int = 60,
    ) -> None:
        """
        Args:
            state: SentinelState with task_queue, p0_active, p0_event.
            client: ClaudeClient instance for LLM calls.
            ring2_path: Path to ring2 directory (for reading source).
            reply_fn: Callable(text: str) -> None to send Telegram reply.
            memory_store: Optional MemoryStore for experiential memories.
            skill_store: Optional SkillStore for reusable skills.
            p1_enabled: Whether P1 autonomous tasks are enabled.
            p1_idle_threshold_sec: Seconds of idle before triggering P1.
            p1_check_interval_sec: Minimum seconds between P1 checks.
        """
        self.state = state
        self.client = client
        self.ring2_path = ring2_path
        self.reply_fn = reply_fn
        self.memory_store = memory_store
        self.skill_store = skill_store
        self.p1_enabled = p1_enabled
        self.p1_idle_threshold_sec = p1_idle_threshold_sec
        self.p1_check_interval_sec = p1_check_interval_sec
        self._running = True
        self._last_p0_time: float = time.time()
        self._last_p1_check: float = 0.0

    def run(self) -> None:
        """Main loop — blocks on queue, executes tasks serially."""
        log.info("Task executor started")
        while self._running:
            try:
                task = self.state.task_queue.get(timeout=2)
            except queue.Empty:
                self._check_p1_opportunity()
                continue
            self._execute_task(task)
            self._last_p0_time = time.time()
        log.info("Task executor stopped")

    def _execute_task(self, task) -> None:
        """Execute a single task: set p0_active -> LLM call -> reply -> clear."""
        self.state.p0_active.set()
        start = time.time()
        response = ""
        try:
            # Build context
            snap = self.state.snapshot()
            ring2_source = ""
            try:
                ring2_source = (self.ring2_path / "main.py").read_text()
            except FileNotFoundError:
                pass

            memories = []
            if self.memory_store:
                try:
                    memories = self.memory_store.get_recent(3)
                except Exception:
                    pass

            skills = []
            if self.skill_store:
                try:
                    skills = self.skill_store.get_active(10)
                except Exception:
                    pass

            context = _build_task_context(snap, ring2_source, memories=memories, skills=skills)
            user_message = f"{context}\n\n## User Request\n{task.text}"

            # LLM call (with web tool support)
            try:
                response = self.client.send_message_with_tools(
                    TASK_SYSTEM_PROMPT, user_message,
                    tools=WEB_TOOLS, tool_executor=_execute_tool,
                )
            except LLMError as exc:
                log.error("Task LLM error: %s", exc)
                response = f"Sorry, I couldn't process that request: {exc}"

            # Truncate if needed
            if len(response) > _MAX_REPLY_LEN:
                response = response[:_MAX_REPLY_LEN] + "\n... (truncated)"

            # Reply
            try:
                self.reply_fn(response)
            except Exception:
                log.error("Failed to send task reply", exc_info=True)
        finally:
            self.state.p0_active.clear()
            # Record task in memory
            duration = time.time() - start
            if self.memory_store:
                try:
                    snap = self.state.snapshot()
                    self.memory_store.add(
                        generation=snap.get("generation", 0),
                        entry_type="task",
                        content=task.text,
                        metadata={
                            "response_summary": response[:200],
                            "duration_sec": round(duration, 2),
                        },
                    )
                except Exception:
                    log.debug("Failed to record task in memory", exc_info=True)

    def _check_p1_opportunity(self) -> None:
        """Check if we should trigger a P1 autonomous task."""
        if not self.p1_enabled:
            return

        now = time.time()
        # Check idle threshold
        if now - self._last_p0_time < self.p1_idle_threshold_sec:
            return
        # Check interval between P1 checks
        if now - self._last_p1_check < self.p1_check_interval_sec:
            return

        self._last_p1_check = now

        # Need task history to infer useful work
        if not self.memory_store:
            return

        try:
            task_history = self.memory_store.get_by_type("task", limit=10)
        except Exception:
            return

        if not task_history:
            return

        # Build P1 decision prompt
        parts = ["## Recent Task History"]
        for task in task_history:
            content = task.get("content", "")
            meta = task.get("metadata", {})
            summary = meta.get("response_summary", "")
            parts.append(f"- Task: {content}")
            if summary:
                parts.append(f"  Result: {summary[:100]}")
        parts.append("")

        # Include directive if set
        snap = self.state.snapshot()
        directive = snap.get("evolution_directive", "")
        if directive:
            parts.append(f"## Standing Directive: {directive}")
            parts.append("")

        user_message = "\n".join(parts)

        try:
            decision = self.client.send_message(P1_SYSTEM_PROMPT, user_message)
        except LLMError as exc:
            log.debug("P1 decision LLM error: %s", exc)
            return

        # Parse decision
        if "## Decision" not in decision or "YES" not in decision.split("## Task")[0]:
            log.debug("P1 decision: NO")
            return

        # Extract task description
        task_desc = ""
        if "## Task" in decision:
            task_desc = decision.split("## Task", 1)[1].strip()

        if not task_desc:
            return

        log.info("P1 autonomous task triggered: %s", task_desc[:80])
        self._execute_p1_task(task_desc)

    def _execute_p1_task(self, task_desc: str) -> None:
        """Execute a P1 autonomous task and report via Telegram."""
        self.state.p1_active.set()
        start = time.time()
        response = ""
        try:
            # Build context (same as P0)
            snap = self.state.snapshot()
            ring2_source = ""
            try:
                ring2_source = (self.ring2_path / "main.py").read_text()
            except FileNotFoundError:
                pass

            memories = []
            if self.memory_store:
                try:
                    memories = self.memory_store.get_recent(3)
                except Exception:
                    pass

            skills = []
            if self.skill_store:
                try:
                    skills = self.skill_store.get_active(10)
                except Exception:
                    pass

            context = _build_task_context(snap, ring2_source, memories=memories, skills=skills)
            user_message = f"{context}\n\n## Autonomous Task\n{task_desc}"

            try:
                response = self.client.send_message_with_tools(
                    TASK_SYSTEM_PROMPT, user_message,
                    tools=WEB_TOOLS, tool_executor=_execute_tool,
                )
            except LLMError as exc:
                log.error("P1 task LLM error: %s", exc)
                response = f"Error: {exc}"

            if len(response) > _MAX_REPLY_LEN:
                response = response[:_MAX_REPLY_LEN] + "\n... (truncated)"

            # Report to user
            report = f"[P1 Autonomous Work] {task_desc}\n\n{response}"
            if len(report) > _MAX_REPLY_LEN:
                report = report[:_MAX_REPLY_LEN] + "\n... (truncated)"
            try:
                self.reply_fn(report)
            except Exception:
                log.error("Failed to send P1 report", exc_info=True)
        finally:
            self.state.p1_active.clear()
            # Record in memory
            duration = time.time() - start
            if self.memory_store:
                try:
                    snap = self.state.snapshot()
                    self.memory_store.add(
                        generation=snap.get("generation", 0),
                        entry_type="p1_task",
                        content=task_desc,
                        metadata={
                            "response_summary": response[:200],
                            "duration_sec": round(duration, 2),
                        },
                    )
                except Exception:
                    log.debug("Failed to record P1 task in memory", exc_info=True)

    def stop(self) -> None:
        """Signal the executor loop to stop."""
        self._running = False


def create_executor(
    config,
    state,
    ring2_path: pathlib.Path,
    reply_fn,
    memory_store=None,
    skill_store=None,
) -> TaskExecutor | None:
    """Create a TaskExecutor from Ring1Config, or None if no API key."""
    if not config.claude_api_key:
        log.warning("Task executor: no CLAUDE_API_KEY — disabled")
        return None
    client = ClaudeClient(
        api_key=config.claude_api_key,
        model=config.claude_model,
        max_tokens=config.claude_max_tokens,
    )
    return TaskExecutor(
        state, client, ring2_path, reply_fn,
        memory_store=memory_store,
        skill_store=skill_store,
        p1_enabled=config.p1_enabled,
        p1_idle_threshold_sec=config.p1_idle_threshold_sec,
        p1_check_interval_sec=config.p1_check_interval_sec,
    )


def start_executor_thread(executor: TaskExecutor) -> threading.Thread:
    """Start the executor in a daemon thread and return the thread handle."""
    thread = threading.Thread(
        target=executor.run, name="task-executor", daemon=True,
    )
    thread.start()
    return thread
