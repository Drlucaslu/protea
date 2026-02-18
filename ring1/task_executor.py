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

import logging
import pathlib
import queue
import threading
import time

from ring1.llm_base import LLMClient, LLMError
from ring1.tool_registry import ToolRegistry

log = logging.getLogger("protea.task_executor")

_MAX_REPLY_LEN = 4000  # Telegram message limit safety margin

TASK_SYSTEM_PROMPT = """\
You are Protea, a self-evolving artificial life agent running on a host machine.
You are helpful and concise.  Answer the user's question or perform the requested
analysis.  You have context about your current state (generation, survival, code).
Keep responses under 3500 characters so they fit in a Telegram message.

⚠️ CRITICAL: PROGRESS REPORTING IS MANDATORY ⚠️

The message tool is your PRIMARY way to communicate progress during work.
You MUST follow these rules WITHOUT EXCEPTION:

WHEN TO REPORT:
✓ Send initial message IMMEDIATELY when starting any task expected to take >10 seconds
✓ Report after EACH major step in multi-step operations (>3 steps)
✓ Report every 100 iterations in loops, OR every 10 seconds (whichever comes first)
✓ Always report BEFORE starting expensive operations (web scraping, file processing)
✓ When using spawn, the subagent MUST report MORE frequently (user can't see logs)

HOW TO REPORT:
✓ Use clear emojis: 🔄 (working), ✅ (done), ❌ (error), 📊 (analyzing), 🔍 (searching)
✓ Include progress metrics: percentages, counts, time estimates when possible
✓ Show what's next: always preview the upcoming step
✓ Keep messages concise but informative

EXAMPLE - Research Task:
User asks: "Research quantum computing trends"
Your response should include:
  1. message("🔄 Starting research on quantum computing...\n\n**Phase 1**: Web search\n**Phase 2**: Content extraction\n**Phase 3**: Analysis")
  2. [perform web_search]
  3. message("✅ **Phase 1 Complete**: Found 10 sources\n\n🔄 **Phase 2**: Extracting content...")
  4. [web_fetch multiple sources]
  5. message("✅ **Phase 2 Complete**: Extracted 15,000 words from 10 sources\n\n🔄 **Phase 3**: Analyzing...")
  6. [analyze content]
  7. [provide final response with completion summary]

You have access to the following tools:

Web tools:
- web_search: Search the web using DuckDuckGo. Use for research or lookup tasks.
- web_fetch: Fetch and read the content of a specific URL.

File tools:
- read_file: Read a file's contents (with line numbers, offset, limit).
- write_file: Write content to a file (creates parent dirs if needed).
- edit_file: Search-and-replace edit on a file (old_string must be unique).
- list_dir: List files and subdirectories.

Shell tool:
- exec: Execute a shell command. Dangerous commands are blocked.

Message tool:
- message: Send a progress update to the user during multi-step work.

Background tool:
- spawn: Start a long-running background task. Results are sent via Telegram when done.

Skill tools:
- run_skill: Start a stored skill by name. Returns status, output, HTTP port.
- view_skill: Read the source code and metadata of a stored skill.
- edit_skill: Edit a skill's source code using search-and-replace (old_string must be unique).
  After editing, use run_skill to restart the skill with the updated code.

Use web tools when the user's request requires current information from the web.
Use file/shell tools when the user asks to read, modify, or explore files and code.
Use the message tool to keep the user informed during long operations.
Use spawn for tasks that may take a long time (complex analysis, multi-file operations).
Do NOT use tools for questions you can answer from your training data alone.

IMPORTANT skill workflow: When working with a skill that exposes an HTTP API, ALWAYS
call view_skill FIRST to read its source code and understand the correct API endpoints,
request methods, and parameters. Do NOT guess endpoint paths — check the code.
If a skill interaction fails, do NOT repeatedly try shell commands to debug. Instead,
use view_skill to read the source and understand the correct usage.

FILE OUTPUT RULES:
- Write all generated files (scripts, PDFs, reports, data) to the output/ directory.
- Example: write_file with path "output/report.pdf", NOT "report.pdf".
- Use subdirectories for organization: output/scripts/, output/reports/, output/data/.
- NEVER write generated files directly to the project root directory.
- You may read any project file, but generated content must go to output/.
"""

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
    chat_history: list[tuple[str, str]] | None = None,
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

    if chat_history:
        parts.append("")
        parts.append("## Recent Conversation")
        for user_msg, assistant_msg in chat_history:
            # Truncate long messages to keep context manageable.
            u = user_msg[:500] + "..." if len(user_msg) > 500 else user_msg
            a = assistant_msg[:1000] + "..." if len(assistant_msg) > 1000 else assistant_msg
            parts.append(f"User: {u}")
            parts.append(f"Assistant: {a}")
            parts.append("")

    return "\n".join(parts)


class TaskExecutor:
    """Processes user tasks from the queue, one at a time."""

    def __init__(
        self,
        state,
        client: LLMClient,
        ring2_path: pathlib.Path,
        reply_fn,
        registry: ToolRegistry | None = None,
        memory_store=None,
        skill_store=None,
        task_store=None,
        p1_enabled: bool = False,
        p1_idle_threshold_sec: int = 600,
        p1_check_interval_sec: int = 60,
        max_tool_rounds: int = 25,
        user_profiler=None,
        embedding_provider=None,
    ) -> None:
        """
        Args:
            state: SentinelState with task_queue, p0_active, p0_event.
            client: ClaudeClient instance for LLM calls.
            ring2_path: Path to ring2 directory (for reading source).
            reply_fn: Callable(text: str) -> None to send Telegram reply.
            registry: ToolRegistry for tool dispatch.  None = no tools.
            memory_store: Optional MemoryStore for experiential memories.
            skill_store: Optional SkillStore for reusable skills.
            task_store: Optional TaskStore for task persistence.
            p1_enabled: Whether P1 autonomous tasks are enabled.
            p1_idle_threshold_sec: Seconds of idle before triggering P1.
            p1_check_interval_sec: Minimum seconds between P1 checks.
            max_tool_rounds: Maximum LLM tool-call round-trips.
            user_profiler: Optional UserProfiler for interest tracking.
            embedding_provider: Optional EmbeddingProvider for semantic vectors.
        """
        self.state = state
        self.client = client
        self.ring2_path = ring2_path
        self.reply_fn = reply_fn
        self.registry = registry
        self.memory_store = memory_store
        self.skill_store = skill_store
        self.task_store = task_store
        self.p1_enabled = p1_enabled
        self.p1_idle_threshold_sec = p1_idle_threshold_sec
        self.p1_check_interval_sec = p1_check_interval_sec
        self.max_tool_rounds = max_tool_rounds
        self.user_profiler = user_profiler
        self.embedding_provider = embedding_provider
        self._running = True
        self._last_p0_time: float = time.time()
        self._last_p1_check: float = 0.0
        # Conversation history: list of (timestamp, user_text, response_text)
        self._chat_history: list[tuple[float, str, str]] = []
        self._chat_history_max = 5
        self._chat_history_ttl = 600  # 10 minutes

    def _get_recent_history(self) -> list[tuple[str, str]]:
        """Return recent conversation pairs, pruning expired entries."""
        now = time.time()
        self._chat_history = [
            (ts, q, a) for ts, q, a in self._chat_history
            if now - ts < self._chat_history_ttl
        ]
        return [(q, a) for _, q, a in self._chat_history[-self._chat_history_max:]]

    def _record_history(self, user_text: str, response_text: str) -> None:
        """Append a Q&A pair to the conversation history."""
        self._chat_history.append((time.time(), user_text, response_text))
        # Keep only the most recent entries.
        if len(self._chat_history) > self._chat_history_max * 2:
            self._chat_history = self._chat_history[-self._chat_history_max:]

    def run(self) -> None:
        """Main loop — blocks on queue, executes tasks serially."""
        log.info("Task executor started")
        self._recover_tasks()
        while self._running:
            try:
                task = self.state.task_queue.get(timeout=2)
            except queue.Empty:
                self._check_p1_opportunity()
                continue
            try:
                self._execute_task(task)
            except Exception:
                log.error("Task executor: unhandled error", exc_info=True)
                self.state.p0_active.clear()
                # Mark task as failed in store
                if self.task_store and hasattr(task, "task_id"):
                    try:
                        self.task_store.set_status(task.task_id, "failed", "unhandled error")
                    except Exception:
                        pass
            self._last_p0_time = time.time()
        log.info("Task executor stopped")

    def _recover_tasks(self) -> None:
        """Recover pending/executing tasks from the store after restart."""
        if not self.task_store:
            return
        try:
            # Reset executing → pending (interrupted by restart)
            for t in self.task_store.get_executing():
                self.task_store.set_status(t["task_id"], "pending")
            # Re-enqueue all pending tasks
            from ring1.telegram_bot import Task
            for t in self.task_store.get_pending():
                task = Task(
                    text=t["text"],
                    chat_id=t["chat_id"],
                    created_at=t["created_at"],
                    task_id=t["task_id"],
                )
                self.state.task_queue.put(task)
            count = self.state.task_queue.qsize()
            if count:
                log.info("Recovered %d tasks from store", count)
        except Exception:
            log.error("Task recovery failed", exc_info=True)

    def _execute_task(self, task) -> None:
        """Execute a single task: set p0_active -> LLM call -> reply -> clear."""
        log.info("P0 task received: %s", task.text[:80])
        self.state.p0_active.set()
        # Mark executing in store
        if self.task_store:
            try:
                self.task_store.set_status(task.task_id, "executing")
            except Exception:
                log.debug("Failed to mark task executing", exc_info=True)
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

            history = self._get_recent_history()
            context = _build_task_context(snap, ring2_source, memories=memories, skills=skills, chat_history=history)
            user_message = f"{context}\n\n## User Request\n{task.text}"

            # LLM call with tool registry
            try:
                if self.registry:
                    response = self.client.send_message_with_tools(
                        TASK_SYSTEM_PROMPT, user_message,
                        tools=self.registry.get_schemas(),
                        tool_executor=self.registry.execute,
                        max_rounds=self.max_tool_rounds,
                    )
                else:
                    response = self.client.send_message(
                        TASK_SYSTEM_PROMPT, user_message,
                    )
            except LLMError as exc:
                log.error("Task LLM error: %s", exc)
                response = f"Sorry, I couldn't process that request: {exc}"

            # Truncate if needed
            if len(response) > _MAX_REPLY_LEN:
                response = response[:_MAX_REPLY_LEN] + "\n... (truncated)"

            # Record conversation history for context continuity.
            self._record_history(task.text, response)

            # Reply
            try:
                self.reply_fn(response)
            except Exception:
                log.error("Failed to send task reply", exc_info=True)
        finally:
            self.state.p0_active.clear()
            duration = time.time() - start
            now = time.time()
            with self.state.lock:
                self.state.last_task_completion = now
            # Mark completed in store
            if self.task_store:
                try:
                    self.task_store.set_status(
                        task.task_id, "completed", response[:500],
                    )
                except Exception:
                    log.debug("Failed to mark task completed", exc_info=True)
            log.info("P0 task done (%.1fs): %s", duration, response[:80])
            # Record task in memory (with optional embedding).
            if self.memory_store:
                try:
                    snap = self.state.snapshot()
                    embedding = None
                    if self.embedding_provider:
                        try:
                            vecs = self.embedding_provider.embed([task.text])
                            embedding = vecs[0] if vecs else None
                        except Exception:
                            log.debug("Embedding generation failed", exc_info=True)
                    if embedding is not None:
                        self.memory_store.add_with_embedding(
                            generation=snap.get("generation", 0),
                            entry_type="task",
                            content=task.text,
                            metadata={
                                "response_summary": response[:200],
                                "duration_sec": round(duration, 2),
                            },
                            embedding=embedding,
                        )
                    else:
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
            # Update user profile.
            if self.user_profiler:
                try:
                    self.user_profiler.update_from_task(task.text, response[:200])
                except Exception:
                    log.debug("Failed to update user profile", exc_info=True)

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
                if self.registry:
                    response = self.client.send_message_with_tools(
                        TASK_SYSTEM_PROMPT, user_message,
                        tools=self.registry.get_schemas(),
                        tool_executor=self.registry.execute,
                        max_rounds=self.max_tool_rounds,
                    )
                else:
                    response = self.client.send_message(
                        TASK_SYSTEM_PROMPT, user_message,
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
    skill_runner=None,
    task_store=None,
    registry_client=None,
    user_profiler=None,
    embedding_provider=None,
) -> TaskExecutor | None:
    """Create a TaskExecutor from Ring1Config, or None if no API key."""
    try:
        client = config.get_llm_client()
    except LLMError as exc:
        log.warning("Task executor: LLM client init failed — %s", exc)
        return None

    # Build tool registry with subagent support
    from ring1.subagent import SubagentManager
    from ring1.tools import create_default_registry

    workspace = getattr(config, "workspace_path", ".") or "."
    # Ensure output directory exists for LLM-generated files.
    (pathlib.Path(workspace) / "output").mkdir(parents=True, exist_ok=True)
    shell_timeout = getattr(config, "shell_timeout", 30)
    max_tool_rounds = getattr(config, "max_tool_rounds", 25)

    # Create subagent manager (needs registry, so we build in two steps)
    base_registry = create_default_registry(
        workspace_path=workspace,
        shell_timeout=shell_timeout,
        reply_fn=reply_fn,
        skill_store=skill_store,
        skill_runner=skill_runner,
        registry_client=registry_client,
    )
    subagent_mgr = SubagentManager(client, base_registry, reply_fn)

    # Rebuild registry with spawn tool included
    registry = create_default_registry(
        workspace_path=workspace,
        shell_timeout=shell_timeout,
        reply_fn=reply_fn,
        subagent_manager=subagent_mgr,
        skill_store=skill_store,
        skill_runner=skill_runner,
        registry_client=registry_client,
    )

    executor = TaskExecutor(
        state, client, ring2_path, reply_fn,
        registry=registry,
        memory_store=memory_store,
        skill_store=skill_store,
        task_store=task_store,
        p1_enabled=config.p1_enabled,
        p1_idle_threshold_sec=config.p1_idle_threshold_sec,
        p1_check_interval_sec=config.p1_check_interval_sec,
        max_tool_rounds=max_tool_rounds,
        user_profiler=user_profiler,
        embedding_provider=embedding_provider,
    )
    executor.subagent_manager = subagent_mgr
    return executor


def start_executor_thread(executor: TaskExecutor) -> threading.Thread:
    """Start the executor in a daemon thread and return the thread handle."""
    thread = threading.Thread(
        target=executor.run, name="task-executor", daemon=True,
    )
    thread.start()
    return thread
