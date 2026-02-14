"""Task Executor — processes P0 user tasks via Claude API.

Runs in a daemon thread.  Pulls tasks from state.task_queue, calls the LLM,
and replies via the bot's _send_reply.  Sets/clears state.p0_active so the
Sentinel can skip evolution while a user task is in flight.

Pure stdlib (threading, queue, logging).
"""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
import time

from ring1.llm_client import ClaudeClient, LLMError

log = logging.getLogger("protea.task_executor")

_MAX_REPLY_LEN = 4000  # Telegram message limit safety margin

TASK_SYSTEM_PROMPT = """\
You are Protea, a self-evolving artificial life agent running on a host machine.
You are helpful and concise.  Answer the user's question or perform the requested
analysis.  You have context about your current state (generation, survival, code).
Keep responses under 3500 characters so they fit in a Telegram message.
"""


def _build_task_context(
    state_snapshot: dict,
    ring2_source: str,
    memories: list[dict] | None = None,
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
    ) -> None:
        """
        Args:
            state: SentinelState with task_queue, p0_active, p0_event.
            client: ClaudeClient instance for LLM calls.
            ring2_path: Path to ring2 directory (for reading source).
            reply_fn: Callable(text: str) -> None to send Telegram reply.
            memory_store: Optional MemoryStore for experiential memories.
        """
        self.state = state
        self.client = client
        self.ring2_path = ring2_path
        self.reply_fn = reply_fn
        self.memory_store = memory_store
        self._running = True

    def run(self) -> None:
        """Main loop — blocks on queue, executes tasks serially."""
        log.info("Task executor started")
        while self._running:
            try:
                task = self.state.task_queue.get(timeout=2)
            except queue.Empty:
                continue
            try:
                self._execute_task(task)
            except Exception:
                log.error("Task execution error (non-fatal)", exc_info=True)
        log.info("Task executor stopped")

    def _execute_task(self, task) -> None:
        """Execute a single task: set p0_active → LLM call → reply → clear."""
        self.state.p0_active.set()
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
            context = _build_task_context(snap, ring2_source, memories=memories)
            user_message = f"{context}\n\n## User Request\n{task.text}"

            # LLM call
            try:
                response = self.client.send_message(TASK_SYSTEM_PROMPT, user_message)
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

    def stop(self) -> None:
        """Signal the executor loop to stop."""
        self._running = False


def create_executor(config, state, ring2_path: pathlib.Path, reply_fn, memory_store=None) -> TaskExecutor | None:
    """Create a TaskExecutor from Ring1Config, or None if no API key."""
    if not config.claude_api_key:
        log.warning("Task executor: no CLAUDE_API_KEY — disabled")
        return None
    client = ClaudeClient(
        api_key=config.claude_api_key,
        model=config.claude_model,
        max_tokens=config.claude_max_tokens,
    )
    return TaskExecutor(state, client, ring2_path, reply_fn, memory_store=memory_store)


def start_executor_thread(executor: TaskExecutor) -> threading.Thread:
    """Start the executor in a daemon thread and return the thread handle."""
    thread = threading.Thread(
        target=executor.run, name="task-executor", daemon=True,
    )
    thread.start()
    return thread
