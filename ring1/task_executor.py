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

_MAX_REPLY_LEN = 8000  # Allow longer replies; split into segments if needed

_TG_MSG_LIMIT = 4000  # Telegram hard limit ~4096, leave margin

import re as _re

_RECALL_KEYWORD_RE = _re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")

# Regex to strip conversation context prefix injected by telegram_bot.
_CONTEXT_PREFIX_RE = _re.compile(
    r"^\[Context:[^\]]*\]\n"
    r"Your (?:previous )?message: \".*?\"[\n]+"
    r"User(?:'s reply|\s+now says): ",
    _re.DOTALL,
)


def _strip_context_prefix(text: str) -> str:
    """Remove conversation context prefix, returning only the user's text."""
    return _CONTEXT_PREFIX_RE.sub("", text)

_SKILL_TOKEN_RE = _re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")

# Patterns that indicate a user correction / standing instruction.
_CORRECTION_PATTERNS: list[_re.Pattern[str]] = [
    _re.compile(r"(不对|错了|不是这样|搞错了|弄错了)"),
    _re.compile(r"(你应该|应该是|正确的是|正确做法)"),
    _re.compile(r"(记住|要记住|你要记住|别忘了).*[，,。.]"),
    _re.compile(r"(下次|以后|每次).*(要|应该|别|不要)"),
    _re.compile(r"(wrong|incorrect|should be|remember|always|never)", _re.IGNORECASE),
]

# --- Task content cleaning for memory storage ---
# Strip code blocks, stack traces, and long URLs to keep only user intent.
_CODE_BLOCK_RE = _re.compile(r"```[\s\S]*?```")
_TRACEBACK_RE = _re.compile(
    r"Traceback \(most recent call last\):[\s\S]*?(?:\n\S+Error:.*)",
)
_LONG_URL_RE = _re.compile(r"https?://\S{80,}")
# Redact credentials: "password/密码/token/secret/key" followed by separator and value,
# or email-like + non-space token pairs that look like "user pass" login lines.
_CREDENTIAL_PATTERNS: list[_re.Pattern[str]] = [
    # Explicit labels: password=xxx, 密码：xxx, token: xxx, etc.
    _re.compile(
        r"(password|passwd|密码|口令|token|secret|api[_-]?key|credential)"
        r"[\s:=：]+\S+",
        _re.IGNORECASE,
    ),
    # "email password" pattern on same line (e.g. "user@host.com MyP@ss123")
    _re.compile(r"\S+@\S+\.\S+\s+\S*[A-Z]\S*[0-9&!@#$%]\S*"),
]
_MAX_MEMORY_CONTENT_LEN = 200


def _clean_for_memory(text: str) -> str:
    """Strip code blocks, stack traces, long URLs, and credentials for memory storage.

    Keeps only the natural-language intent.  Truncates to 200 chars if
    still long after cleaning.
    """
    cleaned = _CODE_BLOCK_RE.sub("", text)
    cleaned = _TRACEBACK_RE.sub("", cleaned)
    cleaned = _LONG_URL_RE.sub("", cleaned)
    for pat in _CREDENTIAL_PATTERNS:
        cleaned = pat.sub("[REDACTED]", cleaned)
    # Collapse whitespace left by removals.
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > _MAX_MEMORY_CONTENT_LEN:
        cleaned = cleaned[:_MAX_MEMORY_CONTENT_LEN] + "..."
    # If cleaning removed everything, fall back to truncated original.
    if not cleaned:
        cleaned = text[:_MAX_MEMORY_CONTENT_LEN] + ("..." if len(text) > _MAX_MEMORY_CONTENT_LEN else "")
    return cleaned


def _tokenize_for_matching(text: str) -> set[str]:
    """Extract English match tokens from text (3+ chars).

    Chinese characters are ignored — callers should translate Chinese text
    to English via LLM before calling this function, since all skill
    names/descriptions/tags are in English.
    """
    raw_tokens = _SKILL_TOKEN_RE.findall(text.lower())
    tokens: set[str] = set()
    for t in raw_tokens:
        if t[0] >= "\u4e00":  # CJK — skip, not useful for English skill matching
            continue
        elif len(t) >= 3:
            tokens.add(t)
    return tokens


# ---------------------------------------------------------------------------
# Fabrication detection — catch LLM "performing" actions in text without tools
# ---------------------------------------------------------------------------

_FABRICATION_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    # Claims of calling tools without actually calling them
    (_re.compile(r"已调用\s*(send_file|web_fetch|web_search|exec|run_skill|message)\s*\("), "tool_call_claim"),
    # Fake phased workflow reports (Phase 1 Complete, Phase 2, etc.)
    (_re.compile(r"(Phase|阶段)\s*\d+\s*(Complete|完成|Done)", _re.IGNORECASE), "fake_phase"),
    # Fake API response JSON embedded in text
    (_re.compile(r'"(sources|binance|coingecko|coinmarketcap|price|average)":\s*[\d{]'), "fake_api_data"),
    # Claims of file delivery without tool
    (_re.compile(r"(已发送|已推送|文件已|报告已发送)(到|至|给)"), "fake_delivery"),
    # Fake timestamps with wrong year or implausible precision
    (_re.compile(r"(Current UTC|当前时间|UTC时间):\s*20\d{2}-\d{2}-\d{2}"), "fake_timestamp"),
]

# Tools that produce real data
_DATA_TOOLS = {"web_fetch", "web_search", "exec", "run_skill", "read_file"}


def _detect_fabrication(response: str, tool_sequence: list[str]) -> list[str]:
    """Detect if the LLM fabricated actions it didn't actually perform.

    Returns a list of fabrication signal descriptions (empty = clean).
    """
    signals: list[str] = []
    tool_set = set(tool_sequence)

    for pattern, signal_type in _FABRICATION_PATTERNS:
        if pattern.search(response):
            if signal_type == "tool_call_claim":
                m = pattern.search(response)
                claimed_tool = m.group(1) if m else ""
                if claimed_tool not in tool_set:
                    signals.append(f"{signal_type}:{claimed_tool}")
            elif signal_type == "fake_api_data":
                if not tool_set & _DATA_TOOLS:
                    signals.append(signal_type)
            elif signal_type == "fake_phase":
                if "message" not in tool_set:
                    signals.append(signal_type)
            elif signal_type == "fake_delivery":
                if "send_file" not in tool_set:
                    signals.append(signal_type)
            elif signal_type == "fake_timestamp":
                if not tool_set & _DATA_TOOLS:
                    signals.append(signal_type)

    return signals


def _match_skills(task_text: str, skills: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split skills into (recommended, other) based on keyword overlap with task.

    Returns recommended skills sorted by match score (descending).
    Uses three-way filtering to reduce false positives:
    - At least _MIN_SCORE tokens must match
    - Matched ratio must be >= _MIN_RATIO
    - At most _MAX_RECOMMENDED skills are recommended
    """
    tokens = _tokenize_for_matching(task_text)
    if not tokens:
        return [], list(skills)

    total_tokens = len(tokens)
    _MIN_SCORE = 2
    _MIN_RATIO = 0.15
    _MAX_RECOMMENDED = 10

    scored: list[tuple[int, dict]] = []
    for skill in skills:
        haystack = " ".join([
            skill.get("name", "").replace("_", " "),
            skill.get("description", ""),
            " ".join(skill.get("tags", [])),
        ]).lower()
        score = sum(1 for t in tokens if t in haystack)
        scored.append((score, skill))

    recommended: list[dict] = []
    other: list[dict] = []
    for score, skill in sorted(scored, key=lambda x: -x[0]):
        ratio = score / total_tokens if total_tokens else 0
        if score >= _MIN_SCORE and ratio >= _MIN_RATIO and len(recommended) < _MAX_RECOMMENDED:
            recommended.append(skill)
        else:
            other.append(skill)
    return recommended, other


def _send_segmented(reply_fn, text: str, limit: int = _TG_MSG_LIMIT) -> None:
    """Send *text* via *reply_fn*, splitting into segments if too long.

    Splits on newline boundaries to avoid breaking mid-sentence.
    """
    if len(text) <= limit:
        reply_fn(text)
        return

    segments: list[str] = []
    while text:
        if len(text) <= limit:
            segments.append(text)
            break
        # Find last newline within limit.
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit  # no good break point — hard cut
        segments.append(text[:cut])
        text = text[cut:].lstrip("\n")

    for i, seg in enumerate(segments):
        if len(segments) > 1:
            seg = f"[{i + 1}/{len(segments)}]\n{seg}"
        reply_fn(seg)


def _extract_recall_keywords(text: str) -> list[str]:
    """Extract keywords from task text for archive recall."""
    tokens = _RECALL_KEYWORD_RE.findall(text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            keywords.append(t)
            if len(keywords) >= 10:
                break
    return keywords

_TASK_SYSTEM_PROMPT_BASE = """\
You are Protea, a self-evolving artificial life agent running on a host machine.
You are helpful and concise.  Answer the user's question or perform the requested
analysis.  You have context about your current state (generation, survival, code).
Keep responses under 3500 characters so they fit in a Telegram message.

PROGRESS REPORTING: For multi-step tasks, call message() to report progress between steps.
Example: message("🔄 Searching...") → [do work] → message("✅ Done, analyzing...")
Report before expensive operations and after each major step. Use emojis: 🔄 ✅ ❌ 📊 🔍.

Use the tools provided to complete the task. Refer to tool descriptions for usage.
Key workflows: spawn for long tasks, send_file after generating any file, view_skill before using a skill's API.
NEVER read/scan the user's personal directories (Documents, Downloads, Desktop, etc.).

ANTI-FABRICATION RULES (严格执行):
- NEVER describe tool calls you didn't make. If you say "I fetched X", you MUST
  have actually called web_fetch. If you say "I sent the file", you MUST have
  actually called send_file.
- NEVER generate fake API responses, fake JSON data, or fake file paths in your
  text. All data must come from actual tool calls.
- NEVER simulate multi-phase workflows in plain text. Use the message tool for
  real progress updates and actual tools for real work.
- If you cannot perform an action (API unavailable, tool missing), SAY SO honestly
  instead of pretending you did it.

SKILL PREFERENCE: When "Recommended Skills" are listed in the context, ALWAYS prefer
using run_skill to execute them instead of reimplementing the same functionality yourself.
Skills are pre-built, tested, and optimized. Only fall back to direct implementation
if no recommended skill matches or if the skill fails.

IMPORTANT skill workflow: When working with a skill that exposes an HTTP API, ALWAYS
call view_skill FIRST to read its source code and understand the correct API endpoints,
request methods, and parameters. Do NOT guess endpoint paths — check the code.
If a skill interaction fails, do NOT repeatedly try shell commands to debug. Instead,
use view_skill to read the source and understand the correct usage.

FILE OUTPUT RULES:
- Write all generated files to the output/ directory.  Files are auto-routed
  by extension into subdirectories: data/ (.json .csv .xml .yaml), reports/ (.pdf),
  scripts/ (.py .sh), docs/ (.md), logs/ (.txt .log).
- Example: write_file with path "output/report.pdf" → saved to output/reports/report.pdf.
- You can also use task subdirs: "output/bitcoin/data.json" → output/data/bitcoin/data.json.
- NEVER write generated files directly to the project root directory.
- You may read any project file, but generated content must go to output/.

REMOTE ACCESS — CRITICAL:
- The user interacts EXCLUSIVELY via Telegram from a remote device.
  They CANNOT access local files on this machine.
- After generating ANY file (PDF, report, script, image, data), you MUST use
  send_file to deliver it. Just writing to disk is NOT enough.
- Uploaded files are in the telegram_output/ directory.
- Workflow: write_file → (generate content) → send_file.
"""

def _task_system_prompt() -> str:
    from ring1.soul import inject
    return inject(_TASK_SYSTEM_PROMPT_BASE)


def _build_task_context(
    state_snapshot: dict,
    ring2_source: str,
    memories: list[dict] | None = None,
    skills: list[dict] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
    recalled: list[dict] | None = None,
    recommended_skills: list[dict] | None = None,
    other_skills: list[dict] | None = None,
    semantic_rules: list[dict] | None = None,
    strategies: list[dict] | None = None,
    reflections: list[dict] | None = None,
    fragment_registry=None,
    task_text: str | None = None,
    preference_summary: str | None = None,
) -> str:
    """Build context string from current Protea state for LLM task calls."""
    # Fragment-based path: rank and select by relevance within token budget.
    if fragment_registry and task_text:
        fragments = fragment_registry.collect(
            state_snapshot, ring2_source, memories, skills,
            recommended_skills, other_skills, semantic_rules,
            strategies, reflections, recalled, chat_history,
            preference_summary=preference_summary,
        )
        ranked = fragment_registry.rank(fragments, task_text)
        selected = fragment_registry.select(ranked)
        log.info(
            "Context fragmentation: %d fragments collected, %d selected (%d/%d tokens)",
            len(fragments), len(selected),
            sum(f.token_est for f in selected),
            fragment_registry.token_budget,
        )
        return fragment_registry.assemble(selected)

    # Fallback: original concatenation logic.
    from datetime import datetime
    parts = ["## Protea State"]
    parts.append(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}")
    parts.append(f"Generation: {state_snapshot.get('generation', '?')}")
    parts.append(f"Alive: {state_snapshot.get('alive', '?')}")
    parts.append(f"Paused: {state_snapshot.get('paused', '?')}")
    parts.append(f"Last score: {state_snapshot.get('last_score', '?')}")
    parts.append(f"Last survived: {state_snapshot.get('last_survived', '?')}")
    parts.append("")

    if ring2_source:
        truncated = ring2_source[:500]
        if len(ring2_source) > 500:
            truncated += "\n... (truncated)"
        parts.append("## Ring 2 Code")
        parts.append("```python")
        parts.append(truncated)
        parts.append("```")

    if strategies:
        parts.append("")
        parts.append("## Proven Strategies")
        for strat in strategies:
            content = strat.get("content", "")
            if len(content) > 200:
                content = content[:197] + "..."
            parts.append(f"- {content}")

    if memories:
        parts.append("")
        parts.append("## Recent Learnings")
        for mem in memories:
            gen = mem.get("generation", "?")
            content = mem.get("content", "")
            parts.append(f"- [Gen {gen}] {content}")

    # Skill sections: recommended first (if skill matching is active), then other.
    if recommended_skills:
        parts.append("")
        parts.append("## ⚡ Recommended Skills (use these first)")
        for skill in recommended_skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            parts.append(f"- **{name}**: {desc}")
        if other_skills:
            parts.append("")
            _MAX_OTHER_SKILLS = 20
            other_names = [s.get("name", "?") for s in other_skills[:_MAX_OTHER_SKILLS]]
            remaining = len(other_skills) - _MAX_OTHER_SKILLS
            names_str = ", ".join(other_names)
            if remaining > 0:
                names_str += f" (and {remaining} more)"
            parts.append(f"Other skills: {names_str}")
    elif other_skills:
        parts.append("")
        _MAX_OTHER_SKILLS = 20
        other_names = [s.get("name", "?") for s in other_skills[:_MAX_OTHER_SKILLS]]
        remaining = len(other_skills) - _MAX_OTHER_SKILLS
        names_str = ", ".join(other_names)
        if remaining > 0:
            names_str += f" (and {remaining} more)"
        parts.append(f"Available skills: {names_str}")
    elif skills:
        parts.append("")
        _MAX_OTHER_SKILLS = 20
        skill_names = [s.get("name", "?") for s in skills[:_MAX_OTHER_SKILLS]]
        remaining = len(skills) - _MAX_OTHER_SKILLS
        names_str = ", ".join(skill_names)
        if remaining > 0:
            names_str += f" (and {remaining} more)"
        parts.append(f"Available skills: {names_str}")

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

    if recalled:
        parts.append("")
        parts.append("## Recalled Memories")
        for mem in recalled:
            gen = mem.get("generation", "?")
            content = mem.get("content", "")[:400]
            parts.append(f"- [Gen {gen}, archived] {content}")

    if semantic_rules:
        parts.append("")
        parts.append("## Learned Patterns")
        for rule in semantic_rules[:10]:
            content = rule.get("content", "")[:300]
            parts.append(f"- {content}")

    if reflections:
        parts.append("")
        parts.append("## Past Reflections (lessons from similar tasks)")
        for ref in reflections[:5]:
            content = ref.get("content", "")[:400]
            parts.append(f"- {content}")

    if preference_summary:
        parts.append("")
        parts.append("## User Preferences")
        parts.append(preference_summary)

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
        max_tool_rounds: int = 25,
        user_profiler=None,
        embedding_provider=None,
        prefer_local_skills: bool = True,
        scheduled_store=None,
        preference_store=None,
        reply_fn_factory=None,
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
            max_tool_rounds: Maximum LLM tool-call round-trips.
            user_profiler: Optional UserProfiler for interest tracking.
            embedding_provider: Optional EmbeddingProvider for semantic vectors.
            prefer_local_skills: Match tasks to skills and recommend them.
            reply_fn_factory: Optional factory(chat_id, reply_to_message_id) -> reply_fn.
        """
        self.state = state
        self.client = client
        self.ring2_path = ring2_path
        self.reply_fn = reply_fn
        self.registry = registry
        self.memory_store = memory_store
        self.skill_store = skill_store
        self.task_store = task_store
        self.max_tool_rounds = max_tool_rounds
        self.user_profiler = user_profiler
        self.embedding_provider = embedding_provider
        self.prefer_local_skills = prefer_local_skills
        self.scheduled_store = scheduled_store
        self.preference_store = preference_store
        self.reply_fn_factory = reply_fn_factory
        self.preference_extractor = None  # Initialized in create_executor()
        self.feedback_fn = None  # Called after complete task response
        self.nudge_fn = None  # Called after feedback: nudge_engine.post_task_nudge
        self.reflector = None  # Set by sentinel after creation
        self._cross_domain_counter: int = 0  # Rate-limit: 1 per 5 tasks
        self._task_counter: int = 0  # P0: triggers strategy distillation every 10 tasks
        self._running = True
        # Convergence detector — initialized later in create_executor().
        self.convergence_detector = None
        # Conversation history: list of (timestamp, user_text, response_text)
        self._chat_history: list[tuple[float, str, str]] = []
        self._chat_history_max = 5
        self._chat_history_ttl = 600  # 10 minutes
        self._last_chat_id: str = ""  # Track most recent chat_id

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
        log.info("Task executor stopped")

    _RECOVER_MAX_AGE_SEC = 300  # skip tasks older than 5 minutes on restart

    def _recover_tasks(self) -> None:
        """Recover pending/executing tasks from the store after restart.

        Tasks older than ``_RECOVER_MAX_AGE_SEC`` are marked as expired
        rather than re-enqueued, so stale work doesn't block new messages.
        """
        if not self.task_store:
            return
        try:
            # Reset executing → pending (interrupted by restart)
            for t in self.task_store.get_executing():
                self.task_store.set_status(t["task_id"], "pending")
            # Re-enqueue pending tasks that are still fresh.
            now = time.time()
            from ring1.telegram_bot import Task
            expired = 0
            for t in self.task_store.get_pending():
                age = now - (t.get("created_at") or 0)
                if age > self._RECOVER_MAX_AGE_SEC:
                    self.task_store.set_status(t["task_id"], "failed", "expired on restart")
                    expired += 1
                    continue
                task = Task(
                    text=t["text"],
                    chat_id=t["chat_id"],
                    created_at=t["created_at"],
                    task_id=t["task_id"],
                )
                self.state.task_queue.put(task)
            count = self.state.task_queue.qsize()
            if expired:
                log.info("Expired %d stale tasks on restart (>%ds old)",
                         expired, self._RECOVER_MAX_AGE_SEC)
            if count:
                log.info("Recovered %d tasks from store", count)
        except Exception:
            log.error("Task recovery failed", exc_info=True)

    def _execute_shell_task(self, task, reply_fn) -> None:
        """Execute a shell-mode task directly via subprocess, bypassing the LLM."""
        import subprocess
        from ring0.scheduled_task_store import ScheduledTaskStore

        command = ScheduledTaskStore.extract_shell_command(task.text)
        if not command:
            log.warning("Shell task but no command extracted, falling back to LLM: %s", task.text[:80])
            self._execute_task_llm(task)
            return

        log.info("Shell bypass: %s", command[:120])
        start = time.time()
        try:
            project_root = self.ring2_path.parent
            result = subprocess.run(
                command, shell=True, capture_output=True, timeout=120,
                cwd=str(project_root), text=True,
            )
            output = result.stdout or ""
            if result.returncode != 0 and result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if not output.strip():
                output = f"(completed with exit code {result.returncode})"
        except subprocess.TimeoutExpired:
            output = "[error] Command timed out after 120s"
        except Exception as exc:
            output = f"[error] {exc}"

        elapsed = time.time() - start
        footer = f"\n---\nshell | {elapsed:.0f}s"
        try:
            _send_segmented(reply_fn, output + footer)
        except Exception:
            log.error("Failed to send shell task reply", exc_info=True)

    def _execute_task(self, task) -> None:
        """Execute a single task: set p0_active -> LLM call -> reply -> clear."""
        log.info("P0 task received: %s", task.text[:80])
        self.state.p0_active.set()

        # Per-task reply routing: group messages go back to their chat/thread.
        task_reply_fn = self.reply_fn
        reply_to_id = getattr(task, "reply_to_message_id", None)
        task_chat_id = getattr(task, "chat_id", "")
        if task_chat_id:
            self._last_chat_id = task_chat_id
        if self.reply_fn_factory and task_chat_id:
            task_reply_fn = self.reply_fn_factory(task_chat_id, reply_to_id)

        # Shell-mode bypass: run script directly, skip LLM pipeline entirely.
        if getattr(task, "exec_mode", "llm") == "shell":
            start = time.time()
            try:
                if self.task_store:
                    try:
                        self.task_store.set_status(task.task_id, "executing")
                    except Exception:
                        pass
                self._execute_shell_task(task, task_reply_fn)
            finally:
                self.state.p0_active.clear()
                now = time.time()
                with self.state.lock:
                    self.state.last_task_completion = now
                if self.task_store:
                    try:
                        self.task_store.set_status(task.task_id, "completed", "(shell)")
                    except Exception:
                        pass
                log.info("Shell task done (%.1fs): %s", time.time() - start, task.text[:80])
            return

        # Mark executing in store
        if self.task_store:
            try:
                self.task_store.set_status(task.task_id, "executing")
            except Exception:
                log.debug("Failed to mark task executing", exc_info=True)
        start = time.time()
        response = ""
        skills_used: list[str] = []
        tool_sequence: list[str] = []
        fab_signals: list[str] = []
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
                    # Load latest status snapshot as context anchor.
                    snapshots = self.memory_store.get_by_type("status_snapshot", limit=1)
                    recent = self.memory_store.get_recent(3)
                    recent = [m for m in recent if m.get("entry_type") != "status_snapshot"]
                    memories = snapshots + recent[:2]
                except Exception:
                    pass

            recalled: list[dict] = []
            if self.memory_store:
                try:
                    keywords = _extract_recall_keywords(task.text)
                    emb = None
                    if self.embedding_provider:
                        try:
                            vecs = self.embedding_provider.embed([task.text])
                            emb = vecs[0] if vecs else None
                        except Exception:
                            pass
                    recalled = self.memory_store.recall(keywords, query_embedding=emb, limit=2)
                except Exception:
                    pass

            # Translate task text to English for skill matching (skills are English).
            english_intent = ""
            try:
                english_intent = self._extract_profile_intent(task.text)
            except Exception:
                pass
            match_text = english_intent if english_intent else task.text

            skills = []
            recommended_skills: list[dict] = []
            other_skills: list[dict] = []
            if self.skill_store:
                try:
                    skills = self.skill_store.get_active()
                except Exception:
                    pass
                if skills and self.prefer_local_skills:
                    recommended_skills, other_skills = _match_skills(match_text, skills)
                else:
                    other_skills = skills
            skills_matched = [s["name"] for s in recommended_skills]
            if skills_matched and self.skill_store:
                try:
                    self.skill_store.record_matches(skills_matched)
                except Exception:
                    pass

            semantic_rules: list[dict] = []
            if self.memory_store:
                try:
                    semantic_rules = self.memory_store.get_semantic_rules(limit=10)
                except Exception:
                    pass

            strategies: list[dict] = []
            if self.memory_store:
                try:
                    strategies = self.memory_store.get_strategies(limit=5)
                except Exception:
                    pass

            history = self._get_recent_history()

            # Reflexion: retrieve relevant historical reflections
            reflections: list[dict] = []
            if self.reflector:
                try:
                    reflections = self.reflector.get_relevant_reflections(task.text)
                except Exception:
                    pass

            # Build fragment registry for context selection (if embedding available).
            frag_registry = None
            if self.embedding_provider:
                try:
                    from ring1.context_fragments import FragmentRegistry
                    budget = getattr(self, '_context_token_budget', 3000)
                    frag_registry = FragmentRegistry(self.embedding_provider, token_budget=budget)
                except Exception:
                    log.debug("FragmentRegistry init failed", exc_info=True)

            # Fetch structured preference summary.
            pref_summary = ""
            if self.preference_store:
                try:
                    pref_summary = self.preference_store.get_preference_summary_text()
                except Exception:
                    log.debug("Failed to get preference summary", exc_info=True)

            context = _build_task_context(
                snap, ring2_source, memories=memories,
                chat_history=history, recalled=recalled,
                recommended_skills=recommended_skills,
                other_skills=other_skills,
                semantic_rules=semantic_rules,
                strategies=strategies,
                reflections=reflections,
                fragment_registry=frag_registry,
                task_text=task.text,
                preference_summary=pref_summary,
            )
            user_message = f"{context}\n\n## User Request\n{task.text}"


            # Store chat_id and reply_to_message_id in thread context for tools (fix routing bug)
            import threading
            threading.current_thread().task_chat_id = task_chat_id
            threading.current_thread().reply_to_message_id = reply_to_id
            # LLM call with tool registry
            try:
                if self.registry:
                    def tracking_execute(tool_name: str, tool_input: dict) -> str:
                        tool_sequence.append(tool_name)
                        if tool_name == "run_skill":
                            skills_used.append(tool_input.get("skill_name", "unknown"))
                        result = self.registry.execute(tool_name, tool_input)
                        # P1 anti-drift: inject goal reminder on long tool chains.
                        if (len(tool_sequence) > 10
                                and len(tool_sequence) % 5 == 0):
                            result = (
                                "[REMINDER: Stay focused on the original user request. "
                                "Do not drift into unrelated work.]\n" + result
                            )
                        return result

                    response = self.client.send_message_with_tools(
                        _task_system_prompt(), user_message,
                        tools=self.registry.get_schemas(),
                        tool_executor=tracking_execute,
                        max_rounds=self.max_tool_rounds,
                    )
                else:
                    response = self.client.send_message(
                        _task_system_prompt(), user_message,
                    )
            except LLMError as exc:
                log.error("Task LLM error: %s", exc)
                response = f"Sorry, I couldn't process that request: {exc}"

            # --- Fabrication detection ---
            fab_signals[:] = _detect_fabrication(response, tool_sequence)
            if fab_signals and not getattr(task, '_fab_retried', False):
                log.warning("Fabrication detected (signals: %s), retrying with correction", fab_signals)
                retry_prompt = (
                    "CRITICAL: Your previous response contained fabricated actions — you described "
                    "calling tools or producing data without actually using any tools. "
                    "This is NOT acceptable.\n\n"
                    "You MUST use actual tools (web_search, web_fetch, exec, run_skill, etc.) "
                    "to perform real actions. Do NOT describe actions in text — execute them.\n\n"
                    "If you genuinely cannot perform the task (no API available, tool error), "
                    "say so honestly.\n\n"
                    f"Original request: {task.text}"
                )
                tool_sequence.clear()
                skills_used.clear()
                try:
                    if self.registry:
                        response = self.client.send_message_with_tools(
                            _task_system_prompt(), retry_prompt,
                            tools=self.registry.get_schemas(),
                            tool_executor=tracking_execute,
                            max_rounds=self.max_tool_rounds,
                        )
                    else:
                        response = self.client.send_message(
                            _task_system_prompt(), retry_prompt,
                        )
                except LLMError as exc:
                    log.error("Retry LLM error: %s", exc)

                # Re-check after retry
                fab_signals[:] = _detect_fabrication(response, tool_sequence)

            if fab_signals:
                log.warning("Fabrication persists after retry: %s", fab_signals)
                response = (
                    "抱歉，这个任务没有完成。我尝试了但没有实际调用任何工具来获取数据，"
                    "生成的内容不可靠，已丢弃。\n\n"
                    "可能的原因：API 超时、网络不稳定、或任务超出当前能力范围。"
                    "请稍后重试，或将任务拆分为更小的步骤。"
                )

            # Truncate if needed
            if len(response) > _MAX_REPLY_LEN:
                response = response[:_MAX_REPLY_LEN] + "\n... (truncated)"

            # Record conversation history for context continuity.
            # Skip fabricated responses — they add noise and can prime
            # the model to continue generating similar fabricated text.
            if not fab_signals:
                self._record_history(task.text, response)

            # Check for correction pattern and persist as semantic_rule.
            # Use stripped text to avoid storing conversation metadata as rules.
            self._check_and_store_correction(_strip_context_prefix(task.text))

            # Multi-round convergence detection.
            if self.convergence_detector:
                try:
                    self.convergence_detector.record(task.text, response)
                    result = self.convergence_detector.check()
                    if result:
                        text, buttons = result
                        self.state.convergence_proposals.put((text, buttons))
                except Exception:
                    log.debug("Convergence check failed", exc_info=True)

            # Cross-domain inspiration check (rate-limited: 1 per 5 tasks).
            inspiration = self._check_cross_domain_inspiration(
                task.text, response,
            )
            if inspiration:
                response += f"\n\n{inspiration}"

            # Resolution footer.
            elapsed = time.time() - start
            if skills_used:
                footer = f"\n---\nskill: {', '.join(skills_used)} | {elapsed:.0f}s"
            else:
                footer = f"\n---\nllm | {elapsed:.0f}s"

            # Reply — split into segments for Telegram's 4096-char limit.
            try:
                _send_segmented(task_reply_fn, response + footer)
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
            usage = (self.client.last_usage if self.client else None) or {}
            if not isinstance(usage, dict):
                usage = {}
            log.info("P0 task done (%.1fs): %s", duration, response[:80])
            log.info("TASK_METRICS task_id=%s duration=%.1f input_tokens=%d output_tokens=%d "
                     "tools=%d skills=%s",
                     getattr(task, "task_id", "?"), duration,
                     usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                     len(tool_sequence), ",".join(skills_used) or "none")
            # Signal urgent reflection for significant events.
            urgent_reason = ""
            input_tokens = usage.get("input_tokens", 0)
            if input_tokens > 50000:
                urgent_reason = f"high_tokens({input_tokens})"
            elif fab_signals:
                urgent_reason = "fabrication_detected"
            elif any(pat.search(task.text) for pat in _CORRECTION_PATTERNS):
                urgent_reason = "user_correction"
            if urgent_reason and hasattr(self.state, "pending_reflection_reason"):
                with self.state.lock:
                    self.state.pending_reflection_reason = urgent_reason
                log.info("Urgent reflection flagged: %s", urgent_reason)
            # Strip conversation context prefix so only the user's actual
            # text is persisted in memory and used for profile analysis.
            clean_text = _strip_context_prefix(task.text)
            # Further clean for memory: strip code blocks, stack traces,
            # long URLs, and truncate — keeps only the user's intent.
            memory_text = _clean_for_memory(clean_text)
            # Record task in memory (with optional embedding).
            if self.memory_store:
                try:
                    snap = self.state.snapshot()
                    embedding = None
                    if self.embedding_provider:
                        try:
                            vecs = self.embedding_provider.embed([memory_text])
                            embedding = vecs[0] if vecs else None
                        except Exception:
                            log.debug("Embedding generation failed", exc_info=True)
                    usage = (self.client.last_usage if self.client else None) or {}
                    if not isinstance(usage, dict):
                        usage = {}
                    task_meta = {
                        "response_summary": response[:200],
                        "duration_sec": round(duration, 2),
                        "skills_used": skills_used,
                        "skills_matched": skills_matched,
                        "tool_sequence": tool_sequence,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    }
                    if embedding is not None:
                        self.memory_store.add_with_embedding(
                            generation=snap.get("generation", 0),
                            entry_type="task",
                            content=memory_text,
                            metadata=task_meta,
                            embedding=embedding,
                        )
                    else:
                        self.memory_store.add(
                            generation=snap.get("generation", 0),
                            entry_type="task",
                            content=memory_text,
                            metadata=task_meta,
                        )
                except Exception:
                    log.debug("Failed to record task in memory", exc_info=True)
            # Update user profile (reuse english_intent from skill matching).
            dominant_category = ""
            if self.user_profiler:
                try:
                    profile_text = english_intent if english_intent else self._extract_profile_intent(memory_text)
                    dominant_category = self.user_profiler.update_from_task(profile_text) or ""
                except Exception:
                    log.debug("Failed to update user profile", exc_info=True)
            # Extract implicit preferences.
            if self.preference_extractor:
                try:
                    self.preference_extractor.extract_and_store(
                        task_text=memory_text,
                        response_text=response[:800],
                        category_hint=dominant_category,
                    )
                except Exception:
                    log.debug("Failed to extract preferences", exc_info=True)
            # Feedback prompt — only after full task completion.
            if self.feedback_fn:
                try:
                    self.feedback_fn()
                except Exception:
                    pass
            # Nudge — contextual suggestion after task completion.
            if self.nudge_fn:
                try:
                    nudge_context = {
                        "response_summary": response[:300],
                        "skills_used": skills_used,
                        "tool_sequence": tool_sequence,
                        "duration": duration,
                    }
                    result = self.nudge_fn(clean_text, response, nudge_context)
                    if result:
                        nudge_q = getattr(self.state, "nudge_queue", None)
                        if nudge_q is not None:
                            nudge_q.put(result)
                except Exception:
                    log.debug("Nudge generation failed", exc_info=True)
            # P0: Trigger strategy distillation every 10 tasks.
            self._task_counter += 1
            if self._task_counter % 10 == 0:
                try:
                    self._distill_strategies()
                except Exception:
                    log.debug("Distillation trigger failed", exc_info=True)
    _PROFILE_INTENT_PROMPT = (
        "You are a concise intent extractor. Given a user message, do two things:\n"
        "1. Identify ONLY the user's core intent (what they want to do). "
        "Ignore any pasted articles, URLs, reference material, code blocks, "
        "stack traces, or quoted text — those are context, not intent.\n"
        "2. Output the intent as a single English sentence (max 20 words). "
        "If the message is already in English, just clean and shorten it.\n"
        "If the intent is unclear or the message is just an acknowledgement "
        '(e.g. "ok", "好的", "thanks"), output: unclear\n'
        "Output ONLY the English sentence, nothing else."
    )

    def _extract_profile_intent(self, text: str) -> str:
        """Use LLM to extract the user's core intent in English.

        Falls back to the original text if the LLM call fails.
        """
        if not self.client:
            return text
        # Skip very short texts (acknowledgements, single words)
        stripped = text.strip()
        if len(stripped) < 5:
            return text
        try:
            result = self.client.send_message(
                self._PROFILE_INTENT_PROMPT, stripped,
            )
            if not isinstance(result, str):
                return text
            result = result.strip()
            if not result or result.lower() == "unclear":
                return ""
            # Sanity check: LLM should return a short sentence
            if len(result) > 200:
                result = result[:200]
            return result
        except Exception:
            log.debug("LLM intent extraction failed, using raw text", exc_info=True)
            return text

    def _check_and_store_correction(self, task_text: str) -> None:
        """If *task_text* contains a correction pattern, store as semantic_rule."""
        if not self.memory_store:
            return
        for pat in _CORRECTION_PATTERNS:
            if pat.search(task_text):
                rule_text = task_text.strip()
                if len(rule_text) > 200:
                    rule_text = rule_text[:200]
                self.memory_store.add(
                    generation=self.state.snapshot().get("generation", 0),
                    entry_type="semantic_rule",
                    content=rule_text,
                    importance=0.8,
                )
                log.info("Correction detected, stored as semantic_rule: %s", rule_text[:60])
                break

    def _check_cross_domain_inspiration(
        self,
        task_text: str,
        response: str,
    ) -> str | None:
        """Check if there's a cross-domain connection to share with the user.

        Conditions (all must be met to avoid being annoying):
        - Task text is substantive (>= 30 characters)
        - memory_store and user_profiler are available
        - Rate limit: triggers at most once per 5 tasks
        - Found a high-relevance cross-domain memory

        Returns inspiration text or None.
        """
        self._cross_domain_counter += 1
        if self._cross_domain_counter % 5 != 0:
            return None

        if len(task_text.strip()) < 30:
            return None

        if not self.memory_store or not self.user_profiler:
            return None

        try:
            profile = self.user_profiler.get_categories()
            if not profile:
                return None

            # Build embedding if available.
            query_embedding = None
            if self.embedding_provider:
                try:
                    vecs = self.embedding_provider.embed([task_text[:200]])
                    query_embedding = vecs[0] if vecs else None
                except Exception:
                    pass

            results = self.memory_store.search_cross_domain(
                current_task=task_text,
                user_profile={"categories": profile},
                limit=1,
                query_embedding=query_embedding,
            )

            if not results:
                return None

            hit = results[0]
            content = hit.get("content", "")
            if not content or len(content) < 10:
                return None

            # Truncate for display.
            if len(content) > 150:
                content = content[:150] + "..."

            return (
                "\u2728 Cross-domain insight:\n"
                f"{content}\n"
                "Want me to explore this connection?"
            )
        except Exception:
            log.debug("Cross-domain inspiration check failed", exc_info=True)
            return None

    # --- P0: Experience Distillation ---

    _DISTILL_PROMPT = (
        "You are a strategy distiller. Given recent task execution data, "
        "extract 1-3 reusable strategies as short, actionable rules. "
        "Each strategy should be a single sentence describing a proven pattern.\n"
        "Format: one strategy per line, starting with '- '.\n"
        "If no clear patterns emerge, output: NONE"
    )

    def _distill_strategies(self) -> None:
        """Distill strategies from recent task execution patterns."""
        if not self.memory_store or not self.client:
            return
        try:
            recent_tasks = self.memory_store.get_by_type("task", limit=15)
            # Filter tasks that have tool_sequence in metadata.
            tasks_with_tools = [
                t for t in recent_tasks
                if t.get("metadata", {}).get("tool_sequence")
            ]
            if len(tasks_with_tools) < 3:
                return

            # Build pattern text.
            parts: list[str] = []
            for t in tasks_with_tools[:15]:
                meta = t.get("metadata", {})
                parts.append(
                    f"- Task: {t['content'][:100]}\n"
                    f"  Tools: {', '.join(meta.get('tool_sequence', []))}\n"
                    f"  Skills: {', '.join(meta.get('skills_used', []))}\n"
                    f"  Duration: {meta.get('duration_sec', '?')}s"
                )
            user_msg = "## Recent Task Executions\n" + "\n".join(parts)

            result = self.client.send_message(self._DISTILL_PROMPT, user_msg)
            if not result or "NONE" in result.upper():
                return

            # Parse strategies (lines starting with '- ').
            snap = self.state.snapshot()
            gen = snap.get("generation", 0)
            for line in result.strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    strategy_text = line[2:].strip()
                    if 10 < len(strategy_text) < 300:
                        self.memory_store.add(
                            generation=gen,
                            entry_type="strategy",
                            content=strategy_text,
                        )
            log.info("Strategy distillation completed")
        except Exception:
            log.debug("Strategy distillation failed", exc_info=True)

    # --- P3: Evolution Signal Extraction ---

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
    user_profiler=None,
    embedding_provider=None,
    scheduled_store=None,
    send_file_fn=None,
    preference_store=None,
    reply_fn_factory=None,
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
        workspace=workspace,
        reply_fn=reply_fn,
        reply_fn_factory=reply_fn_factory,
        skill_store=skill_store,
        skill_runner=skill_runner,
        scheduled_store=scheduled_store,
        send_file_fn=send_file_fn,
    )
    subagent_mgr = SubagentManager(client, base_registry, reply_fn)

    # Rebuild registry with spawn tool included
    registry = create_default_registry(
        workspace=workspace,
        reply_fn=reply_fn,
        reply_fn_factory=reply_fn_factory,
        spawn_fn=subagent_mgr,
        skill_store=skill_store,
        skill_runner=skill_runner,
        scheduled_store=scheduled_store,
        send_file_fn=send_file_fn,
    )

    prefer_local_skills = getattr(config, "prefer_local_skills", True)
    executor = TaskExecutor(
        state, client, ring2_path, reply_fn,
        registry=registry,
        memory_store=memory_store,
        skill_store=skill_store,
        task_store=task_store,
        max_tool_rounds=max_tool_rounds,
        user_profiler=user_profiler,
        embedding_provider=embedding_provider,
        prefer_local_skills=prefer_local_skills,
        scheduled_store=scheduled_store,
        preference_store=preference_store,
        reply_fn_factory=reply_fn_factory,
    )
    executor.subagent_manager = subagent_mgr

    # Context fragment token budget from config.
    emb_cfg = getattr(config, "_raw_cfg", {}).get("ring1", {}).get("embeddings", {})
    executor._context_token_budget = emb_cfg.get("context_token_budget", 3000)

    # Initialize preference extractor.
    if preference_store:
        try:
            from ring1.preference_extractor import PreferenceExtractor
            user_profile_cfg = getattr(config, "_raw_cfg", {}).get("ring1", {}).get("user_profile", {})
            rate_limit = user_profile_cfg.get("extraction_rate_limit_sec", 300)
            executor.preference_extractor = PreferenceExtractor(
                llm_client=client,
                preference_store=preference_store,
                rate_limit_sec=rate_limit,
            )
            log.info("PreferenceExtractor initialized (rate_limit=%ds)", rate_limit)
        except Exception:
            log.debug("PreferenceExtractor init failed (non-fatal)", exc_info=True)

    # Initialize convergence detector.
    if memory_store and embedding_provider:
        from ring1.convergence_detector import ConvergenceDetector
        conv_cfg = getattr(config, "_raw_cfg", {}).get("ring1", {}).get("convergence", {})
        executor.convergence_detector = ConvergenceDetector(
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            llm_client=client,
            convergence_context=state._convergence_context,
            config=conv_cfg,
        )

    return executor


def start_executor_thread(executor: TaskExecutor) -> threading.Thread:
    """Start the executor in a daemon thread and return the thread handle."""
    thread = threading.Thread(
        target=executor.run, name="task-executor", daemon=True,
    )
    thread.start()
    return thread
