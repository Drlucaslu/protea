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

from ring1.habit_detector import _strip_context_prefix
from ring1.llm_base import LLMClient, LLMError
from ring1.tool_registry import ToolRegistry

log = logging.getLogger("protea.task_executor")

_MAX_REPLY_LEN = 8000  # Allow longer replies; split into segments if needed

_TG_MSG_LIMIT = 4000  # Telegram hard limit ~4096, leave margin

import re as _re

_RECALL_KEYWORD_RE = _re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")

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
  Accepts relative paths, absolute paths, or ~/… paths.
- write_file: Write content to a file (creates parent dirs if needed).
  Accepts relative paths, absolute paths, or ~/… paths.
- edit_file: Search-and-replace edit on a file (old_string must be unique).
- list_dir: List files and subdirectories.
All file tools can access any path within the user's home directory (~/).

Shell tool:
- exec: Execute a shell command (timeout 120s). Only truly destructive commands
  (rm -rf /, dd, mkfs, shutdown, fork bombs) are blocked. You CAN run browsers,
  install packages, start services, etc.

Message tool:
- message: Send a progress update to the user during multi-step work.

Background tool:
- spawn: Start a long-running background task. Results are sent via Telegram when done.

Skill tools:
- run_skill: Start a stored skill by name. Returns status, output, HTTP port.
- view_skill: Read the source code and metadata of a stored skill.
- edit_skill: Edit a skill's source code using search-and-replace (old_string must be unique).
  After editing, use run_skill to restart the skill with the updated code.

Schedule tool:
- manage_schedule: Create, list, remove, enable, or disable scheduled/recurring tasks.
  Use when the user wants timers, reminders, cron jobs, or repeating tasks.
  Actions: create (needs name, cron_expr, task_text), list, remove, enable, disable.
  For cron: use 5-field cron expressions (e.g. "*/5 * * * *" = every 5 minutes).
  For one-shot: set schedule_type="once" and cron_expr to an ISO datetime.

File delivery tool:
- send_file: Send a local file to the user via Telegram.
  Use AFTER writing or generating any file that the user needs.

Use web tools when the user's request requires current information from the web.
Use file/shell tools when the user asks to read, modify, or explore files and code.
Use the message tool to keep the user informed during long operations.
Use spawn for tasks that may take a long time (complex analysis, multi-file operations).
Do NOT use tools for questions you can answer from your training data alone.

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
- Write all generated files (scripts, PDFs, reports, data) to the output/ directory.
- Example: write_file with path "output/report.pdf", NOT "report.pdf".
- Use subdirectories for organization: output/scripts/, output/reports/, output/data/.
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

P1_SYSTEM_PROMPT = """\
You are Protea, a self-evolving artificial life agent.  Your owner has been
interacting with you through tasks.  Based on the task history and any standing
directives, decide whether there is useful proactive work you can do right now.

Rules:
- Only suggest work that is clearly valuable based on observed patterns.
- Do NOT suggest work if the history is empty or too sparse to infer needs.
- Do NOT repeat tasks that have already been completed recently.
- Keep the task description concise and actionable.
- NEVER restart, stop, or kill Protea processes. Process lifecycle is managed by the Sentinel automatically.
- NEVER modify .env, config.toml, run.py, or stop_run.sh.

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
    recalled: list[dict] | None = None,
    recommended_skills: list[dict] | None = None,
    other_skills: list[dict] | None = None,
    semantic_rules: list[dict] | None = None,
    gene_patterns: list[dict] | None = None,
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

    if gene_patterns:
        parts.append("")
        parts.append("## Proven Code Patterns")
        for gene in gene_patterns:
            score = gene.get("score", 0)
            task_hits = gene.get("total_task_hits", 0) or 0
            summary = gene.get("gene_summary", "")
            if len(summary) > 200:
                summary = summary[:197] + "..."
            parts.append(f"- [score={score:.2f}, tasks={task_hits}] {summary}")

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
        parts.append("## ⚡ Recommended Skills (match your task — use these first)")
        for skill in recommended_skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            parts.append(f"- **{name}**: {desc}")
        if other_skills:
            parts.append("")
            parts.append("## Other Available Skills")
            for skill in other_skills:
                name = skill.get("name", "?")
                desc = skill.get("description", "")
                parts.append(f"- {name}: {desc}")
    elif other_skills:
        # No recommended — show all as a flat list (fallback / prefer_local_skills off).
        parts.append("")
        parts.append("## Available Skills")
        for skill in other_skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            parts.append(f"- {name}: {desc}")
    elif skills:
        # Legacy path: plain skills list (no matching).
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

    if recalled:
        parts.append("")
        parts.append("## Recalled Memories")
        for mem in recalled:
            gen = mem.get("generation", "?")
            content = mem.get("content", "")[:200]
            parts.append(f"- [Gen {gen}, archived] {content}")

    if semantic_rules:
        parts.append("")
        parts.append("## Correction Rules (MUST follow)")
        for rule in semantic_rules[:10]:
            content = rule.get("content", "")[:150]
            parts.append(f"- {content}")

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
        prefer_local_skills: bool = True,
        scheduled_store=None,
        preference_store=None,
        gene_pool=None,
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
            prefer_local_skills: Match tasks to skills and recommend them.
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
        self.prefer_local_skills = prefer_local_skills
        self.scheduled_store = scheduled_store
        self.preference_store = preference_store
        self.gene_pool = gene_pool
        self.preference_extractor = None  # Initialized in create_executor()
        self.feedback_fn = None  # Called after complete task response
        self._cross_domain_counter: int = 0  # Rate-limit: 1 per 5 tasks
        self._running = True
        self._last_p0_time: float = time.time()
        self._last_p1_check: float = 0.0
        self._last_habit_check: float = 0.0
        # Habit detector — initialized later in create_executor() with templates.
        self.habit_detector = None
        # Convergence detector — initialized later in create_executor().
        self.convergence_detector = None
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
        skills_used: list[str] = []
        tool_sequence: list[str] = []
        _gene_ids_used: list[int] = []
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
                    recalled = self.memory_store.recall(keywords, emb, limit=2)
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

            gene_patterns: list[dict] = []
            if self.gene_pool:
                try:
                    gene_emb = None
                    if self.embedding_provider:
                        try:
                            vecs = self.embedding_provider.embed([match_text])
                            gene_emb = vecs[0] if vecs else None
                        except Exception:
                            pass
                    gene_patterns = self.gene_pool.get_relevant(
                        match_text, 3, query_embedding=gene_emb, min_semantic=1.0,
                    )
                    _gene_ids_used = [g["id"] for g in gene_patterns if "id" in g]
                    if gene_patterns:
                        log.info("Gene injection: %s",
                                 [(g["id"], g.get("_relevance", 0)) for g in gene_patterns])
                except Exception:
                    log.debug("Gene retrieval for task failed", exc_info=True)

            history = self._get_recent_history()
            context = _build_task_context(
                snap, ring2_source, memories=memories,
                chat_history=history, recalled=recalled,
                recommended_skills=recommended_skills,
                other_skills=other_skills,
                semantic_rules=semantic_rules,
                gene_patterns=gene_patterns,
            )
            user_message = f"{context}\n\n## User Request\n{task.text}"

            # LLM call with tool registry
            try:
                if self.registry:
                    def tracking_execute(tool_name: str, tool_input: dict) -> str:
                        tool_sequence.append(tool_name)
                        if tool_name == "run_skill":
                            skills_used.append(tool_input.get("skill_name", "unknown"))
                        return self.registry.execute(tool_name, tool_input)

                    response = self.client.send_message_with_tools(
                        TASK_SYSTEM_PROMPT, user_message,
                        tools=self.registry.get_schemas(),
                        tool_executor=tracking_execute,
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

            # Check for correction pattern and persist as semantic_rule.
            self._check_and_store_correction(task.text)

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
                _send_segmented(self.reply_fn, response + footer)
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
                    task_meta = {
                        "response_summary": response[:200],
                        "duration_sec": round(duration, 2),
                        "skills_used": skills_used,
                        "skills_matched": skills_matched,
                        "tool_sequence": tool_sequence,
                        "gene_ids_used": _gene_ids_used,
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
                        response_text=response[:200],
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

    def _propose_habit(self, pattern) -> None:
        """Push a habit proposal to state.habit_proposals for the bot to send."""
        if not self.habit_detector:
            return
        self.habit_detector.mark_proposed(pattern.pattern_key)

        # Build message text depending on pattern type.
        if pattern.pattern_type == "template":
            label = pattern.task_summary or pattern.template_name
            if pattern.all_samples:
                samples_text = "\n".join(f"• {s}" for s in pattern.all_samples[:5])
                text = (
                    f"*发现重复模式*\n\n"
                    f"你最近 {pattern.count} 次执行了「{label}」相关任务：\n"
                    f"{samples_text}\n\n"
                    f"建议创建定时任务自动执行。"
                )
            else:
                text = (
                    f"*发现重复模式*\n\n"
                    f"你最近 {pattern.count} 次执行了「{label}」相关任务：\n"
                    f"「{pattern.sample_task}」\n\n"
                    f"建议创建定时任务自动执行。"
                )
        else:
            text = (
                f"*发现重复模式*\n\n"
                f"你最近 {pattern.count} 次执行了类似任务：\n"
                f"「{pattern.sample_task}」\n\n"
                f"建议创建定时任务自动执行。"
            )

        buttons = []
        cron = pattern.suggested_cron or "0 9 * * *"
        try:
            from ring0.cron import describe
            desc = describe(cron)
        except Exception:
            desc = cron

        # Encode auto_stop_hours into callback_data after "|"
        auto_stop = getattr(pattern, "auto_stop_hours", 0) or 0
        cb_suffix = f"|{auto_stop}" if auto_stop else ""

        btn_text = f"自动执行 ({desc})"
        if auto_stop:
            btn_text = f"自动执行 ({desc}, {auto_stop}小时后自动停止)"

        buttons.append([{
            "text": btn_text,
            "callback_data": f"habit:schedule:{pattern.pattern_key}:{cron}{cb_suffix}",
        }])
        buttons.append([{
            "text": "不需要",
            "callback_data": f"habit:dismiss:{pattern.pattern_key}",
        }])

        # Store proposal context so callback can build a proper task_text.
        habit_ctx = getattr(self.state, "_habit_context", None)
        if habit_ctx is not None:
            # Look up clarification_prompt from template config.
            clarification_prompt = ""
            if self.habit_detector and pattern.template_name:
                for tmpl in self.habit_detector._templates:
                    if tmpl.get("id") == pattern.template_name:
                        clarification_prompt = tmpl.get("clarification_prompt", "")
                        break
            habit_ctx[pattern.pattern_key] = {
                "task_text": pattern.sample_task,
                "task_summary": pattern.task_summary,
                "all_samples": pattern.all_samples,
                "clarification_prompt": clarification_prompt,
                "cron_expr": cron,
                "auto_stop_hours": auto_stop,
            }

        # Put into state queue for bot to consume.
        habit_q = getattr(self.state, "habit_proposals", None)
        if habit_q is not None:
            habit_q.put((text, buttons))
        else:
            # Fallback: send directly via reply_fn (no keyboard).
            if self.reply_fn:
                try:
                    self.reply_fn(text)
                except Exception:
                    log.debug("Failed to send habit proposal", exc_info=True)

    def _check_p1_opportunity(self) -> None:
        """Check if we should trigger a P1 autonomous task."""
        if not self.p1_enabled:
            return

        now = time.time()

        # Habit detection — two-layer: template + high-threshold repetitive
        if self.habit_detector and now - self._last_habit_check >= 3600:
            self._last_habit_check = now
            try:
                patterns = self.habit_detector.detect()
                log.info("Habit detection: found %d patterns", len(patterns))
                if patterns:
                    self._propose_habit(patterns[0])  # max 1 proposal at a time
            except Exception:
                log.debug("Habit detection failed", exc_info=True)
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
        skills_used: list[str] = []
        tool_sequence: list[str] = []
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

            recalled: list[dict] = []
            if self.memory_store:
                try:
                    keywords = _extract_recall_keywords(task_desc)
                    emb = None
                    if self.embedding_provider:
                        try:
                            vecs = self.embedding_provider.embed([task_desc])
                            emb = vecs[0] if vecs else None
                        except Exception:
                            pass
                    recalled = self.memory_store.recall(keywords, emb, limit=2)
                except Exception:
                    pass

            # P1 task descriptions are LLM-generated (English), but translate if needed.
            p1_match_text = task_desc
            try:
                p1_intent = self._extract_profile_intent(task_desc)
                if p1_intent:
                    p1_match_text = p1_intent
            except Exception:
                pass

            skills = []
            recommended_skills_p1: list[dict] = []
            other_skills_p1: list[dict] = []
            if self.skill_store:
                try:
                    skills = self.skill_store.get_active()
                except Exception:
                    pass
                if skills and self.prefer_local_skills:
                    recommended_skills_p1, other_skills_p1 = _match_skills(p1_match_text, skills)
                else:
                    other_skills_p1 = skills
            skills_matched = [s["name"] for s in recommended_skills_p1]
            if skills_matched and self.skill_store:
                try:
                    self.skill_store.record_matches(skills_matched)
                except Exception:
                    pass

            semantic_rules_p1: list[dict] = []
            if self.memory_store:
                try:
                    semantic_rules_p1 = self.memory_store.get_semantic_rules(limit=10)
                except Exception:
                    pass

            context = _build_task_context(
                snap, ring2_source, memories=memories, recalled=recalled,
                recommended_skills=recommended_skills_p1,
                other_skills=other_skills_p1,
                semantic_rules=semantic_rules_p1,
            )
            user_message = f"{context}\n\n## Autonomous Task\n{task_desc}"

            try:
                if self.registry:
                    def tracking_execute(tool_name: str, tool_input: dict) -> str:
                        tool_sequence.append(tool_name)
                        if tool_name == "run_skill":
                            skills_used.append(tool_input.get("skill_name", "unknown"))
                        return self.registry.execute(tool_name, tool_input)

                    response = self.client.send_message_with_tools(
                        TASK_SYSTEM_PROMPT, user_message,
                        tools=self.registry.get_schemas(),
                        tool_executor=tracking_execute,
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
            elapsed = time.time() - start
            if skills_used:
                footer = f"\n---\nskill: {', '.join(skills_used)} | {elapsed:.0f}s"
            else:
                footer = f"\n---\nllm | {elapsed:.0f}s"
            report = f"[P1 Autonomous Work] {task_desc}\n\n{response}{footer}"
            if len(report) > _MAX_REPLY_LEN:
                report = report[:_MAX_REPLY_LEN] + "\n... (truncated)"
            try:
                _send_segmented(self.reply_fn, report)
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
                            "skills_used": skills_used,
                            "skills_matched": skills_matched,
                            "tool_sequence": tool_sequence,
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
    scheduled_store=None,
    send_file_fn=None,
    preference_store=None,
    gene_pool=None,
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
        scheduled_store=scheduled_store,
        send_file_fn=send_file_fn,
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
        p1_enabled=config.p1_enabled,
        p1_idle_threshold_sec=config.p1_idle_threshold_sec,
        p1_check_interval_sec=config.p1_check_interval_sec,
        max_tool_rounds=max_tool_rounds,
        user_profiler=user_profiler,
        embedding_provider=embedding_provider,
        prefer_local_skills=prefer_local_skills,
        scheduled_store=scheduled_store,
        preference_store=preference_store,
        gene_pool=gene_pool,
    )
    executor.subagent_manager = subagent_mgr

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

    # Initialize habit detector with templates and LLM client.
    if memory_store:
        from ring1.habit_detector import HabitDetector, load_templates
        project_root = pathlib.Path(ring2_path).parent
        templates = load_templates(project_root / "config" / "task_templates.json")
        executor.habit_detector = HabitDetector(
            memory_store,
            scheduled_store,
            templates=templates,
            llm_client=client,
        )

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
