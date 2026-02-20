"""Evolution and crystallization prompt templates for Ring 1.

Builds system + user prompts for Claude to mutate Ring 2 code.
Extracts Python code blocks from LLM responses.
Crystallization: analyse surviving Ring 2 code and extract reusable skills.
"""

from __future__ import annotations

import ast
import json
import platform
import re

SYSTEM_PROMPT = """\
You are the evolution engine for Protea, a self-evolving artificial life system.
Your task is to mutate the Ring 2 code to create a new generation.

## Absolute Constraints (MUST follow — violation = immediate death)
1. The code MUST maintain a heartbeat protocol:
   - Read the heartbeat file path from PROTEA_HEARTBEAT environment variable
   - Write the file every 2 seconds with format: "{pid}\\n{timestamp}\\n"
   - The heartbeat keeps the program alive — without it, Ring 0 will kill the process
2. The code MUST have a `main()` function as entry point
3. The code MUST be a single valid Python file (pure stdlib only, no pip packages)
4. The code MUST handle KeyboardInterrupt gracefully and clean up the heartbeat file
5. The code MUST NOT create or use SQLite databases or any persistent storage files.
   All output should go to stdout or files in the `output/` directory. Never use sqlite3
   to create .db files — the Protea memory system is managed by Ring 0, not Ring 2.

## Evolution Strategy
Beyond the heartbeat constraint, evolve the code to be PRACTICALLY USEFUL.
Prioritize capabilities that solve real-world problems users face daily.
Refer to user task history (if provided) to guide evolution direction.
Avoid duplicating existing skills — develop complementary capabilities.

### Design Principles
1. High-frequency, low-friction: solve problems users encounter daily
2. Composable: produce structured output (JSON) that other tools can consume
3. Config-driven: use JSON config files, not hardcoded values
4. Cross-platform: use subprocess with OS-appropriate commands, NOT /proc
5. Output-friendly: produce reports suitable for Telegram/PDF/JSON export

## Fitness (scored 0.0–1.0)
Survival is necessary but NOT sufficient — a program that only heartbeats scores 0.50.
- Base survival: 0.50 (survived max_runtime)
- Output volume: up to +0.10 (meaningful non-empty lines, saturates at 50 lines)
- Output diversity: up to +0.10 (unique lines / total lines)
- Output novelty: up to +0.10 (how different from recent generations — CRITICAL)
- Structured output: up to +0.10 (JSON blocks, tables, key:value reports)
- Functional bonus: up to +0.05 (real I/O, HTTP, file operations, API interaction)
- Error penalty: up to −0.10 (traceback/error/exception lines reduce score)

IMPORTANT: Novelty is scored by comparing output tokens against recent generations.
Repeating the same program pattern will score LOW on novelty. Each generation should
produce genuinely different output to maximise its score.

## Capability Evolution (optional)
If the user's recent tasks require capabilities that pure stdlib cannot provide
(e.g., browser automation, email management, calendar access), you may propose
a CAPABILITY SKILL alongside your Ring 2 code mutation.

Propose a capability ONLY when:
1. Recent user tasks clearly need it (evidence in task history)
2. No existing skill already covers it
3. The required packages are well-known and safe

Format: After the Ring 2 code block, add:

```capability
{
  "name": "skill_name_snake_case",
  "description": "One-sentence description of what this skill does",
  "dependencies": ["package1", "package2"],
  "tags": ["tag1", "tag2"],
  "source_code": "full Python source code of the skill"
}
```

If no capability is needed, omit this section entirely.

## Response Format
Start with a SHORT reflection (1-2 sentences max), then the complete code.
Keep the reflection brief — the code is what matters.

## Reflection
[1-2 sentences: what pattern you noticed and your mutation strategy]

```python
# your complete mutated code here
```
"""


def _compress_source(code: str) -> str:
    """Strip docstrings, comments, and excess blank lines to save tokens."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Collect line ranges of all docstrings.
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                ds = node.body[0]
                for ln in range(ds.lineno, ds.end_lineno + 1):
                    docstring_lines.add(ln)

    lines = code.splitlines()
    result: list[str] = []
    prev_blank = False
    for i, line in enumerate(lines, 1):
        if i in docstring_lines:
            continue
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('#!'):
            continue
        if not stripped:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line.rstrip())

    return '\n'.join(result)


def build_evolution_prompt(
    current_source: str,
    fitness_history: list[dict],
    best_performers: list[dict],
    params: dict,
    generation: int,
    survived: bool,
    directive: str = "",
    memories: list[dict] | None = None,
    task_history: list[dict] | None = None,
    skills: list[dict] | None = None,
    crash_logs: list[dict] | None = None,
    persistent_errors: list[str] | None = None,
    is_plateaued: bool = False,
    gene_pool: list[dict] | None = None,
    evolution_intent: dict | None = None,
    user_profile_summary: str = "",
    structured_preferences: str = "",
    tool_names: list[str] | None = None,
    permanent_capabilities: list[dict] | None = None,
    allowed_packages: list[str] | None = None,
    skill_hit_summary: dict | None = None,
    semantic_rules: list[dict] | None = None,
    evolution_direction: str = "",
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the evolution LLM call."""
    parts: list[str] = []

    parts.append(f"## Generation {generation}")
    parts.append(f"Previous generation {'SURVIVED' if survived else 'DIED'}.")
    parts.append(f"Mutation rate: {params.get('mutation_rate', 0.1)}")
    parts.append(f"Max runtime: {params.get('max_runtime_sec', 60)}s")
    parts.append("")

    # Platform info — prevent generating OS-specific code that can't run here.
    parts.append("## Platform")
    parts.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    if platform.system() == "Darwin":
        parts.append("WARNING: This is macOS. /proc filesystem does NOT exist. "
                      "Do NOT use /proc/net, /proc/stat, /proc/meminfo, or "
                      "/proc/[pid] paths. Use subprocess with macOS commands "
                      "(sysctl, vm_stat, netstat, ps) or platform-agnostic "
                      "Python APIs instead.")
    parts.append("")

    # Intent-driven context selection flags.
    _intent = (evolution_intent or {}).get("intent", "optimize")

    include_persistent_errors = _intent in ("repair", "optimize")
    include_user_profile = _intent in ("optimize", "explore", "adapt")
    include_gene_pool = _intent in ("optimize", "explore")
    include_semantic_rules = _intent in ("optimize", "explore", "adapt")
    include_skill_coverage = _intent in ("optimize", "explore")

    task_history_limit = 2 if _intent == "repair" else 5
    fitness_history_limit = 2 if _intent in ("explore", "adapt") else 5

    # Current source code (compressed to save tokens)
    parts.append("## Current Ring 2 Code")
    parts.append("```python")
    parts.append(_compress_source(current_source).rstrip())
    parts.append("```")
    parts.append("")

    # Fitness history (compact — limit by intent)
    if fitness_history:
        parts.append("## Recent Fitness History")
        for entry in fitness_history[:fitness_history_limit]:
            status = "SURVIVED" if entry.get("survived") else "DIED"
            detail_str = ""
            detail_raw = entry.get("detail")
            if detail_raw:
                try:
                    d = json.loads(detail_raw) if isinstance(detail_raw, str) else detail_raw
                    novelty = d.get("novelty", "?")
                    detail_str = f" novelty={novelty}"
                except (json.JSONDecodeError, TypeError):
                    pass
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f},{detail_str} "
                f"{status}"
            )
        parts.append("")

    # Best performers (compact — limit to 3)
    if best_performers:
        parts.append("## Best Performers (by score)")
        for entry in best_performers[:3]:
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f}"
            )
        parts.append("")

    # Persistent errors — MUST FIX (high priority, intent-gated)
    if include_persistent_errors and persistent_errors:
        parts.append("## PERSISTENT BUGS (must fix!)")
        parts.append("These errors have appeared across multiple generations "
                      "and MUST be fixed in this evolution:")
        for err in persistent_errors[:3]:
            parts.append(f"- {err}")
        parts.append("")

    # Learned patterns from memory (compact — limit to 3)
    if memories:
        parts.append("## Learned Patterns")
        for mem in memories[:3]:
            gen = mem.get("generation", "?")
            content = mem.get("content", "")
            # Truncate long memories to save tokens.
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"- [Gen {gen}] {content}")
        parts.append("")

    # Semantic rules — validated patterns from experience (intent-gated)
    if include_semantic_rules and semantic_rules:
        parts.append("## Semantic Rules")
        parts.append("Validated patterns from experience:")
        for rule in semantic_rules[:10]:
            content = rule.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            parts.append(f"- {content}")
        parts.append("")

    # Recent user tasks — PRIMARY evolution signal (compact, intent-limited)
    if task_history:
        parts.append("## User Tasks (PRIORITY)")
        for task in task_history[:task_history_limit]:
            content = task.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"- {content}")
        parts.append("")

    # Dynamic evolution direction (from genes + user profile)
    if evolution_direction:
        parts.append("## Evolution Direction")
        parts.append(evolution_direction)
        parts.append("")

    # User profile — aggregated interests and directions (intent-gated)
    if include_user_profile and user_profile_summary:
        parts.append("## User Profile")
        profile_text = user_profile_summary
        if len(profile_text) > 200:
            profile_text = profile_text[:200] + "..."
        parts.append(profile_text)
        parts.append("")

    # Structured preferences — detailed user preference signals (intent-gated)
    if include_user_profile and structured_preferences:
        parts.append("## Structured Preferences")
        pref_text = structured_preferences
        if len(pref_text) > 300:
            pref_text = pref_text[:300] + "..."
        parts.append(pref_text)
        parts.append("")

    # Available tools/capabilities (so LLM knows what already exists)
    if tool_names:
        parts.append("## Available Tools")
        for name in tool_names:
            parts.append(f"- {name}")
        parts.append("")

    # Evolved capabilities (permanent skills — compact one-line list)
    if permanent_capabilities:
        parts.append("## Capabilities (do NOT duplicate)")
        cap_items = []
        for cap in permanent_capabilities:
            name = cap.get("name", "?")
            usage = cap.get("usage_count", 0)
            cap_items.append(f"{name}({usage}x)")
        parts.append(", ".join(cap_items))
        parts.append("")

    # Allowed packages for capability proposals
    if allowed_packages:
        parts.append("## Allowed Packages for Capability Skills")
        parts.append(f"You may use: {', '.join(sorted(allowed_packages))}")
        parts.append("Do NOT propose packages outside this list.")
        parts.append("")

    # Skill coverage — tell LLM about task resolution effectiveness (intent-gated).
    if include_skill_coverage and skill_hit_summary and skill_hit_summary.get("total", 0) > 0:
        hit = skill_hit_summary
        parts.append("## Skill Coverage (recent tasks)")
        parts.append(
            f"{hit['skill']} of {hit['total']} tasks ({hit['ratio']:.0%}) "
            f"were resolved using existing skills."
        )
        if hit.get("top_skills"):
            top_str = ", ".join(f"{name} ({count}x)" for name, count in hit["top_skills"].items())
            parts.append(f"Most effective: {top_str}")
        if hit["ratio"] >= 0.7:
            parts.append(
                "Skills are covering most user needs. Focus evolution on "
                "NOVEL capabilities outside current skill coverage."
            )
        elif hit["ratio"] >= 0.3:
            parts.append(
                "Skills partially cover user needs. Consider what task types "
                "still require raw LLM reasoning and evolve toward those gaps."
            )
        parts.append("")

    # Existing skills — compact format
    if skills:
        parts.append("## Skills")
        used_items = []
        unused_count = 0
        for skill in skills[:15]:
            name = skill.get("name", "?")
            usage = skill.get("usage_count", 0)
            if usage > 0:
                used_items.append(f"{name}({usage}x)")
            else:
                unused_count += 1

        if used_items:
            parts.append(f"Used: {', '.join(used_items)}")
        if unused_count > 0:
            parts.append(f"Unused: {unused_count} skills — evolve toward uncovered domains.")
        parts.append("")

    # Inherited gene patterns from best past generations (intent-gated).
    if include_gene_pool and gene_pool:
        parts.append("## Inherited Patterns")
        for gene in gene_pool[:3]:
            gen = gene.get("generation", "?")
            score = gene.get("score", 0)
            task_hits = gene.get("task_hit_count", 0) or 0
            summary = gene.get("gene_summary", "")
            if len(summary) > 150:
                summary = summary[:147] + "..."
            parts.append(f"- [Gen {gen}, score={score:.2f}, task_hits={task_hits}] {summary}")
        parts.append("")

    # Recent crash logs — only on failure path or repair intent
    _is_repair = _intent == "repair"
    if crash_logs and (not survived or _is_repair):
        parts.append("## Recent Crashes")
        for log_entry in crash_logs[:2]:
            gen = log_entry.get("generation", "?")
            content = log_entry.get("content", "")
            parts.append(f"- Gen {gen}: {content[:500]}")
        parts.append("")

    # Evolution intent (structured) or legacy instructions (fallback)
    if evolution_intent:
        intent = evolution_intent.get("intent", "optimize")
        signals = evolution_intent.get("signals", [])

        parts.append(f"## Evolution Intent: {intent.upper()}")
        if intent == "repair":
            parts.append(
                "FIX the issues below. Do not add new features "
                "— focus on making the code survive."
            )
            for sig in signals:
                parts.append(f"- {sig}")
        elif intent == "explore":
            parts.append(
                "Scores have PLATEAUED. Evolve toward capabilities that better serve "
                "the user based on their profile and recent tasks. "
                "Do NOT explore randomly — focus on gaps between what the user needs "
                "and what the current code provides."
            )
        elif intent == "adapt":
            parts.append(
                "Follow the user directive below. Prioritize it above "
                "all other guidance."
            )
        else:  # optimize
            parts.append(
                "The code survived. Improve fitness: better output quality, "
                "novelty, or efficiency."
            )
    else:
        # Legacy fallback (no evolution_intent provided)
        parts.append("## Instructions")
        if is_plateaued:
            parts.append(
                "WARNING: Scores have PLATEAUED. The current approach is stagnant. "
                "You MUST try something fundamentally different — a new algorithm, "
                "a new domain, a new interaction pattern. Do NOT make incremental "
                "changes to the existing code. Start fresh with a novel idea."
            )
        elif survived:
            parts.append(
                "The previous code survived. Evolve it — try something genuinely "
                "NEW and different while keeping the heartbeat alive."
            )
        else:
            parts.append(
                "The previous code DIED (heartbeat lost). Fix the issue and make it "
                "more robust. Ensure the heartbeat loop runs reliably."
            )

    if directive:
        parts.append("")
        parts.append(f"## User Directive\n{directive}")
        parts.append("Prioritize this directive above all other guidance.")

    return SYSTEM_PROMPT, "\n".join(parts)


def extract_python_code(response: str) -> str | None:
    """Extract the first ```python code block from an LLM response.

    Returns None if no valid code block is found.
    """
    # Match ```python ... ``` blocks (non-greedy).
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


def extract_reflection(response: str) -> str | None:
    """Extract reflection text from an LLM response.

    Looks for text between ``## Reflection`` and the first
    ````` ```python ````` code fence.  Returns ``None`` if no reflection found.
    """
    pattern = r"## Reflection\s*\n(.*?)```python"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        text = match.group(1).strip()
        if text:
            return text
    return None


def extract_capability_proposal(response: str) -> dict | None:
    """Extract optional capability proposal from evolution LLM response.

    Looks for ```capability ... ``` block containing JSON.
    Returns parsed dict or None if no proposal.
    """
    pattern = r"```capability\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    # Validate required fields.
    required = {"name", "description", "dependencies", "source_code"}
    if not required.issubset(data.keys()):
        return None
    if not isinstance(data["dependencies"], list):
        return None
    return data


# ---------------------------------------------------------------------------
# Skill Crystallization
# ---------------------------------------------------------------------------

CRYSTALLIZE_SYSTEM_PROMPT = """\
You are the skill crystallization engine for Protea, a self-evolving artificial life system.

Your task: analyse Ring 2 source code that has successfully survived, and decide \
whether it represents a reusable *skill* worth preserving.

## What to ignore
- Heartbeat boilerplate (PROTEA_HEARTBEAT, write_heartbeat, heartbeat loop)
- Generic setup code (import os, pathlib, signal handling)
- Trivial programs that only maintain the heartbeat and do nothing else

## What to extract
Focus on the **core capability** — the interesting algorithm, interaction pattern, \
data processing, game logic, web server, visualisation, or other useful behaviour \
beyond the heartbeat.

## Decision
Compare the code's capability against the list of existing skills provided.
- **create**: The code demonstrates a genuinely new capability not covered by any \
existing skill.
- **update**: The code is an improved or extended version of an existing skill.
- **skip**: The existing skills already cover this capability, or the code is too \
trivial to crystallize.

## Response format
Respond with a single JSON object (no markdown fences, no extra text):

For create:
{"action": "create", "name": "skill_name_snake_case", "description": "One-sentence description", "prompt_template": "Core pattern description with key code snippets and algorithms", "tags": ["tag1", "tag2"]}

For update:
{"action": "update", "existing_name": "skill_name", "description": "Updated description", "prompt_template": "Updated core pattern", "tags": ["tag1", "tag2"]}

For skip:
{"action": "skip", "reason": "Brief explanation of why this was skipped"}
"""


def build_crystallize_prompt(
    source_code: str,
    output: str,
    generation: int,
    existing_skills: list[dict],
    skill_cap: int = 100,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the crystallization LLM call."""
    parts: list[str] = []

    parts.append(f"## Ring 2 Source (Generation {generation})")
    parts.append("```python")
    parts.append(source_code.rstrip())
    parts.append("```")
    parts.append("")

    if output:
        parts.append("## Program Output (last lines)")
        parts.append(output[-2000:])
        parts.append("")

    if existing_skills:
        parts.append("## Existing Skills")
        for skill in existing_skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            tags = skill.get("tags", [])
            parts.append(f"- {name}: {desc} (tags: {', '.join(tags) if tags else 'none'})")
        parts.append("")

    active_count = len(existing_skills)
    parts.append(f"## Capacity: {active_count}/{skill_cap} skills")
    if active_count >= skill_cap:
        parts.append("The skill store is FULL. Only create if this is clearly better than the least-used existing skill.")
    parts.append("")

    parts.append("Respond with a single JSON object.")

    return CRYSTALLIZE_SYSTEM_PROMPT, "\n".join(parts)


_VALID_ACTIONS = {"create", "update", "skip"}


def parse_crystallize_response(response: str) -> dict | None:
    """Parse the JSON response from the crystallization LLM call.

    Handles optional markdown code-block wrappers. Returns None on
    parse failure or invalid action.
    """
    text = response.strip()
    # Strip markdown code fences if present.
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("action") not in _VALID_ACTIONS:
        return None
    return data


# ---------------------------------------------------------------------------
# Memory Curation
# ---------------------------------------------------------------------------

MEMORY_CURATION_SYSTEM_PROMPT = """\
You are the memory curator for Protea, a self-evolving AI system.
Your task: review memory entries and decide which to keep, discard, summarize, or extract_rule.

## Decision criteria
- keep: Unique insights, user preferences, important lessons, recurring patterns
- summarize: Valuable but verbose — condense to 1-2 sentences
- discard: Redundant, outdated, trivial, or superseded by newer memories
- extract_rule: When you see 2+ related entries forming a pattern, distill into a reusable rule/preference. Return: {"id": [1, 2, 3], "action": "extract_rule", "rule": "Concise rule text"}

## Response format
Respond with a JSON array (no markdown fences):
[{"id": 1, "action": "keep"}, {"id": 2, "action": "summarize", "summary": "..."}, ...]
"""


def build_memory_curation_prompt(candidates: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_message) for memory curation.

    Args:
        candidates: List of dicts with id, entry_type/type, content, importance.

    Returns:
        (system_prompt, user_message) tuple.
    """
    parts = ["## Memory Entries to Review\n"]
    for c in candidates:
        entry_id = c.get("id", "?")
        entry_type = c.get("entry_type", c.get("type", "unknown"))
        content = c.get("content", "")
        importance = c.get("importance", 0.5)
        # Truncate long content.
        if len(content) > 200:
            content = content[:200] + "..."
        parts.append(
            f"- **ID {entry_id}** [{entry_type}] (importance: {importance:.2f}): {content}"
        )
    parts.append("")
    parts.append(f"Total: {len(candidates)} entries. Review each and decide: keep, discard, summarize, or extract_rule.")

    return MEMORY_CURATION_SYSTEM_PROMPT, "\n".join(parts)
