"""Evolution and crystallization prompt templates for Ring 1.

Builds system + user prompts for Claude to mutate Ring 2 code.
Extracts Python code blocks from LLM responses.
Crystallization: analyse surviving Ring 2 code and extract reusable skills.
"""

from __future__ import annotations

import json
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

## Evolution Strategy
Beyond the heartbeat constraint, evolve the code to be USEFUL to the user.
Prioritize capabilities that align with the user's recent tasks and needs.
If no task history is available, explore interesting computational abilities.
The code can:
- Compute things (math, fractals, cellular automata, simulations)
- Generate data (sequences, patterns, art in text)
- Explore algorithms (sorting, searching, optimization)
- Build data structures
- Implement games or puzzles
- Do file I/O (within the ring2 directory)
- Anything else that's interesting and runs with pure stdlib

Refer to user task history (if provided) to guide evolution direction.
Avoid duplicating existing skills — develop complementary capabilities.

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

## Response Format
Start with a SHORT reflection (1-2 sentences max), then the complete code.
Keep the reflection brief — the code is what matters.

## Reflection
[1-2 sentences: what pattern you noticed and your mutation strategy]

```python
# your complete mutated code here
```
"""


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
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the evolution LLM call."""
    parts: list[str] = []

    parts.append(f"## Generation {generation}")
    parts.append(f"Previous generation {'SURVIVED' if survived else 'DIED'}.")
    parts.append(f"Mutation rate: {params.get('mutation_rate', 0.1)}")
    parts.append(f"Max runtime: {params.get('max_runtime_sec', 60)}s")
    parts.append("")

    # Current source code
    parts.append("## Current Ring 2 Code")
    parts.append("```python")
    parts.append(current_source.rstrip())
    parts.append("```")
    parts.append("")

    # Fitness history (compact — limit to 5 to save tokens)
    if fitness_history:
        parts.append("## Recent Fitness History")
        for entry in fitness_history[:5]:
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

    # Persistent errors — MUST FIX (high priority)
    if persistent_errors:
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

    # Recent user tasks — guide evolution direction
    if task_history:
        parts.append("## Recent User Tasks (guide your evolution!)")
        for task in task_history[:5]:
            content = task.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            parts.append(f"- {content}")
        parts.append("Evolve capabilities directly useful for these tasks.")
        parts.append("")

    # User profile — aggregated interests and directions
    if user_profile_summary:
        parts.append("## User Profile")
        parts.append(user_profile_summary)
        parts.append("Align evolution with the user's primary interests and active domains.")
        parts.append("")

    # Existing skills — avoid duplication + highlight unused ones
    if skills:
        parts.append("## Existing Skills")
        used_skills = []
        unused_skills = []
        for skill in skills[:15]:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            usage = skill.get("usage_count", 0)
            if usage > 0:
                used_skills.append(f"- {name}: {desc} (used {usage}x)")
            else:
                unused_skills.append(name)

        if used_skills:
            parts.append("### Popular (avoid duplicating):")
            parts.extend(used_skills)
        if unused_skills:
            parts.append(f"### Never used: {', '.join(unused_skills)}")
            parts.append("Consider exploring DIFFERENT directions from these unused skills.")
        parts.append("")

    # Inherited gene patterns from best past generations.
    if gene_pool:
        parts.append("## Inherited Patterns (matched to current context)")
        parts.append("Reuse or build upon these proven code patterns:")
        for gene in gene_pool[:3]:
            gen = gene.get("generation", "?")
            score = gene.get("score", 0)
            summary = gene.get("gene_summary", "")
            if len(summary) > 300:
                summary = summary[:297] + "..."
            parts.append(f"- [Gen {gen}, score={score:.2f}] {summary}")
        parts.append("")

    # Recent crash logs (compact)
    if crash_logs:
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
                "Scores have PLATEAUED. Try something fundamentally different "
                "— new algorithm, new domain."
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
Your task: review memory entries and decide which to keep, discard, or summarize.

## Decision criteria
- keep: Unique insights, user preferences, important lessons, recurring patterns
- summarize: Valuable but verbose — condense to 1-2 sentences
- discard: Redundant, outdated, trivial, or superseded by newer memories

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
    parts.append(f"Total: {len(candidates)} entries. Review each and decide: keep, discard, or summarize.")

    return MEMORY_CURATION_SYSTEM_PROMPT, "\n".join(parts)
