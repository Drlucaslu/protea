"""Evolution prompt templates for Ring 1.

Builds system + user prompts for Claude to mutate Ring 2 code.
Extracts Python code blocks from LLM responses.
"""

from __future__ import annotations

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

## Freedom
Beyond the heartbeat constraint, you have COMPLETE FREEDOM to make the code do
anything interesting. Be creative! The code can:
- Compute things (math, fractals, cellular automata, simulations)
- Generate data (sequences, patterns, art in text)
- Explore algorithms (sorting, searching, optimization)
- Build data structures
- Implement games or puzzles
- Do file I/O (within the ring2 directory)
- Anything else that's interesting and runs with pure stdlib

## Fitness
Your code is scored on:
- Survival: Did it run for the full max_runtime without crashing? (primary)
- Behavioral diversity: Does it do something DIFFERENT from previous generations?

## Response Format
First, write a brief reflection (2-3 sentences) about what you've observed
and your strategy. Then provide the code.

## Reflection
[Your observations about patterns, what works/fails, and your strategy]

```python
# your mutated code here
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

    # Fitness history
    if fitness_history:
        parts.append("## Recent Fitness History")
        for entry in fitness_history[:10]:
            status = "SURVIVED" if entry.get("survived") else "DIED"
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f}, "
                f"runtime={entry.get('runtime_sec', 0):.1f}s, "
                f"{status}"
            )
        parts.append("")

    # Best performers
    if best_performers:
        parts.append("## Best Performers (by score)")
        for entry in best_performers[:5]:
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f}, "
                f"hash={entry.get('commit_hash', '?')[:8]}"
            )
        parts.append("")

    # Learned patterns from memory
    if memories:
        parts.append("## Learned Patterns (from memory)")
        for mem in memories:
            gen = mem.get("generation", "?")
            mtype = mem.get("entry_type", "?")
            content = mem.get("content", "")
            parts.append(f"- [Gen {gen}, {mtype}] {content}")
        parts.append("")

    # Instructions based on outcome
    if survived:
        parts.append("## Instructions")
        parts.append(
            "The previous code survived! Evolve it further — make it do something "
            "more interesting or complex while keeping the heartbeat alive. "
            "Be creative and try something NEW."
        )
    else:
        parts.append("## Instructions")
        parts.append(
            "The previous code DIED (heartbeat lost). Fix the issue and make it "
            "more robust. Ensure the heartbeat loop runs reliably. "
            "Then add interesting behavior on top."
        )

    if directive:
        parts.append("")
        parts.append("## User Directive")
        parts.append(
            f"The user has requested a specific direction for evolution: {directive}\n"
            "Prioritize this directive while still following all constraints above."
        )

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
