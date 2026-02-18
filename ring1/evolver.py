"""Evolution engine — orchestrates LLM-driven Ring 2 mutations.

Reads current Ring 2 code, queries fitness history, builds prompts,
calls Claude API, validates the result, and writes the mutated code.
Pure stdlib.
"""

from __future__ import annotations

import logging
import pathlib
from typing import NamedTuple

from ring1.llm_base import LLMClient, LLMError
from ring1.prompts import build_evolution_prompt, extract_python_code, extract_reflection

log = logging.getLogger("protea.evolver")

_REQUIRED_PATTERNS = ("PROTEA_HEARTBEAT", "write_heartbeat", "def main")


class EvolutionResult(NamedTuple):
    success: bool
    reason: str
    new_source: str  # empty string on failure
    metadata: dict = {}  # {"intent": ..., "signals": [...], "blast_radius": {...}}


def validate_ring2_code(source: str) -> tuple[bool, str]:
    """Pre-deployment validation of mutated Ring 2 code.

    Checks:
    1. Compiles without syntax errors
    2. Contains heartbeat mechanism (PROTEA_HEARTBEAT)
    3. Has a main() function
    """
    # 1. Syntax check.
    try:
        compile(source, "<ring2>", "exec")
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # 2. Heartbeat check — must reference PROTEA_HEARTBEAT.
    if "PROTEA_HEARTBEAT" not in source:
        return False, "Missing PROTEA_HEARTBEAT reference"

    # 3. Must define main().
    if "def main" not in source:
        return False, "Missing main() function"

    return True, "OK"


class Evolver:
    """Orchestrates a single evolution step for Ring 2."""

    def __init__(self, config, fitness_tracker, memory_store=None) -> None:
        """
        Args:
            config: Ring1Config with API credentials.
            fitness_tracker: FitnessTracker instance for history queries.
            memory_store: Optional MemoryStore for experiential memories.
        """
        self.config = config
        self.fitness = fitness_tracker
        self.memory_store = memory_store
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = self.config.get_llm_client()
        return self._client

    def evolve(
        self,
        ring2_path: pathlib.Path,
        generation: int,
        params: dict,
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
    ) -> EvolutionResult:
        """Run one evolution cycle.

        1. Read current ring2/main.py
        2. Query fitness history
        3. Build prompt (with memories, persistent errors, plateau status)
        4. Call Claude API
        5. Extract reflection + store in memory
        6. Extract + validate code
        7. Write new ring2/main.py

        Returns EvolutionResult indicating success/failure.
        """
        main_py = ring2_path / "main.py"

        # 1. Read current source.
        try:
            current_source = main_py.read_text()
        except FileNotFoundError:
            return EvolutionResult(False, "ring2/main.py not found", "")

        # 2. Query fitness history (compact: 5 entries to save tokens).
        history_limit = min(self.config.max_prompt_history, 5)
        fitness_history = self.fitness.get_history(limit=history_limit)
        best_performers = self.fitness.get_best(n=3)

        # 3. Build prompt.
        system_prompt, user_message = build_evolution_prompt(
            current_source=current_source,
            fitness_history=fitness_history,
            best_performers=best_performers,
            params=params,
            generation=generation,
            survived=survived,
            directive=directive,
            memories=memories,
            task_history=task_history,
            skills=skills,
            crash_logs=crash_logs,
            persistent_errors=persistent_errors,
            is_plateaued=is_plateaued,
            gene_pool=gene_pool,
            evolution_intent=evolution_intent,
            user_profile_summary=user_profile_summary,
        )

        # 4. Call Claude API.
        try:
            client = self._get_client()
            response = client.send_message(system_prompt, user_message)
        except LLMError as exc:
            log.error("LLM call failed: %s", exc)
            return EvolutionResult(False, f"LLM error: {exc}", "")

        # 5. Extract reflection and store in memory.
        reflection = extract_reflection(response)
        if reflection and self.memory_store:
            try:
                self.memory_store.add(generation, "reflection", reflection)
                log.debug("Stored reflection for gen-%d", generation)
            except Exception:
                log.debug("Failed to store reflection", exc_info=True)

        # 6. Extract code.
        new_source = extract_python_code(response)
        if new_source is None:
            log.error("No Python code block found in LLM response")
            return EvolutionResult(False, "No code block in response", "")

        # 7. Validate.
        valid, reason = validate_ring2_code(new_source)
        if not valid:
            log.error("Validation failed: %s", reason)
            return EvolutionResult(False, f"Validation: {reason}", "")

        # 8. Write.
        main_py.write_text(new_source)
        log.info("Evolution gen-%d: new code written (%d bytes)", generation, len(new_source))

        # 9. Compute blast radius and build metadata.
        from ring0.evolution_intent import compute_blast_radius

        blast_radius = compute_blast_radius(current_source, new_source)
        metadata = (
            {**evolution_intent, "blast_radius": blast_radius}
            if evolution_intent
            else {"blast_radius": blast_radius}
        )

        # Store evolution intent in memory.
        if self.memory_store and evolution_intent:
            try:
                self.memory_store.add(
                    generation,
                    "evolution_intent",
                    f"{evolution_intent['intent']}: {', '.join(evolution_intent['signals'])}",
                    metadata=metadata,
                )
            except Exception:
                log.debug("Failed to store evolution intent", exc_info=True)

        return EvolutionResult(True, "OK", new_source, metadata)
