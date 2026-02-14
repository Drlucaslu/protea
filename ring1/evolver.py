"""Evolution engine — orchestrates LLM-driven Ring 2 mutations.

Reads current Ring 2 code, queries fitness history, builds prompts,
calls Claude API, validates the result, and writes the mutated code.
Pure stdlib.
"""

from __future__ import annotations

import logging
import pathlib
from typing import NamedTuple

from ring1.llm_client import ClaudeClient, LLMError
from ring1.prompts import build_evolution_prompt, extract_python_code, extract_reflection

log = logging.getLogger("protea.evolver")

_REQUIRED_PATTERNS = ("PROTEA_HEARTBEAT", "write_heartbeat", "def main")


class EvolutionResult(NamedTuple):
    success: bool
    reason: str
    new_source: str  # empty string on failure


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
        self._client: ClaudeClient | None = None

    def _get_client(self) -> ClaudeClient:
        if self._client is None:
            self._client = ClaudeClient(
                api_key=self.config.claude_api_key,
                model=self.config.claude_model,
                max_tokens=self.config.claude_max_tokens,
            )
        return self._client

    def evolve(
        self,
        ring2_path: pathlib.Path,
        generation: int,
        params: dict,
        survived: bool,
        directive: str = "",
        memories: list[dict] | None = None,
    ) -> EvolutionResult:
        """Run one evolution cycle.

        1. Read current ring2/main.py
        2. Query fitness history
        3. Build prompt (with memories)
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

        # 2. Query fitness history.
        history_limit = self.config.max_prompt_history
        fitness_history = self.fitness.get_history(limit=history_limit)
        best_performers = self.fitness.get_best(n=5)

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
        return EvolutionResult(True, "OK", new_source)
