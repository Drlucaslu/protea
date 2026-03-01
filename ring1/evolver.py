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
from ring1.prompts import build_evolution_prompt, extract_python_code, extract_reflection, extract_capability_proposal

log = logging.getLogger("protea.evolver")


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
        structured_preferences: str = "",
        tool_names: list[str] | None = None,
        permanent_capabilities: list[dict] | None = None,
        allowed_packages: list[str] | None = None,
        skill_hit_summary: dict | None = None,
        semantic_rules: list[dict] | None = None,
        evolution_direction: str = "",
        accepted_capabilities: list[dict] | None = None,
        rejected_directions: list[dict] | None = None,
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

        # 2. Query fitness history.
        history_limit = min(self.config.max_prompt_history, 8)
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
            structured_preferences=structured_preferences,
            tool_names=tool_names,
            permanent_capabilities=permanent_capabilities,
            allowed_packages=allowed_packages,
            skill_hit_summary=skill_hit_summary,
            semantic_rules=semantic_rules,
            evolution_direction=evolution_direction,
            accepted_capabilities=accepted_capabilities,
            rejected_directions=rejected_directions,
        )

        # 4. Call LLM API (with truncation detection).
        try:
            client = self._get_client()
            response, meta = client.send_message_ex(system_prompt, user_message)
            llm_usage = client.last_usage
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

        # 5b. Extract optional capability proposal.
        capability_proposal = extract_capability_proposal(response)
        if capability_proposal:
            deps = capability_proposal.get("dependencies", [])
            if deps:
                log.info("Capability proposal detected: %s (deps: %s)",
                         capability_proposal.get("name"), deps)
            else:
                log.info("Capability proposal detected: %s (no deps — stdlib only)",
                         capability_proposal.get("name"))

        # 6. Extract code (with compact retry on truncation or missing code).
        new_source = extract_python_code(response)
        if new_source is None:
            reason = meta.get("stop_reason", "unknown")
            log.warning(
                "Evolution produced no code (stop_reason=%s), "
                "retrying with compact prompt",
                reason,
            )
            system_compact, user_compact = build_evolution_prompt(
                current_source=current_source,
                fitness_history=fitness_history,
                best_performers=best_performers,
                params=params,
                generation=generation,
                survived=survived,
                directive=directive,
                compact_mode=True,
            )
            try:
                response, meta = client.send_message_ex(
                    system_compact, user_compact,
                )
                retry_usage = client.last_usage
                llm_usage = {
                    "input_tokens": llm_usage["input_tokens"] + retry_usage["input_tokens"],
                    "output_tokens": llm_usage["output_tokens"] + retry_usage["output_tokens"],
                }
            except LLMError as exc:
                log.error("Compact retry LLM call failed: %s", exc)
                return EvolutionResult(False, f"LLM error on compact retry: {exc}", "")
            new_source = extract_python_code(response)
            if new_source is None:
                log.error(
                    "Evolution failed even with compact prompt "
                    "(stop_reason=%s, first 500 chars): %s",
                    meta.get("stop_reason"), response[:500],
                )
                return EvolutionResult(
                    False, "No code block in response (even after compact retry)", "",
                )

        if new_source is None:
            log.error("No Python code block found in LLM response (first 500 chars): %s", response[:500])
            return EvolutionResult(False, "No code block in response", "")

        # 7. Validate.
        valid, reason = validate_ring2_code(new_source)
        if not valid:
            log.error("Validation failed: %s", reason)
            return EvolutionResult(False, f"Validation: {reason}", "")

        # 8. Compute blast radius and reject full rewrites for non-adapt intents.
        from ring0.evolution_intent import compute_blast_radius

        blast_radius = compute_blast_radius(current_source, new_source)
        _intent = (evolution_intent or {}).get("intent", "optimize")
        if blast_radius["scope"] == "full_rewrite" and _intent == "optimize":
            log.warning(
                "Rejected full rewrite (lines_changed=%d, intent=%s) — "
                "incremental changes preferred when code is working",
                blast_radius["lines_changed"], _intent,
            )
            return EvolutionResult(
                False,
                f"Rejected: full rewrite not allowed for intent={_intent}",
                "",
            )

        # 9. Write.
        main_py.write_text(new_source)
        log.info("Evolution gen-%d: new code written (%d bytes)", generation, len(new_source))
        metadata = (
            {**evolution_intent, "blast_radius": blast_radius}
            if evolution_intent
            else {"blast_radius": blast_radius}
        )
        metadata["llm_usage"] = llm_usage

        # Include capability proposal in metadata if present.
        if capability_proposal:
            metadata["capability_proposal"] = capability_proposal

        # Store evolution intent in memory (with fuzzy deduplication).
        if self.memory_store and evolution_intent:
            try:
                # Build the content string
                content = f"{evolution_intent['intent']}: {', '.join(evolution_intent['signals'])}"
                
                # Check for duplicates using fuzzy matching (85% similarity threshold)
                is_duplicate = self.memory_store.is_duplicate_content(
                    "evolution_intent",
                    content,
                    lookback=5,
                    similarity_threshold=0.85
                )
                
                if not is_duplicate:
                    self.memory_store.add(
                        generation,
                        "evolution_intent",
                        content,
                        metadata=metadata,
                    )
                else:
                    log.debug(f"Skipping duplicate evolution_intent: {content}")
            except Exception:
                log.debug("Failed to store evolution intent", exc_info=True)

        return EvolutionResult(True, "OK", new_source, metadata)
