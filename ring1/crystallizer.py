"""Skill crystallization engine — extracts reusable skills from surviving Ring 2 code.

Mirrors the structure of ring1/evolver.py.  Analyses surviving Ring 2 source
via an LLM call and persists the result in the SkillStore.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from ring1.llm_base import LLMClient, LLMError
from ring1.prompts import build_crystallize_prompt, parse_crystallize_response

log = logging.getLogger("protea.crystallizer")


class CrystallizationResult(NamedTuple):
    action: str      # "create" | "update" | "skip" | "error"
    skill_name: str
    reason: str
    llm_usage: dict = {}


class Crystallizer:
    """Orchestrates a single crystallization step for surviving Ring 2 code."""

    def __init__(self, config, skill_store) -> None:
        self.config = config
        self.skill_store = skill_store
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = self.config.get_llm_client()
        return self._client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def crystallize(
        self,
        source_code: str,
        output: str,
        generation: int,
        skill_cap: int = 100,
    ) -> CrystallizationResult:
        """Analyse *source_code* and create/update/skip a skill.

        1. Fetch existing skills from the store.
        2. Build the crystallization prompt.
        3. Call Claude API.
        4. Parse the JSON response.
        5. Execute the decided action.
        """
        # 1. Existing skills.
        try:
            existing = self.skill_store.get_active(limit=200)
        except Exception as exc:
            return CrystallizationResult("error", "", f"skill store read error: {exc}")

        # 2. Build prompt.
        system_prompt, user_message = build_crystallize_prompt(
            source_code=source_code,
            output=output,
            generation=generation,
            existing_skills=existing,
            skill_cap=skill_cap,
        )

        # 3. Call LLM.
        try:
            client = self._get_client()
            response = client.send_message(system_prompt, user_message)
            llm_usage = client.last_usage
        except LLMError as exc:
            return CrystallizationResult("error", "", f"LLM error: {exc}")

        # 4. Parse.
        data = parse_crystallize_response(response)
        if data is None:
            return CrystallizationResult("error", "", "failed to parse LLM response", llm_usage)

        # 5. Execute.
        action = data["action"]
        if action == "create":
            result = self._handle_create(data, source_code, skill_cap)
            return CrystallizationResult(result.action, result.skill_name, result.reason, llm_usage)
        if action == "update":
            result = self._handle_update(data, source_code)
            return CrystallizationResult(result.action, result.skill_name, result.reason, llm_usage)
        # skip
        return CrystallizationResult("skip", "", data.get("reason", ""), llm_usage)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_create(
        self, data: dict, source_code: str, skill_cap: int,
    ) -> CrystallizationResult:
        name = data.get("name", "")
        if not name:
            return CrystallizationResult("error", "", "missing skill name in create")

        # Duplicate name → redirect to update.
        if self.skill_store.get_by_name(name) is not None:
            log.info("Skill %r already exists — converting create to update", name)
            return self._handle_update(
                {**data, "existing_name": name, "action": "update"},
                source_code,
            )

        # Capacity check.
        if self.skill_store.count_active() >= skill_cap:
            least = self.skill_store.get_least_used(limit=1)
            if least:
                evicted = least[0]["name"]
                self.skill_store.deactivate(evicted)
                log.info("Evicted least-used skill %r to make room", evicted)

        self.skill_store.add(
            name=name,
            description=data.get("description", ""),
            prompt_template=data.get("prompt_template", ""),
            tags=data.get("tags"),
            source="crystallized",
            source_code=source_code,
        )
        log.info("Crystallized new skill %r", name)
        return CrystallizationResult("create", name, "OK")

    def _handle_update(
        self, data: dict, source_code: str,
    ) -> CrystallizationResult:
        name = data.get("existing_name", "")
        if not name:
            return CrystallizationResult("error", "", "missing existing_name in update")

        existing = self.skill_store.get_by_name(name)
        if existing is None:
            log.warning("Skill %r not found for update — skipping", name)
            return CrystallizationResult("skip", name, f"skill {name!r} not found")

        self.skill_store.update(
            name=name,
            description=data.get("description"),
            prompt_template=data.get("prompt_template"),
            tags=data.get("tags"),
            source_code=source_code,
        )
        log.info("Updated skill %r", name)
        return CrystallizationResult("update", name, "OK")
