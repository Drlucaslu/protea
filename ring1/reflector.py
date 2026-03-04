"""Reflector — event-driven reflection system for Protea.

Implements the Reflexion architecture (NeurIPS 2023):
  Task Executor (Generator) → Reflector → Episodic Memory → Future task enhancement

Triggered after task batches or during idle periods.  Generates proposals
that are auto-executed (high confidence) or sent to Telegram for approval.
"""

from __future__ import annotations

import json
import logging
import time
from typing import NamedTuple

log = logging.getLogger("protea.reflector")


class ReflectionProposal(NamedTuple):
    category: str       # "ring2_patch", "memory_cleanup", "config_tune", "task_pattern"
    description: str    # human-readable description
    confidence: float   # 0.0-1.0
    dimensions: dict    # multi-dimensional scores
    action: dict        # machine-readable action spec
    evidence: list      # triggering evidence


# Per-category risk levels (Self-Reflecting Agent inspired).
CATEGORY_RISK = {
    "memory_cleanup": 0.2,
    "task_pattern": 0.3,
    "config_tune": 0.4,
    "ring2_patch": 0.7,
}

DIMENSION_WEIGHTS = {
    "evidence_strength": 0.35,
    "impact_scope": 0.25,
    "risk_level": 0.25,
    "reversibility": 0.15,
}

# Reflection system prompt.
_REFLECTION_SYSTEM_PROMPT = """\
You are the reflection engine for Protea, a self-evolving AI system.
Analyze task execution data and identify actionable improvements.

## Evaluation Dimensions
1. Task effectiveness: Were tasks completed correctly? Any failures/timeouts/retries?
2. Token efficiency: Are input/output tokens reasonable? Any waste?
3. System health: Ring 2 fitness trends, error patterns
4. Memory quality: Any contradictory/outdated memories? Is retrieval effective?

## Response Format
JSON array. Each proposal:
- category: "ring2_patch" | "memory_cleanup" | "config_tune" | "task_pattern"
- description: What to change and why (concise)
- confidence: 0.0-1.0 (based on evidence strength)
- evidence: [specific observations] (must have data support, no speculation)
- action: {category-specific execution spec}

Return [] when no improvements needed. Prefer no proposals over low-quality ones.
"""

_IDLE_REFLECTION_SYSTEM_PROMPT = """\
You are the reflection engine for Protea, a self-evolving AI system.
The system has been idle. Perform a deep review including memory organization.

## Review Areas
1. Memory health: contradictions, stale entries, organization opportunities
2. Pattern analysis: recurring task types, common tool sequences
3. Token trends: usage patterns over recent period
4. System configuration: any config values that should be tuned

## Response Format
Same as task reflection: JSON array of proposals.
Each must have: category, description, confidence, evidence, action.
Return [] if everything looks healthy.
"""


class Reflector:
    """Event-driven reflection engine implementing Reflexion + MemEvolve patterns."""

    def __init__(
        self,
        config,
        fitness_tracker,
        memory_store,
        skill_store=None,
        notifier=None,
        auto_confidence: float = 0.8,
    ):
        self.config = config
        self.fitness = fitness_tracker
        self.memory_store = memory_store
        self.skill_store = skill_store
        self.notifier = notifier
        self.auto_confidence = auto_confidence
        self._client = None
        self._consecutive_empty: int = 0
        self._consecutive_rejected: int = 0
        self._cooldown_multiplier: float = 1.0

    def _get_client(self):
        if self._client is None:
            from ring1.config import load_ring1_config
            import pathlib
            # Walk up from ring0 config to find project root.
            project_root = pathlib.Path(__file__).resolve().parent.parent
            r1_config = load_ring1_config(project_root)
            self._client = r1_config.get_llm_client()
        return self._client

    # === Reflexion Core Loop ===

    def reflect_after_task(self) -> list[ReflectionProposal]:
        """Post-task reflection (Reflexion Self-Reflection stage).

        Analyzes recent task metrics batch.
        """
        user_message = self._build_task_reflection_context()
        if not user_message:
            return []

        try:
            client = self._get_client()
            response = client.send_message(_REFLECTION_SYSTEM_PROMPT, user_message)
            proposals = self._parse_proposals(response)

            if not proposals:
                self._consecutive_empty += 1
                if self._consecutive_empty >= 3:
                    self._cooldown_multiplier = min(self._cooldown_multiplier * 2, 4.0)
                    log.info("Reflection cooldown extended: %.1fx (consecutive empty)", self._cooldown_multiplier)
            else:
                self._consecutive_empty = 0
                self._cooldown_multiplier = 1.0

            # Record LLM usage.
            usage = client.last_usage
            if usage.get("input_tokens"):
                try:
                    gen = self.fitness.get_max_generation()
                    self.fitness.record_llm_usage(
                        gen, "reflection",
                        usage["input_tokens"], usage["output_tokens"],
                    )
                except Exception:
                    pass
                log.info("Reflection tokens: in=%d out=%d proposals=%d",
                         usage["input_tokens"], usage["output_tokens"], len(proposals))

            return proposals
        except Exception as exc:
            log.error("Reflection failed: %s", exc)
            return []

    def reflect_on_idle(self) -> list[ReflectionProposal]:
        """Idle-period deep reflection including MemEvolve memory strategy review."""
        user_message = self._build_idle_reflection_context()
        if not user_message:
            return []

        try:
            client = self._get_client()
            response = client.send_message(_IDLE_REFLECTION_SYSTEM_PROMPT, user_message)
            proposals = self._parse_proposals(response)

            # Record LLM usage.
            usage = client.last_usage
            if usage.get("input_tokens"):
                try:
                    gen = self.fitness.get_max_generation()
                    self.fitness.record_llm_usage(
                        gen, "reflection_idle",
                        usage["input_tokens"], usage["output_tokens"],
                    )
                except Exception:
                    pass

            return proposals
        except Exception as exc:
            log.error("Idle reflection failed: %s", exc)
            return []

    def process_proposals(self, proposals: list[ReflectionProposal]) -> None:
        """Route proposals by confidence: auto-execute or notify for approval."""
        for proposal in proposals:
            if proposal.confidence >= self.auto_confidence:
                log.info("Auto-executing proposal: %s (conf=%.2f)",
                         proposal.description[:80], proposal.confidence)
                success = self.execute_proposal(proposal)
                outcome = "auto_applied" if success else "auto_failed"
            else:
                log.info("Proposal needs approval: %s (conf=%.2f)",
                         proposal.description[:80], proposal.confidence)
                if self.notifier:
                    try:
                        self.notifier.notify_error(
                            0,
                            f"[Reflection] {proposal.category}: {proposal.description}\n"
                            f"Confidence: {proposal.confidence:.0%}",
                        )
                    except Exception:
                        pass
                outcome = "pending"

            # Store in episodic memory (Reflexion core).
            self._store_reflection(proposal, outcome)

    def execute_proposal(self, proposal: ReflectionProposal) -> bool:
        """Execute a single proposal and return success status."""
        try:
            category = proposal.category
            action = proposal.action

            if category == "memory_cleanup":
                return self._execute_memory_cleanup(action)
            elif category == "task_pattern":
                # Task patterns are stored as reflection findings for future retrieval.
                return True
            elif category == "config_tune":
                log.info("Config tune proposal noted: %s", proposal.description[:80])
                return True  # Config changes are advisory only.
            elif category == "ring2_patch":
                log.info("Ring2 patch proposal noted: %s", proposal.description[:80])
                return True  # Ring2 patches need manual review.
            else:
                log.warning("Unknown proposal category: %s", category)
                return False
        except Exception as exc:
            log.error("Proposal execution failed: %s", exc)
            return False

    # === Episodic Memory Integration (Reflexion Core) ===

    def get_relevant_reflections(self, task_text: str) -> list[dict]:
        """Retrieve relevant historical reflections for a new task.

        Called by task_executor to augment task context (Reflexion core mechanism).
        """
        if not self.memory_store:
            return []

        try:
            # Get recent reflection findings.
            reflections = self.memory_store.get_by_type("reflection_finding", limit=10)
            if not reflections:
                return []

            # Simple keyword-based relevance filtering.
            task_lower = task_text.lower()
            task_words = set(task_lower.split())

            relevant = []
            for r in reflections:
                content = r.get("content", "").lower()
                meta = r.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, ValueError):
                        meta = {}

                # Skip rejected or failed reflections.
                outcome = meta.get("outcome", "")
                if outcome in ("user_rejected", "auto_failed"):
                    continue

                # Check keyword overlap.
                content_words = set(content.split())
                overlap = len(task_words & content_words)
                if overlap >= 2 or any(w in content for w in task_words if len(w) >= 4):
                    relevant.append(r)

            return relevant[:3]  # Limit to avoid bloating context.
        except Exception:
            log.debug("Reflection retrieval failed", exc_info=True)
            return []

    # === Multi-dimensional Scoring (Self-Reflecting Agent) ===

    def _score_proposal(self, raw: dict) -> ReflectionProposal:
        """Score a raw proposal across multiple dimensions."""
        category = raw.get("category", "task_pattern")
        evidence = raw.get("evidence", [])
        confidence_raw = raw.get("confidence", 0.5)

        # Evidence strength: based on number of evidence items.
        evidence_strength = min(len(evidence) / 3, 1.0)

        # Impact scope: category-dependent.
        impact_map = {
            "ring2_patch": 0.8,
            "config_tune": 0.5,
            "memory_cleanup": 0.4,
            "task_pattern": 0.6,
        }
        impact_scope = impact_map.get(category, 0.5)

        # Risk level (inverted — lower risk = higher score).
        risk = CATEGORY_RISK.get(category, 0.5)
        risk_score = 1.0 - risk

        # Reversibility.
        reversibility_map = {
            "memory_cleanup": 0.7,
            "task_pattern": 1.0,
            "config_tune": 0.8,
            "ring2_patch": 0.3,
        }
        reversibility = reversibility_map.get(category, 0.5)

        dimensions = {
            "evidence_strength": evidence_strength,
            "impact_scope": impact_scope,
            "risk_level": risk_score,
            "reversibility": reversibility,
        }

        # Weighted confidence.
        confidence = sum(
            dimensions[dim] * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        # Blend with LLM's own confidence assessment.
        confidence = 0.6 * confidence + 0.4 * confidence_raw
        confidence = max(0.0, min(1.0, confidence))

        return ReflectionProposal(
            category=category,
            description=raw.get("description", ""),
            confidence=round(confidence, 3),
            dimensions=dimensions,
            action=raw.get("action", {}),
            evidence=evidence,
        )

    # === Internal Helpers ===

    def _build_task_reflection_context(self) -> str:
        """Build context for post-task reflection from recent task metrics."""
        parts = []

        # Recent task metrics from memory.
        if self.memory_store:
            try:
                tasks = self.memory_store.get_by_type("task", limit=10)
                if len(tasks) < 3:
                    return ""  # Not enough data.

                parts.append("## Recent Task Metrics")
                for t in tasks[:8]:
                    meta = t.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except (json.JSONDecodeError, ValueError):
                            meta = {}
                    content = t.get("content", "")[:100]
                    duration = meta.get("duration_sec", "?")
                    in_tok = meta.get("input_tokens", 0)
                    out_tok = meta.get("output_tokens", 0)
                    tools = len(meta.get("tool_sequence", []))
                    skills = ",".join(meta.get("skills_used", [])) or "none"
                    parts.append(
                        f"- task={content} duration={duration}s "
                        f"in_tokens={in_tok} out_tokens={out_tok} "
                        f"tools={tools} skills={skills}"
                    )
                parts.append("")
            except Exception:
                pass

        # Recent fitness scores.
        if self.fitness:
            try:
                history = self.fitness.get_history(limit=5)
                if history:
                    parts.append("## Recent Fitness")
                    for h in history:
                        parts.append(
                            f"- gen={h.get('generation')} score={h.get('score', 0):.3f} "
                            f"survived={h.get('survived')}"
                        )
                    parts.append("")
            except Exception:
                pass

        # Token usage summary.
        if self.fitness:
            try:
                usage = self.fitness.get_llm_usage_summary()
                if usage.get("total_calls", 0) > 0:
                    parts.append("## Token Usage (24h)")
                    parts.append(
                        f"Total: {usage.get('total_input', 0)} in / "
                        f"{usage.get('total_output', 0)} out / "
                        f"{usage.get('total_calls', 0)} calls"
                    )
                    by_caller = usage.get("by_caller", {})
                    for caller, stats in by_caller.items():
                        parts.append(
                            f"  {caller}: {stats.get('input_tokens', 0)} in / "
                            f"{stats.get('output_tokens', 0)} out / "
                            f"{stats.get('calls', 0)} calls"
                        )
                    parts.append("")
            except Exception:
                pass

        # Persistent errors.
        if self.fitness:
            try:
                errors = self.fitness.get_recent_error_signatures(limit=3)
                if errors:
                    parts.append("## Persistent Errors")
                    for err in errors:
                        parts.append(f"- {err}")
                    parts.append("")
            except Exception:
                pass

        return "\n".join(parts) if parts else ""

    def _build_idle_reflection_context(self) -> str:
        """Build context for idle-period deep reflection."""
        parts = []

        # Include task reflection context.
        task_ctx = self._build_task_reflection_context()
        if task_ctx:
            parts.append(task_ctx)

        # Memory statistics.
        if self.memory_store:
            try:
                stats = self.memory_store.get_stats()
                parts.append("## Memory Statistics")
                parts.append(f"Total entries: {stats.get('total', 0)}")
                by_type = stats.get("by_type", {})
                for etype, count in by_type.items():
                    parts.append(f"  {etype}: {count}")
                parts.append("")
            except Exception:
                pass

        # Recent reflection outcomes for meta-reflection.
        if self.memory_store:
            try:
                past_reflections = self.memory_store.get_by_type("reflection_finding", limit=5)
                if past_reflections:
                    parts.append("## Past Reflection Outcomes")
                    for r in past_reflections:
                        content = r.get("content", "")[:100]
                        meta = r.get("metadata", {})
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except (json.JSONDecodeError, ValueError):
                                meta = {}
                        outcome = meta.get("outcome", "?")
                        parts.append(f"- [{outcome}] {content}")
                    parts.append("")
            except Exception:
                pass

        return "\n".join(parts) if parts else ""

    def _parse_proposals(self, response: str) -> list[ReflectionProposal]:
        """Parse LLM response into scored proposals."""
        # Strip markdown code fences.
        text = response.strip()
        import re
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            log.debug("Failed to parse reflection response as JSON")
            return []

        if not isinstance(data, list):
            return []

        proposals = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            if raw.get("category") not in CATEGORY_RISK:
                continue
            if not raw.get("description"):
                continue
            proposal = self._score_proposal(raw)
            proposals.append(proposal)

        return proposals

    def _store_reflection(self, proposal: ReflectionProposal, outcome: str) -> None:
        """Store reflection result in episodic memory."""
        if not self.memory_store:
            return

        try:
            gen = self.fitness.get_max_generation() if self.fitness else 0
            self.memory_store.add(
                generation=gen,
                entry_type="reflection_finding",
                content=f"{proposal.category}: {proposal.description}",
                metadata={
                    "confidence": proposal.confidence,
                    "dimensions": proposal.dimensions,
                    "evidence": proposal.evidence,
                    "outcome": outcome,
                    "effect": None,
                },
            )
        except Exception:
            log.debug("Failed to store reflection finding", exc_info=True)

    def _execute_memory_cleanup(self, action: dict) -> bool:
        """Execute a memory cleanup action."""
        if not self.memory_store:
            return False

        action_type = action.get("type", "")
        if action_type == "delete_stale":
            entry_ids = action.get("entry_ids", [])
            if not entry_ids:
                return False
            try:
                for eid in entry_ids[:10]:  # Safety limit.
                    self.memory_store.delete(eid)
                log.info("Memory cleanup: deleted %d stale entries", len(entry_ids))
                return True
            except Exception:
                return False
        elif action_type == "resolve_contradiction":
            keep_id = action.get("keep_id")
            delete_id = action.get("delete_id")
            if keep_id and delete_id:
                try:
                    self.memory_store.delete(delete_id)
                    log.info("Resolved contradiction: kept %s, deleted %s", keep_id, delete_id)
                    return True
                except Exception:
                    return False
        return False
