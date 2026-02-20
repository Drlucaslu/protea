"""Multi-round Convergence Detector — detects correction→convergence patterns.

When the user repeatedly corrects the bot (wrong → try again → wrong → got it),
this detector clusters recent interactions by topic (embedding similarity),
counts negation/affirmation signals, and when convergence is detected, uses
an LLM to extract a reusable rule for user confirmation.

Pure stdlib + project dependencies (no pip packages).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time

from ring0.memory import _cosine_similarity

log = logging.getLogger("protea.convergence_detector")

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(不对|错了|不行|不是这样|重新|搞错|弄错|再试|不是我要的)"),
    re.compile(r"(wrong|incorrect|no[, ]+that'?s not|try again|not right|redo)", re.IGNORECASE),
]

_AFFIRMATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(对了|可以了|这次对了|好了|终于对了|这才对|没错)"),
    re.compile(r"(that'?s right|correct|yes[, ]+that'?s it|perfect|exactly|got it)", re.IGNORECASE),
]

_REINFORCEMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(以后都这样|记住这个|以后.*都.*这样|记住这个规则|以后别再)"),
    re.compile(r"(from now on|always do|remember this|never do.* again)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Rule extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_RULE_PROMPT = """\
You are a rule extractor. Analyze the following conversation where a user \
corrected the assistant multiple times before reaching the right answer.

Extract ONE concise rule that the assistant should follow in the future \
to avoid the same mistake. Use the user's language (Chinese or English).

Format your response EXACTLY as:
RULE: <the rule>
CONTEXT: <when this rule applies>

If no meaningful rule can be extracted, respond with exactly: NONE

Conversation:
{conversation}
"""

# ---------------------------------------------------------------------------
# ConvergenceDetector
# ---------------------------------------------------------------------------


class ConvergenceDetector:
    """Detect multi-round correction→convergence and extract rules."""

    def __init__(
        self,
        memory_store,
        embedding_provider,
        llm_client,
        convergence_context: dict,
        config: dict | None = None,
    ) -> None:
        self._memory_store = memory_store
        self._embedding_provider = embedding_provider
        self._llm_client = llm_client
        self._convergence_context = convergence_context  # shared with SentinelState

        cfg = config or {}
        self._cluster_window_sec: int = cfg.get("cluster_window_sec", 900)
        self._min_corrections: int = cfg.get("min_corrections", 2)
        self._similarity_threshold: float = cfg.get("similarity_threshold", 0.5)
        self._cooldown_sec: int = cfg.get("cooldown_sec", 600)
        self._context_expiry_sec: int = 1800  # 30 min auto-cleanup for pending rules

        # Internal conversation history (separate from executor's _chat_history).
        self._history: list[tuple[float, str, str]] = []
        self._history_max: int = 15
        self._history_ttl: int = self._cluster_window_sec  # same as cluster window

        self._last_trigger_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, user_text: str, response_text: str) -> None:
        """Append a conversation pair and prune expired entries."""
        now = time.time()
        self._history.append((now, user_text, response_text))
        # Prune by TTL and capacity.
        cutoff = now - self._history_ttl
        self._history = [
            (ts, u, r) for ts, u, r in self._history if ts >= cutoff
        ]
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]
        # Prune expired pending proposals.
        self._cleanup_expired_context(now)

    def check(self) -> tuple[str, list] | None:
        """Main entry: cluster → detect convergence → extract rule → return UI."""
        if len(self._history) < 3:
            return None

        # Cooldown check.
        now = time.time()
        if now - self._last_trigger_time < self._cooldown_sec:
            return None

        cluster = self._cluster_recent()
        if not cluster:
            return None

        if not self._detect_convergence(cluster):
            return None

        rule_text, context_text = self._extract_rule(cluster)
        if not rule_text:
            return None

        self._last_trigger_time = now

        # Build inline keyboard for Telegram.
        rule_key = hashlib.md5(rule_text.encode()).hexdigest()[:8]
        self._convergence_context[rule_key] = {
            "rule_text": rule_text,
            "context_text": context_text,
            "cluster_size": len(cluster),
            "timestamp": now,
        }

        message = (
            f"我注意到你经过 {len(cluster)} 轮纠正后收敛了一条规则：\n\n"
            f"📌 {rule_text}\n"
        )
        if context_text:
            message += f"📎 适用场景：{context_text}\n"
        message += "\n要保存这条规则吗？以后我会自动遵守。"

        buttons = [[
            {"text": "✅ 是，保存", "callback_data": f"convergence:confirm:{rule_key}"},
            {"text": "❌ 不保存", "callback_data": f"convergence:dismiss:{rule_key}"},
        ]]

        return message, buttons

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _cluster_recent(self) -> list[tuple[float, str, str]] | None:
        """Cluster recent history entries by embedding similarity.

        Uses the last entry as the anchor; entries with cosine similarity
        >= threshold are included in the cluster.
        """
        if len(self._history) < 3:
            return None

        # Build texts for embedding: combine user + response for each entry.
        texts = [f"{u} {r}" for _, u, r in self._history]

        try:
            embeddings = self._embedding_provider.embed(texts)
        except Exception:
            log.debug("Embedding failed during clustering", exc_info=True)
            return None

        if not embeddings or len(embeddings) != len(texts):
            return None

        anchor = embeddings[-1]
        cluster: list[tuple[float, str, str]] = []

        for i, (ts, u, r) in enumerate(self._history):
            sim = _cosine_similarity(anchor, embeddings[i])
            if sim >= self._similarity_threshold:
                cluster.append((ts, u, r))

        if len(cluster) < 3:
            return None

        return cluster

    def _detect_convergence(self, cluster: list[tuple[float, str, str]]) -> bool:
        """Check if cluster shows correction→convergence pattern.

        Requires: negation_count >= min_corrections AND
                  (affirmation_count >= 1 OR reinforcement_count >= 1).
        """
        negation_count = 0
        affirmation_count = 0
        reinforcement_count = 0

        for _, user_text, _ in cluster:
            for pat in _NEGATION_PATTERNS:
                if pat.search(user_text):
                    negation_count += 1
                    break
            for pat in _AFFIRMATION_PATTERNS:
                if pat.search(user_text):
                    affirmation_count += 1
                    break
            for pat in _REINFORCEMENT_PATTERNS:
                if pat.search(user_text):
                    reinforcement_count += 1
                    break

        return (
            negation_count >= self._min_corrections
            and (affirmation_count >= 1 or reinforcement_count >= 1)
        )

    def _extract_rule(self, cluster: list[tuple[float, str, str]]) -> tuple[str, str]:
        """Use LLM to extract a rule from the converged conversation.

        Returns (rule_text, context_text) or ("", "") on failure.
        """
        conversation = "\n".join(
            f"User: {u}\nAssistant: {r}" for _, u, r in cluster
        )
        prompt = _EXTRACT_RULE_PROMPT.format(conversation=conversation)

        try:
            result = self._llm_client.send_message(
                "You are a concise rule extractor.", prompt,
            )
        except Exception:
            log.debug("LLM rule extraction failed", exc_info=True)
            return "", ""

        result = result.strip()
        if result.upper() == "NONE" or not result:
            return "", ""

        # Parse RULE: and CONTEXT: lines.
        rule_text = ""
        context_text = ""
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("RULE:"):
                rule_text = line[5:].strip()
            elif line.upper().startswith("CONTEXT:"):
                context_text = line[8:].strip()

        return rule_text, context_text

    def _cleanup_expired_context(self, now: float) -> None:
        """Remove pending proposals older than _context_expiry_sec."""
        expired = [
            k for k, v in self._convergence_context.items()
            if now - v.get("timestamp", 0) > self._context_expiry_sec
        ]
        for k in expired:
            del self._convergence_context[k]
