"""Prompt quality helpers: contradiction detection, context validation, token estimation."""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Keywords that indicate negation / opposition.
_NEGATION_WORDS = frozenset({
    "不", "没", "无", "别", "非", "未", "勿", "莫",
    "not", "no", "never", "don't", "doesn't", "shouldn't", "won't",
    "avoid", "stop", "remove", "disable", "without",
})

_SPLIT_RE = re.compile(r"[\s,;，；。！？!?]+")


def _extract_key_terms(text: str) -> set[str]:
    """Extract lowercase key terms from text for overlap comparison."""
    tokens = _SPLIT_RE.split(text.lower().strip())
    return {t for t in tokens if len(t) > 1}


def _has_negation(text: str) -> bool:
    """Check if text contains negation words."""
    tokens = _SPLIT_RE.split(text.lower().strip())
    return bool(_NEGATION_WORDS & set(tokens))


def check_memory_contradictions(memories: list[dict]) -> list[str]:
    """Check memory entries for potential contradictions.

    Uses keyword overlap + negation detection (no LLM).
    Returns list of warning strings for contradictory pairs.
    """
    warnings: list[str] = []
    if not memories or len(memories) < 2:
        return warnings

    # Group by rough topic (first 3 key terms).
    indexed: list[tuple[int, str, set[str], bool]] = []
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        terms = _extract_key_terms(content)
        neg = _has_negation(content)
        indexed.append((i, content[:80], terms, neg))

    # O(n^2) but n is small (typically < 20 memories in context).
    for i in range(len(indexed)):
        for j in range(i + 1, len(indexed)):
            idx_i, desc_i, terms_i, neg_i = indexed[i]
            idx_j, desc_j, terms_j, neg_j = indexed[j]

            # Need topic overlap.
            overlap = terms_i & terms_j
            if len(overlap) < 2:
                continue

            # Contradiction: same topic but one has negation and other doesn't.
            if neg_i != neg_j:
                warnings.append(
                    f"Potential contradiction: [{desc_i}] vs [{desc_j}] "
                    f"(shared terms: {', '.join(sorted(overlap)[:5])})"
                )

    return warnings


def validate_context(context_parts: list[str]) -> tuple[list[str], list[str]]:
    """Validate context parts before sending to LLM.

    Returns (filtered_parts, warnings).
    Filters out parts that are empty or excessively long.
    """
    filtered: list[str] = []
    warnings: list[str] = []

    for part in context_parts:
        if not part or not part.strip():
            continue

        token_est = estimate_token_count(part)
        if token_est > 4000:
            warnings.append(
                f"Context part truncated ({token_est} est. tokens): "
                f"{part[:60]}..."
            )
            # Truncate to ~3000 tokens worth.
            filtered.append(part[:6000])
        else:
            filtered.append(part)

    return filtered, warnings


def estimate_token_count(text: str) -> int:
    """Fast token estimation for mixed Chinese/English text.

    Approximation: ~1.3 tokens per word for English, ~2 tokens per CJK char.
    """
    if not text:
        return 0

    cjk_count = 0
    ascii_words = 0
    in_word = False

    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs range.
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            cjk_count += 1
            in_word = False
        elif ch.isalnum() or ch == '_':
            if not in_word:
                ascii_words += 1
                in_word = True
        else:
            in_word = False

    return int(cjk_count * 2 + ascii_words * 1.3)
