"""Context fragmentation — embedding-based relevance selection for task prompts.

Splits task context into tagged fragments, ranks them by relevance to the
current task, and assembles a prompt within a token budget.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Rough token estimate: ~4 chars per token for mixed CJK/English.
_CHARS_PER_TOKEN = 4

# Section ordering for final assembly.
_SECTION_ORDER = [
    "state", "history", "skills", "skills_list",
    "memories", "recalled", "rules", "strategies",
    "reflections", "ring2",
]


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Keyword-based category detection (imported from ring0.user_profile)
# ---------------------------------------------------------------------------

def _get_category_keywords() -> dict[str, set[str]]:
    """Import CATEGORY_KEYWORDS from user_profile, with fallback."""
    try:
        from ring0.user_profile import CATEGORY_KEYWORDS
        return CATEGORY_KEYWORDS
    except ImportError:
        return {
            "coding": {"python", "code", "debug", "function", "api", "git", "bug"},
            "data": {"csv", "json", "database", "sql", "pandas", "analysis"},
            "web": {"http", "html", "css", "scrape", "url", "browser"},
            "ai": {"model", "llm", "embedding", "prompt", "gpt", "claude"},
            "finance": {"stock", "portfolio", "investment", "trading", "market"},
        }


_WORD_RE = re.compile(r"[a-zA-Z\u4e00-\u9fff]+")


def _classify_text(text: str) -> str:
    """Return the best-matching category tag for text, or 'general'."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    best_cat = "general"
    best_score = 0
    for cat, keywords in _get_category_keywords().items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


def _task_category_set(task_text: str) -> set[str]:
    """Return all categories that have any keyword match with task text."""
    words = {w.lower() for w in _WORD_RE.findall(task_text)}
    cats = set()
    for cat, keywords in _get_category_keywords().items():
        if words & keywords:
            cats.add(cat)
    return cats


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """A tagged piece of context with metadata for ranking."""
    tag: str              # "global", "coding", "finance", etc.
    section: str          # "state", "ring2", "memories", etc.
    content: str          # Actual text content
    token_est: int = 0    # Estimated token count
    importance: float = 0.5  # Static weight 0.0-1.0
    embedding: list[float] | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.token_est == 0:
            self.token_est = _estimate_tokens(self.content)


# ---------------------------------------------------------------------------
# FragmentRegistry
# ---------------------------------------------------------------------------

class FragmentRegistry:
    """Collect, rank, select, and assemble context fragments."""

    def __init__(self, embedding_provider, token_budget: int = 3000):
        self.embedding_provider = embedding_provider
        self.token_budget = token_budget

    # -- collect --

    def collect(
        self,
        state_snapshot: dict,
        ring2_source: str,
        memories: list[dict] | None = None,
        skills: list[dict] | None = None,
        recommended_skills: list[dict] | None = None,
        other_skills: list[dict] | None = None,
        semantic_rules: list[dict] | None = None,
        strategies: list[dict] | None = None,
        reflections: list[dict] | None = None,
        recalled: list[dict] | None = None,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> list[Fragment]:
        """Split all context sources into Fragment objects."""
        fragments: list[Fragment] = []

        # 1. State (global, always included)
        from datetime import datetime
        state_lines = [
            "## Protea State",
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}",
            f"Generation: {state_snapshot.get('generation', '?')}",
            f"Alive: {state_snapshot.get('alive', '?')}",
            f"Paused: {state_snapshot.get('paused', '?')}",
            f"Last score: {state_snapshot.get('last_score', '?')}",
            f"Last survived: {state_snapshot.get('last_survived', '?')}",
        ]
        fragments.append(Fragment(
            tag="global", section="state",
            content="\n".join(state_lines),
            importance=1.0,
        ))

        # 2. Ring 2 source code
        if ring2_source:
            truncated = ring2_source[:500]
            if len(ring2_source) > 500:
                truncated += "\n... (truncated)"
            content = "## Ring 2 Code\n```python\n" + truncated + "\n```"
            fragments.append(Fragment(
                tag="coding", section="ring2",
                content=content,
                importance=0.4,
            ))

        # 3. Strategies
        if strategies:
            lines = ["", "## Proven Strategies"]
            for strat in strategies:
                c = strat.get("content", "")
                if len(c) > 200:
                    c = c[:197] + "..."
                lines.append(f"- {c}")
            tag = _classify_text("\n".join(lines))
            fragments.append(Fragment(
                tag=tag, section="strategies",
                content="\n".join(lines),
                importance=0.6,
            ))

        # 4. Recent memories
        if memories:
            lines = ["", "## Recent Learnings"]
            for mem in memories:
                gen = mem.get("generation", "?")
                content = mem.get("content", "")
                lines.append(f"- [Gen {gen}] {content}")
            tag = _classify_text("\n".join(lines))
            fragments.append(Fragment(
                tag=tag, section="memories",
                content="\n".join(lines),
                importance=0.7,
            ))

        # 5. Recommended skills
        if recommended_skills:
            lines = ["", "## Recommended Skills (use these first)"]
            for skill in recommended_skills:
                name = skill.get("name", "?")
                desc = skill.get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"- **{name}**: {desc}")
            fragments.append(Fragment(
                tag="global", section="skills",
                content="\n".join(lines),
                importance=0.9,
            ))

        # 6. Other skills (names list only)
        skill_list_source = other_skills or skills
        if skill_list_source:
            _MAX = 20
            names = [s.get("name", "?") for s in skill_list_source[:_MAX]]
            remaining = len(skill_list_source) - _MAX
            names_str = ", ".join(names)
            if remaining > 0:
                names_str += f" (and {remaining} more)"
            label = "Other skills" if recommended_skills else "Available skills"
            fragments.append(Fragment(
                tag="global", section="skills_list",
                content=f"\n{label}: {names_str}",
                importance=0.3,
            ))

        # 7. Chat history (global — conversation continuity)
        if chat_history:
            lines = ["", "## Recent Conversation"]
            for user_msg, assistant_msg in chat_history:
                u = user_msg[:500] + "..." if len(user_msg) > 500 else user_msg
                a = assistant_msg[:1000] + "..." if len(assistant_msg) > 1000 else assistant_msg
                lines.append(f"User: {u}")
                lines.append(f"Assistant: {a}")
                lines.append("")
            fragments.append(Fragment(
                tag="global", section="history",
                content="\n".join(lines),
                importance=0.8,
            ))

        # 8. Recalled memories
        if recalled:
            lines = ["", "## Recalled Memories"]
            for mem in recalled:
                gen = mem.get("generation", "?")
                content = mem.get("content", "")[:200]
                lines.append(f"- [Gen {gen}, archived] {content}")
            tag = _classify_text("\n".join(lines))
            fragments.append(Fragment(
                tag=tag, section="recalled",
                content="\n".join(lines),
                importance=0.6,
            ))

        # 9. Semantic rules
        if semantic_rules:
            lines = ["", "## Learned Patterns"]
            for rule in semantic_rules[:5]:
                content = rule.get("content", "")[:100]
                lines.append(f"- {content}")
            tag = _classify_text("\n".join(lines))
            fragments.append(Fragment(
                tag=tag, section="rules",
                content="\n".join(lines),
                importance=0.5,
            ))

        # 10. Reflections
        if reflections:
            lines = ["", "## Past Reflections (lessons from similar tasks)"]
            for ref in reflections[:3]:
                content = ref.get("content", "")[:200]
                lines.append(f"- {content}")
            tag = _classify_text("\n".join(lines))
            fragments.append(Fragment(
                tag=tag, section="reflections",
                content="\n".join(lines),
                importance=0.7,
            ))

        return fragments

    # -- rank --

    def rank(self, fragments: list[Fragment], task_text: str) -> list[tuple[Fragment, float]]:
        """Score each fragment's relevance to the task.

        Returns (fragment, score) pairs sorted by score descending.
        """
        # Compute task embedding
        task_embedding = None
        try:
            from ring1.embeddings import NoOpEmbedding
            if not isinstance(self.embedding_provider, NoOpEmbedding):
                vecs = self.embedding_provider.embed([task_text])
                task_embedding = vecs[0] if vecs else None
        except Exception:
            log.debug("Task embedding failed, using keyword fallback", exc_info=True)

        # Compute fragment embeddings in batch (skip globals)
        non_global = [f for f in fragments if f.tag != "global"]
        if task_embedding and non_global:
            try:
                texts = [f.content[:500] for f in non_global]
                embeddings = self.embedding_provider.embed(texts)
                for frag, emb in zip(non_global, embeddings):
                    frag.embedding = emb
            except Exception:
                log.debug("Fragment embedding failed", exc_info=True)

        task_cats = _task_category_set(task_text)

        scored: list[tuple[Fragment, float]] = []
        for frag in fragments:
            score = self._compute_relevance(task_embedding, task_cats, frag)
            scored.append((frag, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_relevance(
        self,
        task_embedding: list[float] | None,
        task_cats: set[str],
        fragment: Fragment,
    ) -> float:
        """Compute relevance score for a single fragment."""
        if fragment.tag == "global":
            return 1.0  # Always include

        if fragment.embedding is not None and task_embedding is not None:
            sim = _cosine_similarity(task_embedding, fragment.embedding)
            if sim > 0.01:  # Meaningful similarity — use embedding score
                return sim * fragment.importance
            # Near-zero similarity means embeddings are uninformative; fall through.

        # Keyword fallback
        return self._keyword_relevance(task_cats, fragment)

    def _keyword_relevance(self, task_cats: set[str], fragment: Fragment) -> float:
        """Fallback relevance when embeddings are unavailable."""
        if fragment.tag in task_cats:
            return 0.7 * fragment.importance
        return 0.2 * fragment.importance

    # -- select --

    def select(self, ranked: list[tuple[Fragment, float]]) -> list[Fragment]:
        """Greedily select fragments within token budget.

        1. Global fragments are always included.
        2. Remaining budget filled by score descending.
        """
        selected: list[Fragment] = []
        budget_remaining = self.token_budget

        # Pass 1: globals
        for frag, _score in ranked:
            if frag.tag == "global":
                selected.append(frag)
                budget_remaining -= frag.token_est

        # Pass 2: non-globals by score
        for frag, score in ranked:
            if frag.tag == "global":
                continue
            if budget_remaining <= 0:
                break
            if frag.token_est <= budget_remaining:
                selected.append(frag)
                budget_remaining -= frag.token_est

        return selected

    # -- assemble --

    def assemble(self, selected: list[Fragment]) -> str:
        """Join selected fragments in canonical section order."""
        section_map: dict[str, list[Fragment]] = {}
        for frag in selected:
            section_map.setdefault(frag.section, []).append(frag)

        parts: list[str] = []
        for section in _SECTION_ORDER:
            if section in section_map:
                for frag in section_map[section]:
                    parts.append(frag.content)

        # Any sections not in the ordering
        for section, frags in section_map.items():
            if section not in _SECTION_ORDER:
                for frag in frags:
                    parts.append(frag.content)

        return "\n".join(parts)
