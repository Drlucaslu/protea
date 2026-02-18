"""Periodic skill synchronization with the Hub.

Handles two-way sync:
1. **Publish** — push quality unpublished local skills to the Hub.
2. **Discover** — search the Hub for relevant skills based on user profile,
   download, validate, and install them locally.

Designed to be called periodically (e.g. every 2 hours) from the sentinel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ring0.skill_store import SkillStore
    from ring0.user_profile import UserProfiler
    from ring1.registry_client import RegistryClient

log = logging.getLogger("protea.skill_sync")


class SkillSyncer:
    """Two-way skill synchronization between local store and Hub."""

    def __init__(
        self,
        skill_store: SkillStore,
        registry_client: RegistryClient,
        user_profiler: UserProfiler | None = None,
        max_discover: int = 5,
    ) -> None:
        self.skill_store = skill_store
        self.registry = registry_client
        self.profiler = user_profiler
        self.max_discover = max_discover

    def sync(self) -> dict:
        """Run a full sync cycle: publish then discover.

        Returns a summary dict with counts.
        """
        result = {"published": 0, "discovered": 0, "rejected": 0, "errors": 0}

        # Phase 1: Publish unpublished quality skills.
        try:
            result["published"] = self._publish_unpublished()
        except Exception:
            log.debug("Publish phase failed", exc_info=True)
            result["errors"] += 1

        # Phase 2: Discover relevant skills from Hub.
        try:
            discovered, rejected = self._discover_relevant()
            result["discovered"] = discovered
            result["rejected"] = rejected
        except Exception:
            log.debug("Discover phase failed", exc_info=True)
            result["errors"] += 1

        return result

    # ------------------------------------------------------------------
    # Phase 1: Publish
    # ------------------------------------------------------------------

    def _publish_unpublished(self) -> int:
        """Publish quality local skills that haven't been pushed to the Hub."""
        unpublished = self.skill_store.get_unpublished(min_usage=2)
        if not unpublished:
            return 0

        published = 0
        for skill in unpublished:
            try:
                resp = self.registry.publish(
                    name=skill["name"],
                    description=skill.get("description", ""),
                    prompt_template=skill.get("prompt_template", ""),
                    parameters=skill.get("parameters"),
                    tags=skill.get("tags"),
                    source_code=skill.get("source_code", ""),
                )
                if resp is not None:
                    self.skill_store.mark_published(skill["name"])
                    published += 1
                    log.info("Sync: published skill %r to Hub", skill["name"])
            except Exception:
                log.debug("Failed to publish skill %r", skill["name"], exc_info=True)

        return published

    # ------------------------------------------------------------------
    # Phase 2: Discover
    # ------------------------------------------------------------------

    def _discover_relevant(self) -> tuple[int, int]:
        """Search Hub for relevant skills and install validated ones.

        Returns (discovered_count, rejected_count).
        """
        queries = self._build_search_queries()
        if not queries:
            return 0, 0

        local_names = self.skill_store.get_local_names()
        discovered = 0
        rejected = 0
        seen_names: set[str] = set()

        for query in queries:
            if discovered >= self.max_discover:
                break

            results = self.registry.search(query=query, limit=10)
            for skill_info in results:
                if discovered >= self.max_discover:
                    break

                name = skill_info.get("name", "")
                node_id = skill_info.get("node_id", "")
                if not name or not node_id:
                    continue

                # Skip already-installed or already-seen skills.
                if name in local_names or name in seen_names:
                    continue
                seen_names.add(name)

                # Skip own skills (already local).
                if node_id == self.registry.node_id:
                    continue

                # Download full skill data.
                skill_data = self.registry.download(node_id, name)
                if not skill_data:
                    continue

                # Validate security.
                source_code = skill_data.get("source_code", "")
                if not self._validate_skill(name, source_code):
                    rejected += 1
                    continue

                # Install locally.
                try:
                    self.skill_store.install_from_hub(skill_data)
                    discovered += 1
                    log.info(
                        "Sync: discovered and installed skill %r from %s",
                        name, node_id,
                    )
                except Exception:
                    log.debug("Failed to install skill %r", name, exc_info=True)

        return discovered, rejected

    def _build_search_queries(self) -> list[str]:
        """Build search queries from user profile topics."""
        if not self.profiler:
            return ["popular"]

        categories = self.profiler.get_category_distribution()
        if not categories:
            return ["popular"]

        # Use top 3 categories as search queries.
        queries: list[str] = []
        for category in list(categories.keys())[:3]:
            if category != "general":
                queries.append(category)

        # Also add top 3 specific topics for more targeted search.
        topics = self.profiler.get_top_topics(limit=5)
        for topic in topics[:3]:
            t = topic["topic"]
            if t not in queries and len(t) >= 4:
                queries.append(t)

        return queries or ["popular"]

    @staticmethod
    def _validate_skill(name: str, source_code: str) -> bool:
        """Run security validation on skill source code."""
        from ring1.skill_validator import validate_skill

        result = validate_skill(source_code)
        if not result.safe:
            log.warning(
                "Sync: rejected skill %r — security issues: %s",
                name, "; ".join(result.errors),
            )
            return False
        if result.warnings:
            log.info(
                "Sync: skill %r passed with warnings: %s",
                name, "; ".join(result.warnings),
            )
        return True
