"""Periodic task template synchronization with the Hub.

Handles two-phase sync:
1. **Publish** — push quality scheduled tasks (run_count >= 2) as templates.
2. **Discover** — search the Hub for relevant templates based on user profile.

Designed to be called periodically (e.g. every 2 hours) from the sentinel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ring0.scheduled_task_store import ScheduledTaskStore
    from ring0.user_profile import UserProfiler
    from ring1.registry_client import RegistryClient

log = logging.getLogger("protea.task_sync")


class TaskSyncer:
    """Two-phase task template synchronization between local store and Hub."""

    def __init__(
        self,
        scheduled_store: ScheduledTaskStore,
        registry_client: RegistryClient,
        user_profiler: UserProfiler | None = None,
        max_discover: int = 5,
    ) -> None:
        self.scheduled_store = scheduled_store
        self.registry = registry_client
        self.profiler = user_profiler
        self.max_discover = max_discover
        # Track discovered templates for dashboard display.
        self.discovered_templates: list[dict] = []

    def sync(self) -> dict:
        """Run a full sync cycle: publish then discover.

        Returns a summary dict with counts.
        """
        result = {"published": 0, "discovered": 0, "errors": 0}

        # Phase 1: Publish quality scheduled tasks as templates.
        try:
            result["published"] = self._publish_templates()
        except Exception:
            log.debug("Publish phase failed", exc_info=True)
            result["errors"] += 1

        # Phase 2: Discover relevant templates from Hub.
        try:
            result["discovered"] = self._discover_templates()
        except Exception:
            log.debug("Discover phase failed", exc_info=True)
            result["errors"] += 1

        return result

    def _publish_templates(self) -> int:
        """Publish unpublished quality tasks as templates to the Hub."""
        candidates = self.scheduled_store.get_publishable(min_runs=2)
        published = 0

        for task in candidates:
            # Skip already-published.
            if task.get("published_template_hash"):
                continue

            template = self.scheduled_store.extract_template(task)
            resp = self.registry.publish_task_template(
                name=template["name"],
                task_text=template["task_text"],
                cron_expr=template["cron_expr"],
                schedule_type=template["schedule_type"],
                tags=template.get("tags"),
                template_hash=template["template_hash"],
            )
            if resp is not None:
                self.scheduled_store.mark_template_published(
                    task["schedule_id"], template["template_hash"],
                )
                published += 1
                log.info("Published task template: %s", template["name"])

        return published

    def _discover_templates(self) -> int:
        """Discover relevant templates from the Hub based on user profile."""
        queries = self._build_discovery_queries()
        if not queries:
            queries = ["daily"]  # default fallback

        seen_names: set[str] = set()
        # Don't re-discover what we already have locally.
        for task in self.scheduled_store.get_all():
            seen_names.add(task.get("name", ""))

        discovered = 0
        new_templates: list[dict] = []

        for query in queries:
            if discovered >= self.max_discover:
                break
            results = self.registry.search_task_templates(query=query, limit=10)
            for tmpl in results:
                if discovered >= self.max_discover:
                    break
                name = tmpl.get("name", "")
                if name in seen_names:
                    continue
                seen_names.add(name)
                new_templates.append(tmpl)
                discovered += 1

        self.discovered_templates = new_templates
        return discovered

    def _build_discovery_queries(self) -> list[str]:
        """Build search queries from user profile interests."""
        if not self.profiler:
            return []
        try:
            profile = self.profiler.get_profile()
            interests = profile.get("interests", [])
            if isinstance(interests, list):
                return interests[:3]
        except Exception:
            pass
        return []
