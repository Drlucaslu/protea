"""Periodic skill + gene synchronization with the Hub.

Handles multi-phase sync:
1. **Publish skills** — push quality unpublished local skills to the Hub.
2. **Discover skills** — search the Hub (+ external sources) for relevant skills.
3. **Publish genes** — push proven genes to the Hub.
4. **Discover genes** — search the Hub for useful genes and install locally.

Designed to be called periodically (e.g. every 2 hours) from the sentinel.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ring0.gene_pool import GenePool
    from ring0.skill_store import SkillStore
    from ring0.user_profile import UserProfiler
    from ring1.registry_client import RegistryClient
    from ring1.skill_sources import SkillSource

log = logging.getLogger("protea.skill_sync")


class SkillSyncer:
    """Multi-phase skill + gene synchronization between local store and Hub."""

    def __init__(
        self,
        skill_store: SkillStore,
        registry_client: RegistryClient,
        user_profiler: UserProfiler | None = None,
        max_discover: int = 5,
        allowed_packages: frozenset[str] | None = None,
        sources: list[SkillSource] | None = None,
        gene_pool: GenePool | None = None,
        embedding_provider=None,
    ) -> None:
        self.skill_store = skill_store
        self.registry = registry_client
        self.profiler = user_profiler
        self.max_discover = max_discover
        self.embedding_provider = embedding_provider
        self._allowed_packages = allowed_packages
        self._sources = sources or []
        self.gene_pool = gene_pool

    def sync(self) -> dict:
        """Run a full sync cycle: publish then discover (skills + genes).

        Returns a summary dict with counts.
        """
        result = {
            "published": 0, "discovered": 0, "rejected": 0, "errors": 0,
            "genes_published": 0, "genes_discovered": 0,
        }

        # Phase 1: Publish unpublished quality skills.
        try:
            result["published"] = self._publish_unpublished()
        except Exception:
            log.debug("Publish phase failed", exc_info=True)
            result["errors"] += 1

        # Phase 2: Discover relevant skills from Hub + external sources.
        try:
            discovered, rejected = self._discover_relevant()
            result["discovered"] = discovered
            result["rejected"] = rejected
        except Exception:
            log.debug("Discover phase failed", exc_info=True)
            result["errors"] += 1

        # Phase 3: Publish quality genes.
        try:
            result["genes_published"] = self._publish_genes()
        except Exception:
            log.debug("Gene publish phase failed", exc_info=True)
            result["errors"] += 1

        # Phase 4: Discover genes from Hub.
        try:
            result["genes_discovered"] = self._discover_genes()
        except Exception:
            log.debug("Gene discover phase failed", exc_info=True)
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
                    dependencies=skill.get("dependencies"),
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
        """Search Hub + external sources for relevant skills and install validated ones.

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

            results = self.registry.search(query=query, limit=10, order="gdi", min_downloads=1)
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

                # Validate security (code + dependencies).
                source_code = skill_data.get("source_code", "")
                if not self._validate_skill(name, source_code):
                    rejected += 1
                    continue

                dependencies = skill_data.get("dependencies", [])
                if dependencies and not self._validate_dependencies(name, dependencies):
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

        # External sources (ClawHub, Skills.sh, etc.).
        for source in self._sources:
            if discovered >= self.max_discover:
                break
            for query in queries:
                if discovered >= self.max_discover:
                    break
                try:
                    ext_results = source.search(query, limit=10)
                except Exception:
                    log.debug("Source %s search failed", source.name, exc_info=True)
                    continue
                for skill_info in ext_results:
                    if discovered >= self.max_discover:
                        break
                    name = skill_info.get("name", "")
                    if not name or name in local_names or name in seen_names:
                        continue
                    seen_names.add(name)

                    try:
                        skill_data = source.download(name)
                    except Exception:
                        log.debug("Source %s download failed for %r",
                                  source.name, name, exc_info=True)
                        continue
                    if not skill_data:
                        continue

                    source_code = skill_data.get("source_code", "")
                    if not self._validate_skill(name, source_code):
                        rejected += 1
                        continue
                    dependencies = skill_data.get("dependencies", [])
                    if dependencies and not self._validate_dependencies(name, dependencies):
                        rejected += 1
                        continue

                    try:
                        self.skill_store.install_from_hub(skill_data)
                        discovered += 1
                        log.info("Sync: discovered skill %r from %s",
                                 name, source.name)
                    except Exception:
                        log.debug("Failed to install skill %r from %s",
                                  name, source.name, exc_info=True)

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

    # ------------------------------------------------------------------
    # Phase 3 & 4: Gene sync
    # ------------------------------------------------------------------

    def _publish_genes(self) -> int:
        """Publish quality genes to Hub."""
        if not self.gene_pool:
            return 0
        publishable = self.gene_pool.get_publishable_genes()
        if not publishable:
            return 0

        from ring0.gene_pool import GenePool

        published = 0
        for gene in publishable:
            tags_str = gene.get("tags", "")
            name = GenePool.derive_gene_name(tags_str)
            tags = sorted(tags_str.split()) if tags_str else []
            try:
                resp = self.registry.publish_gene(
                    name=name,
                    gene_summary=gene.get("gene_summary", ""),
                    tags=tags,
                    score=gene.get("score", 0.0),
                    embedding=gene.get("embedding", ""),
                )
                if resp is not None:
                    published += 1
                    log.info("Sync: published gene %r to Hub", name)
            except Exception:
                log.debug("Failed to publish gene %r", name, exc_info=True)
        return published

    def _discover_genes(self) -> int:
        """Discover genes from Hub and install locally."""
        if not self.gene_pool:
            return 0

        queries = self._build_search_queries()
        if not queries:
            return 0

        discovered = 0
        seen: set[str] = set()

        from ring0.gene_pool import _cosine_similarity

        # Pre-load local gene source hashes to skip already-known content
        # regardless of node_id (handles hostname changes, re-publishing, etc.)
        local_hashes: set[str] = set()
        try:
            with self.gene_pool._connect() as con:
                for row in con.execute("SELECT source_hash FROM gene_pool").fetchall():
                    local_hashes.add(row["source_hash"])
        except Exception:
            pass

        for query in queries:
            if discovered >= self.max_discover:
                break

            # Build query embedding for semantic relevance filtering.
            query_emb = None
            if self.embedding_provider:
                try:
                    vecs = self.embedding_provider.embed([query])
                    query_emb = vecs[0] if vecs else None
                except Exception:
                    pass

            results = self.registry.search_genes(query=query, order="gdi", limit=10, min_score=0.5)
            for gene_info in results:
                if discovered >= self.max_discover:
                    break
                node_id = gene_info.get("node_id", "")
                name = gene_info.get("name", "")
                if not name or not node_id:
                    continue
                key = f"{node_id}/{name}"
                if key in seen:
                    continue
                seen.add(key)

                # Skip own genes (exact node_id match).
                if node_id == self.registry.node_id:
                    continue

                # Skip genes whose content already exists locally.
                gene_summary = gene_info.get("gene_summary", "")
                if gene_summary:
                    content_hash = hashlib.sha256(gene_summary.encode()).hexdigest()
                    if content_hash in local_hashes:
                        continue

                # Skip genes with low semantic relevance.
                if query_emb and gene_info.get("embedding"):
                    try:
                        raw = gene_info["embedding"]
                        gene_emb = json.loads(raw) if isinstance(raw, str) else raw
                        sim = _cosine_similarity(query_emb, gene_emb)
                        if sim < 0.3:
                            continue
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass  # allow through if embedding is unparseable

                try:
                    if self.gene_pool.install_from_hub(gene_info):
                        discovered += 1
                        log.info("Sync: installed gene %r from %s", name, node_id)
                except Exception:
                    log.debug("Failed to install gene %r", name, exc_info=True)

        return discovered

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

    def _validate_dependencies(self, name: str, dependencies: list[str]) -> bool:
        """Run dependency validation against the allowlist."""
        from ring1.skill_validator import validate_dependencies

        result = validate_dependencies(dependencies, self._allowed_packages)
        if not result.safe:
            log.warning(
                "Sync: rejected skill %r — bad deps: %s",
                name, "; ".join(result.errors),
            )
            return False
        return True
