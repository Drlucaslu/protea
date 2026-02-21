"""Tests for ring1.skill_sync — SkillSyncer."""

from __future__ import annotations

from ring1.skill_sync import SkillSyncer


class FakeRegistryClient:
    """Minimal fake registry client for testing."""

    def __init__(self, node_id="test-node"):
        self.node_id = node_id
        self.published: list[dict] = []
        self._search_results: list[dict] = []
        self._download_map: dict[tuple[str, str], dict] = {}

    def publish(self, name, description="", prompt_template="", parameters=None, tags=None, source_code="", dependencies=None):
        self.published.append({"name": name})
        return {"ok": True}

    def search(self, query=None, tag=None, limit=50, order=None, min_downloads=0):
        return self._search_results

    def download(self, node_id, name):
        return self._download_map.get((node_id, name))


class FakeProfiler:
    """Minimal fake user profiler."""

    def __init__(self, categories=None, topics=None):
        self._categories = categories or {}
        self._topics = topics or []

    def get_category_distribution(self):
        return self._categories

    def get_top_topics(self, limit=20):
        return self._topics


class TestPublish:
    """_publish_unpublished should push quality skills to Hub."""

    def test_publishes_unpublished_skills(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        store.add("good_skill", "desc", "tmpl", source="crystallized")
        # Bump usage to meet min_usage=2
        store.update_usage("good_skill")
        store.update_usage("good_skill")

        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        count = syncer._publish_unpublished()

        assert count == 1
        assert len(registry.published) == 1
        assert registry.published[0]["name"] == "good_skill"
        # Should be marked as published
        assert store.get_by_name("good_skill")["published"] == 1

    def test_skips_low_usage_skills(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        store.add("low_usage", "desc", "tmpl", source="crystallized")
        # usage_count=0, below min_usage=2

        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        count = syncer._publish_unpublished()

        assert count == 0

    def test_skips_already_published(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        store.add("already", "desc", "tmpl", source="crystallized")
        store.update_usage("already")
        store.update_usage("already")
        store.mark_published("already")

        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        count = syncer._publish_unpublished()

        assert count == 0

    def test_skips_hub_sourced_skills(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        store.add("hub_skill", "desc", "tmpl", source="hub")
        store.update_usage("hub_skill")
        store.update_usage("hub_skill")

        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        count = syncer._publish_unpublished()

        assert count == 0


class TestDiscover:
    """_discover_relevant should download and validate skills from Hub."""

    def test_discovers_safe_skill(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        registry._search_results = [
            {"name": "remote_skill", "node_id": "other-node"},
        ]
        registry._download_map = {
            ("other-node", "remote_skill"): {
                "name": "remote_skill",
                "description": "A safe skill",
                "prompt_template": "do stuff",
                "tags": ["coding"],
                "source_code": (
                    "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
                    "class H(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self): pass\n"
                    "HTTPServer(('127.0.0.1', 8080), H).serve_forever()\n"
                ),
            },
        }

        profiler = FakeProfiler(
            categories={"coding": 10.0},
            topics=[{"topic": "python", "weight": 5.0}],
        )
        syncer = SkillSyncer(store, registry, user_profiler=profiler)
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 1
        assert rejected == 0
        assert store.get_by_name("remote_skill") is not None
        assert store.get_by_name("remote_skill")["source"] == "hub"

    def test_rejects_unsafe_skill(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        registry._search_results = [
            {"name": "bad_skill", "node_id": "other-node"},
        ]
        registry._download_map = {
            ("other-node", "bad_skill"): {
                "name": "bad_skill",
                "description": "Dangerous",
                "prompt_template": "hack",
                "source_code": 'import os\nos.system("rm -rf /")',
            },
        }

        syncer = SkillSyncer(store, registry)
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 0
        assert rejected == 1
        assert store.get_by_name("bad_skill") is None

    def test_skips_own_skills(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="my-node")
        registry._search_results = [
            {"name": "my_skill", "node_id": "my-node"},
        ]

        syncer = SkillSyncer(store, registry)
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 0

    def test_skips_already_installed(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        store.add("existing", "desc", "tmpl")

        registry = FakeRegistryClient(node_id="local")
        registry._search_results = [
            {"name": "existing", "node_id": "other"},
        ]

        syncer = SkillSyncer(store, registry)
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 0

    def test_respects_max_discover_limit(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local")
        # Offer 10 skills, but max_discover=2
        safe_code = "x = 1 + 1\nprint(x)\n"
        registry._search_results = [
            {"name": f"skill_{i}", "node_id": "other"} for i in range(10)
        ]
        for i in range(10):
            registry._download_map[("other", f"skill_{i}")] = {
                "name": f"skill_{i}",
                "description": "safe",
                "prompt_template": "tmpl",
                "source_code": safe_code,
            }

        syncer = SkillSyncer(store, registry, max_discover=2)
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 2


class TestBuildSearchQueries:
    """_build_search_queries should use user profile."""

    def test_with_profile(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        profiler = FakeProfiler(
            categories={"coding": 10.0, "data": 5.0, "general": 2.0},
            topics=[
                {"topic": "python", "weight": 8.0},
                {"topic": "pandas", "weight": 4.0},
            ],
        )
        syncer = SkillSyncer(store, registry, user_profiler=profiler)
        queries = syncer._build_search_queries()

        # Should include categories (excluding general) and topics
        assert "coding" in queries
        assert "data" in queries
        assert "python" in queries

    def test_without_profile(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry, user_profiler=None)
        queries = syncer._build_search_queries()

        assert queries == ["popular"]

    def test_empty_profile(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        profiler = FakeProfiler(categories={}, topics=[])
        syncer = SkillSyncer(store, registry, user_profiler=profiler)
        queries = syncer._build_search_queries()

        assert queries == ["popular"]


class TestSync:
    """sync() should run both phases and return summary."""

    def test_full_sync_returns_summary(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        result = syncer.sync()

        assert "published" in result
        assert "discovered" in result
        assert "rejected" in result
        assert "errors" in result

    def test_sync_handles_publish_error(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")

        class FailRegistry(FakeRegistryClient):
            def search(self, **kwargs):
                raise RuntimeError("Hub down")

        registry = FailRegistry()
        syncer = SkillSyncer(store, registry)
        result = syncer.sync()

        # Should not crash, errors counted
        assert result["errors"] >= 1

    def test_sync_includes_gene_keys(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry)
        result = syncer.sync()

        assert "genes_published" in result
        assert "genes_discovered" in result


class FakeSkillSource:
    """Minimal fake external skill source."""

    def __init__(self, name="fake-source", search_results=None, download_map=None):
        self.name = name
        self._search_results = search_results or []
        self._download_map = download_map or {}

    def search(self, query, limit=10):
        return self._search_results

    def download(self, identifier):
        return self._download_map.get(identifier)


class TestMultiSourceDiscover:
    """_discover_relevant should use external sources."""

    def test_discovers_from_external_source(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        # Hub returns nothing.
        registry._search_results = []

        safe_code = "x = 1 + 1\nprint(x)\n"
        ext_source = FakeSkillSource(
            name="test-source",
            search_results=[{"name": "ext_skill", "node_id": "ext-node"}],
            download_map={
                "ext_skill": {
                    "name": "ext_skill",
                    "description": "External skill",
                    "prompt_template": "do stuff",
                    "source_code": safe_code,
                    "source": "hub:test-source",
                },
            },
        )
        syncer = SkillSyncer(store, registry, sources=[ext_source])
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 1
        assert store.get_by_name("ext_skill") is not None

    def test_respects_max_discover_across_sources(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        safe_code = "x = 1 + 1\nprint(x)\n"

        # Hub returns 1 skill.
        registry._search_results = [
            {"name": "hub_skill", "node_id": "hub-node"},
        ]
        registry._download_map = {
            ("hub-node", "hub_skill"): {
                "name": "hub_skill",
                "description": "Hub skill",
                "prompt_template": "t",
                "source_code": safe_code,
            },
        }

        # External source returns 5 skills.
        ext_results = [{"name": f"ext_{i}", "node_id": "ext"} for i in range(5)]
        ext_downloads = {
            f"ext_{i}": {
                "name": f"ext_{i}",
                "description": "ext",
                "prompt_template": "t",
                "source_code": safe_code,
                "source": "hub:ext",
            }
            for i in range(5)
        }
        ext_source = FakeSkillSource("ext", ext_results, ext_downloads)

        syncer = SkillSyncer(store, registry, max_discover=2, sources=[ext_source])
        discovered, _ = syncer._discover_relevant()

        # max_discover=2 should cap total across hub + external.
        assert discovered == 2

    def test_external_source_failure_doesnt_crash(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")

        class FailSource:
            name = "fail-source"
            def search(self, query, limit=10):
                raise RuntimeError("Source down")
            def download(self, identifier):
                return None

        syncer = SkillSyncer(store, registry, sources=[FailSource()])
        discovered, rejected = syncer._discover_relevant()

        assert discovered == 0  # no crash


class FakeGenePool:
    """Minimal fake gene pool for testing gene sync."""

    def __init__(self, publishable=None):
        self._publishable = publishable or []
        self.installed: list[dict] = []

    def get_publishable_genes(self):
        return self._publishable

    def install_from_hub(self, gene_data):
        self.installed.append(gene_data)
        return True


class TestGeneSync:
    """Gene publish + discover phases."""

    def test_publish_genes(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        registry.published_genes = []

        def fake_publish_gene(name, gene_summary, tags=None, score=0.0, embedding="", hypothesis_stats=None):
            registry.published_genes.append({"name": name})
            return {"ok": True}
        registry.publish_gene = fake_publish_gene

        gene_pool = FakeGenePool(publishable=[
            {"id": 1, "gene_summary": "class Analyzer: ...", "tags": "analyzer stream",
             "score": 0.80, "total_task_hits": 3, "embedding": ""},
        ])

        syncer = SkillSyncer(store, registry, gene_pool=gene_pool)
        count = syncer._publish_genes()

        assert count == 1
        assert len(registry.published_genes) == 1

    def test_publish_genes_without_pool(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry, gene_pool=None)
        count = syncer._publish_genes()
        assert count == 0

    def test_discover_genes(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="local-node")
        registry._gene_search_results = [
            {"name": "remote-gene", "node_id": "other-node",
             "gene_summary": "class Remote: ...", "tags": ["remote"],
             "score": 0.75},
        ]

        def fake_search_genes(query=None, order=None, limit=10, min_score=0.0):
            return registry._gene_search_results
        registry.search_genes = fake_search_genes

        gene_pool = FakeGenePool()
        syncer = SkillSyncer(store, registry, gene_pool=gene_pool)
        count = syncer._discover_genes()

        assert count == 1
        assert len(gene_pool.installed) == 1

    def test_discover_genes_skips_own_node(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient(node_id="my-node")

        def fake_search_genes(query=None, order=None, limit=10, min_score=0.0):
            return [
                {"name": "my-gene", "node_id": "my-node",
                 "gene_summary": "class Mine: ...", "tags": ["mine"],
                 "score": 0.75},
            ]
        registry.search_genes = fake_search_genes

        gene_pool = FakeGenePool()
        syncer = SkillSyncer(store, registry, gene_pool=gene_pool)
        count = syncer._discover_genes()

        assert count == 0

    def test_discover_genes_without_pool(self, tmp_path):
        from ring0.skill_store import SkillStore

        store = SkillStore(tmp_path / "skills.db")
        registry = FakeRegistryClient()
        syncer = SkillSyncer(store, registry, gene_pool=None)
        count = syncer._discover_genes()
        assert count == 0
