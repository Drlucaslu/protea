"""Tests for ring0.skill_store — SkillStore."""

from __future__ import annotations

import sqlite3

import pytest

from ring0.skill_store import SkillStore


class TestAdd:
    """add() should insert rows and return their rowid."""

    def test_insert_returns_rowid(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        rid = store.add("greet", "Greeting skill", "Hello {{name}}")
        assert rid == 1

    def test_successive_inserts_increment_rowid(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        r1 = store.add("skill_a", "Desc A", "Template A")
        r2 = store.add("skill_b", "Desc B", "Template B")
        assert r2 == r1 + 1

    def test_parameters_stored_as_json(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", parameters={"key": "value", "n": 42})
        skill = store.get_by_name("s1")
        assert skill["parameters"] == {"key": "value", "n": 42}

    def test_parameters_default_empty_dict(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["parameters"] == {}

    def test_tags_stored_as_json(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", tags=["tag1", "tag2"])
        skill = store.get_by_name("s1")
        assert skill["tags"] == ["tag1", "tag2"]

    def test_tags_default_empty_list(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["tags"] == []

    def test_source_default_user(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["source"] == "user"

    def test_source_custom(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="evolution")
        skill = store.get_by_name("s1")
        assert skill["source"] == "evolution"

    def test_unique_name_constraint(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("unique_skill", "desc", "tmpl")
        with pytest.raises(sqlite3.IntegrityError):
            store.add("unique_skill", "other desc", "other tmpl")


class TestGetByName:
    """get_by_name() should return a skill dict or None."""

    def test_found(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("greet", "Greeting", "Hello {{name}}")
        skill = store.get_by_name("greet")
        assert skill is not None
        assert skill["name"] == "greet"
        assert skill["description"] == "Greeting"
        assert skill["prompt_template"] == "Hello {{name}}"

    def test_not_found(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.get_by_name("nonexistent") is None

    def test_returns_all_fields(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", parameters={"k": "v"}, tags=["t1"])
        skill = store.get_by_name("s1")
        assert "id" in skill
        assert "created_at" in skill
        assert skill["usage_count"] == 0
        assert skill["active"] is True


class TestGetActive:
    """get_active() should return active skills ordered by usage_count."""

    def test_returns_active_only(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("active1", "desc", "tmpl")
        store.add("active2", "desc", "tmpl")
        store.add("inactive", "desc", "tmpl")
        store.deactivate("inactive")

        skills = store.get_active()
        names = [s["name"] for s in skills]
        assert "active1" in names
        assert "active2" in names
        assert "inactive" not in names

    def test_ordered_by_usage_count(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("low", "desc", "tmpl")
        store.add("high", "desc", "tmpl")
        store.update_usage("high")
        store.update_usage("high")
        store.update_usage("low")

        skills = store.get_active()
        assert skills[0]["name"] == "high"
        assert skills[1]["name"] == "low"

    def test_respects_limit(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        for i in range(10):
            store.add(f"skill_{i}", "desc", "tmpl")
        skills = store.get_active(limit=3)
        assert len(skills) == 3

    def test_empty_returns_empty_list(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.get_active() == []


class TestUpdateUsage:
    """update_usage() should increment the usage count."""

    def test_increments(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        assert store.get_by_name("s1")["usage_count"] == 0
        store.update_usage("s1")
        assert store.get_by_name("s1")["usage_count"] == 1
        store.update_usage("s1")
        assert store.get_by_name("s1")["usage_count"] == 2

    def test_nonexistent_no_error(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.update_usage("nonexistent")  # should not raise


class TestDeactivate:
    """deactivate() should set active to False."""

    def test_deactivates(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        assert store.get_by_name("s1")["active"] is True
        store.deactivate("s1")
        assert store.get_by_name("s1")["active"] is False

    def test_nonexistent_no_error(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.deactivate("nonexistent")  # should not raise


class TestCount:
    """count() should return total number of skills."""

    def test_empty(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.count() == 0

    def test_after_inserts(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        assert store.count() == 2

    def test_includes_inactive(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.deactivate("a")
        assert store.count() == 1


class TestClear:
    """clear() should delete all skills."""

    def test_clears_all(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        assert store.count() == 2
        store.clear()
        assert store.count() == 0
        assert store.get_active() == []

    def test_clear_empty(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.clear()  # should not raise
        assert store.count() == 0

    def test_add_after_clear(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("before", "desc", "tmpl")
        store.clear()
        store.add("after", "desc", "tmpl")
        assert store.count() == 1
        assert store.get_by_name("after") is not None


class TestCountActive:
    """count_active() should return only active skills."""

    def test_empty(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.count_active() == 0

    def test_all_active(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        assert store.count_active() == 2

    def test_excludes_deactivated(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        store.deactivate("b")
        assert store.count_active() == 1

    def test_all_deactivated(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.deactivate("a")
        assert store.count_active() == 0


class TestGetLeastUsed:
    """get_least_used() should return least-used active skills."""

    def test_empty(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.get_least_used() == []

    def test_ordered_by_usage(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("high", "desc", "tmpl")
        store.add("low", "desc", "tmpl")
        store.update_usage("high")
        store.update_usage("high")

        result = store.get_least_used(limit=2)
        assert result[0]["name"] == "low"
        assert result[1]["name"] == "high"

    def test_same_usage_ordered_by_id(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("first", "desc", "tmpl")
        store.add("second", "desc", "tmpl")

        result = store.get_least_used(limit=2)
        assert result[0]["name"] == "first"
        assert result[1]["name"] == "second"

    def test_excludes_deactivated(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("active", "desc", "tmpl")
        store.add("inactive", "desc", "tmpl")
        store.deactivate("inactive")

        result = store.get_least_used(limit=5)
        names = [s["name"] for s in result]
        assert "inactive" not in names
        assert "active" in names

    def test_limit(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        for i in range(5):
            store.add(f"s{i}", "desc", "tmpl")
        assert len(store.get_least_used(limit=2)) == 2


class TestUpdate:
    """update() should modify specified fields of an existing skill."""

    def test_update_description(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "old desc", "tmpl")
        assert store.update("s1", description="new desc") is True
        assert store.get_by_name("s1")["description"] == "new desc"

    def test_update_multiple_fields(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", tags=["old"])
        store.update("s1", description="new", prompt_template="new tmpl", tags=["new"])
        skill = store.get_by_name("s1")
        assert skill["description"] == "new"
        assert skill["prompt_template"] == "new tmpl"
        assert skill["tags"] == ["new"]

    def test_update_source_code(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        store.update("s1", source_code="print('hello')")
        assert store.get_by_name("s1")["source_code"] == "print('hello')"

    def test_nonexistent_returns_false(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.update("nonexistent", description="x") is False

    def test_no_fields_returns_false(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        assert store.update("s1") is False

    def test_preserves_other_fields(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", tags=["keep"], source_code="keep code")
        store.update("s1", description="new desc")
        skill = store.get_by_name("s1")
        assert skill["tags"] == ["keep"]
        assert skill["prompt_template"] == "tmpl"
        assert skill["source_code"] == "keep code"


class TestSourceCode:
    """source_code field should be stored and retrieved correctly."""

    def test_add_with_source_code(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source_code="print('hi')")
        skill = store.get_by_name("s1")
        assert skill["source_code"] == "print('hi')"

    def test_default_empty(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["source_code"] == ""

    def test_get_by_name_returns_source_code(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        code = "import os\ndef main(): pass"
        store.add("s1", "desc", "tmpl", source_code=code)
        assert store.get_by_name("s1")["source_code"] == code


class TestSchemaMigration:
    """Opening an old database should auto-migrate the schema."""

    def test_migrate_adds_source_code_column(self, tmp_path):
        db_path = tmp_path / "old.db"
        # Create a database WITHOUT source_code column.
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE skills ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL UNIQUE,"
            "  description TEXT NOT NULL,"
            "  prompt_template TEXT NOT NULL,"
            "  parameters TEXT DEFAULT '{}',"
            "  tags TEXT DEFAULT '[]',"
            "  source TEXT NOT NULL DEFAULT 'user',"
            "  usage_count INTEGER DEFAULT 0,"
            "  active BOOLEAN DEFAULT 1,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        con.execute("INSERT INTO skills (name, description, prompt_template) VALUES ('old', 'old desc', 'old tmpl')")
        con.commit()
        con.close()

        # Open with SkillStore — should auto-migrate.
        store = SkillStore(db_path)
        skill = store.get_by_name("old")
        assert skill is not None
        assert skill["source_code"] == ""

        # New inserts should also work.
        store.add("new", "new desc", "new tmpl", source_code="code")
        assert store.get_by_name("new")["source_code"] == "code"


class TestUpdateUsageLastUsedAt:
    """update_usage() should also set last_used_at."""

    def test_sets_last_used_at(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        assert store.get_by_name("s1").get("last_used_at") is None
        store.update_usage("s1")
        assert store.get_by_name("s1")["last_used_at"] is not None


class TestEvictStale:
    """evict_stale() should remove old hub skills but keep local ones."""

    def test_evicts_old_hub_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("old_hub", "desc", "tmpl", source="hub")
        # Manually backdate created_at
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-60 days') WHERE name = 'old_hub'"
            )
        evicted = store.evict_stale(days=30)
        assert evicted == 1
        assert store.get_by_name("old_hub") is None

    def test_keeps_recent_hub_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("new_hub", "desc", "tmpl", source="hub")
        evicted = store.evict_stale(days=30)
        assert evicted == 0
        assert store.get_by_name("new_hub") is not None

    def test_never_evicts_local_skills(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("local", "desc", "tmpl", source="user")
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-60 days') WHERE name = 'local'"
            )
        evicted = store.evict_stale(days=30)
        assert evicted == 0
        assert store.get_by_name("local") is not None

    def test_keeps_recently_used_hub_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("hub_used", "desc", "tmpl", source="hub")
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-60 days') WHERE name = 'hub_used'"
            )
        store.update_usage("hub_used")  # sets last_used_at to now
        evicted = store.evict_stale(days=30)
        assert evicted == 0
        assert store.get_by_name("hub_used") is not None

    def test_empty_store_returns_zero(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        assert store.evict_stale() == 0


class TestSchemaMigrationLastUsedAt:
    """Opening an old database should add last_used_at column."""

    def test_migrate_adds_last_used_at(self, tmp_path):
        db_path = tmp_path / "old.db"
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE skills ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL UNIQUE,"
            "  description TEXT NOT NULL,"
            "  prompt_template TEXT NOT NULL,"
            "  parameters TEXT DEFAULT '{}',"
            "  tags TEXT DEFAULT '[]',"
            "  source TEXT NOT NULL DEFAULT 'user',"
            "  source_code TEXT DEFAULT '',"
            "  usage_count INTEGER DEFAULT 0,"
            "  active BOOLEAN DEFAULT 1,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        con.execute("INSERT INTO skills (name, description, prompt_template) VALUES ('old', 'desc', 'tmpl')")
        con.commit()
        con.close()

        store = SkillStore(db_path)
        skill = store.get_by_name("old")
        assert skill is not None
        assert skill.get("last_used_at") is None
        store.update_usage("old")
        assert store.get_by_name("old")["last_used_at"] is not None


class TestSharedDatabase:
    """SkillStore should coexist with other tables in same db."""

    def test_coexists_with_memory_and_fitness(self, tmp_path):
        from ring0.fitness import FitnessTracker
        from ring0.memory import MemoryStore

        db_path = tmp_path / "protea.db"
        fitness = FitnessTracker(db_path)
        memory = MemoryStore(db_path)
        skills = SkillStore(db_path)

        fitness.record(1, "abc", 0.9, 60.0, True)
        memory.add(1, "observation", "survived")
        skills.add("greet", "Greeting", "Hello")

        assert len(fitness.get_history()) == 1
        assert memory.count() == 1
        assert skills.count() == 1


class TestPublished:
    """published column and related methods."""

    def test_default_not_published(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="crystallized")
        skill = store.get_by_name("s1")
        assert skill["published"] == 0

    def test_get_local_names(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        store.add("c", "desc", "tmpl")
        store.deactivate("c")

        names = store.get_local_names()
        assert "a" in names
        assert "b" in names
        assert "c" not in names  # deactivated


class TestEvictStaleEnhanced:
    """Enhanced eviction: never-used hub skills expire after 7 days."""

    def test_never_used_hub_skill_evicted_after_7_days(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("unused_hub", "desc", "tmpl", source="hub")
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-10 days') "
                "WHERE name = 'unused_hub'"
            )
        evicted = store.evict_stale(days=30)
        assert evicted == 1

    def test_never_used_hub_skill_kept_within_7_days(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("fresh_hub", "desc", "tmpl", source="hub")
        evicted = store.evict_stale(days=30)
        assert evicted == 0

    def test_used_hub_skill_uses_normal_expiry(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("used_hub", "desc", "tmpl", source="hub")
        store.update_usage("used_hub")
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-10 days'), "
                "last_used_at = datetime('now', '-10 days') WHERE name = 'used_hub'"
            )
        # 10 days old, used — should NOT be evicted with 30-day threshold
        evicted = store.evict_stale(days=30)
        assert evicted == 0


class TestSchemaMigrationPublished:
    """Opening an old database should add published column."""

    def test_migrate_adds_published(self, tmp_path):
        db_path = tmp_path / "old.db"
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE skills ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL UNIQUE,"
            "  description TEXT NOT NULL,"
            "  prompt_template TEXT NOT NULL,"
            "  parameters TEXT DEFAULT '{}',"
            "  tags TEXT DEFAULT '[]',"
            "  source TEXT NOT NULL DEFAULT 'user',"
            "  source_code TEXT DEFAULT '',"
            "  usage_count INTEGER DEFAULT 0,"
            "  active BOOLEAN DEFAULT 1,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
            "  last_used_at TEXT DEFAULT NULL"
            ")"
        )
        con.execute("INSERT INTO skills (name, description, prompt_template) VALUES ('old', 'desc', 'tmpl')")
        con.commit()
        con.close()

        store = SkillStore(db_path)
        skill = store.get_by_name("old")
        assert skill is not None
        assert skill["published"] == 0


class TestDependencies:
    """dependencies column should store and retrieve JSON lists."""

    def test_add_with_dependencies(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", dependencies=["requests", "pandas"])
        skill = store.get_by_name("s1")
        assert skill["dependencies"] == ["requests", "pandas"]

    def test_default_empty_list(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["dependencies"] == []



class TestPermanent:
    """permanent column and related methods."""

    def test_default_not_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["permanent"] == 0

    def test_mark_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        store.mark_permanent("s1")
        skill = store.get_by_name("s1")
        assert skill["permanent"] == 1

    def test_get_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("perm", "desc", "tmpl")
        store.add("normal", "desc", "tmpl")
        store.mark_permanent("perm")
        permanent = store.get_permanent()
        names = [s["name"] for s in permanent]
        assert "perm" in names
        assert "normal" not in names

    def test_get_permanent_excludes_inactive(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("perm", "desc", "tmpl")
        store.mark_permanent("perm")
        store.deactivate("perm")
        assert store.get_permanent() == []

    def test_evict_stale_skips_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("perm_hub", "desc", "tmpl", source="hub")
        store.mark_permanent("perm_hub")
        # Backdate to trigger eviction
        with store._connect() as con:
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-60 days') "
                "WHERE name = 'perm_hub'"
            )
        evicted = store.evict_stale(days=30)
        assert evicted == 0
        assert store.get_by_name("perm_hub") is not None


class TestAutoPromoteEvolved:
    """update_usage() should auto-promote evolved skills to permanent."""

    def test_evolved_skill_promoted_on_first_use(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("cap", "desc", "tmpl", source="evolved")
        assert store.get_by_name("cap")["permanent"] == 0
        store.update_usage("cap")
        skill = store.get_by_name("cap")
        assert skill["permanent"] == 1
        assert skill["usage_count"] == 1

    def test_non_evolved_skill_not_promoted(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("user_skill", "desc", "tmpl", source="user")
        store.update_usage("user_skill")
        assert store.get_by_name("user_skill")["permanent"] == 0

    def test_already_permanent_stays_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("cap", "desc", "tmpl", source="evolved")
        store.mark_permanent("cap")
        store.update_usage("cap")
        store.update_usage("cap")
        assert store.get_by_name("cap")["permanent"] == 1
        assert store.get_by_name("cap")["usage_count"] == 2

    def test_promoted_skill_appears_in_get_permanent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("cap", "desc", "tmpl", source="evolved")
        assert store.get_permanent() == []
        store.update_usage("cap")
        permanent = store.get_permanent()
        assert len(permanent) == 1
        assert permanent[0]["name"] == "cap"

    def test_promoted_skill_not_evicted(self, tmp_path):
        """Evolved skill that was promoted should survive eviction."""
        store = SkillStore(tmp_path / "skills.db")
        store.add("cap", "desc", "tmpl", source="evolved")
        store.update_usage("cap")  # promotes to permanent
        # Even if we change source to hub and backdate, permanent should protect it.
        with store._connect() as con:
            con.execute("UPDATE skills SET source = 'hub' WHERE name = 'cap'")
            con.execute(
                "UPDATE skills SET created_at = datetime('now', '-60 days'), "
                "last_used_at = datetime('now', '-60 days') WHERE name = 'cap'"
            )
        evicted = store.evict_stale(days=30)
        assert evicted == 0
        assert store.get_by_name("cap") is not None


class TestSchemaMigrationDependenciesPermanent:
    """Opening an old database should add dependencies and permanent columns."""

    def test_migrate_adds_new_columns(self, tmp_path):
        db_path = tmp_path / "old.db"
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE skills ("
            "  id INTEGER PRIMARY KEY,"
            "  name TEXT NOT NULL UNIQUE,"
            "  description TEXT NOT NULL,"
            "  prompt_template TEXT NOT NULL,"
            "  parameters TEXT DEFAULT '{}',"
            "  tags TEXT DEFAULT '[]',"
            "  source TEXT NOT NULL DEFAULT 'user',"
            "  source_code TEXT DEFAULT '',"
            "  usage_count INTEGER DEFAULT 0,"
            "  active BOOLEAN DEFAULT 1,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
            "  last_used_at TEXT DEFAULT NULL,"
            "  published BOOLEAN DEFAULT 0"
            ")"
        )
        con.execute("INSERT INTO skills (name, description, prompt_template) VALUES ('old', 'desc', 'tmpl')")
        con.commit()
        con.close()

        store = SkillStore(db_path)
        skill = store.get_by_name("old")
        assert skill is not None
        assert skill["dependencies"] == []
        assert skill["permanent"] == 0

        # New features should work.
        store.add("new", "desc", "tmpl", dependencies=["requests"])
        assert store.get_by_name("new")["dependencies"] == ["requests"]
        store.mark_permanent("new")
        assert store.get_by_name("new")["permanent"] == 1


class TestMatchCount:
    """match_count column and record_matches() method."""

    def test_default_zero(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        skill = store.get_by_name("s1")
        assert skill["match_count"] == 0

    def test_record_matches_increments(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        store.record_matches(["s1"])
        assert store.get_by_name("s1")["match_count"] == 1
        store.record_matches(["s1"])
        assert store.get_by_name("s1")["match_count"] == 2

    def test_record_matches_multiple_names(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("a", "desc", "tmpl")
        store.add("b", "desc", "tmpl")
        store.record_matches(["a", "b"])
        assert store.get_by_name("a")["match_count"] == 1
        assert store.get_by_name("b")["match_count"] == 1

    def test_record_matches_empty_list_safe(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.record_matches([])  # should not raise

    def test_match_count_independent_of_usage_count(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl")
        store.record_matches(["s1"])
        store.update_usage("s1")
        skill = store.get_by_name("s1")
        assert skill["match_count"] == 1
        assert skill["usage_count"] == 1

    def test_nonexistent_skill_no_error(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.record_matches(["nonexistent"])  # should not raise


class TestBackfillLineage:
    """backfill_lineage() heuristic skill→gene traceability."""

    def _mock_gene_pool(self, genes=None):
        """Create a mock gene_pool with get_relevant."""
        from unittest.mock import MagicMock
        gp = MagicMock()
        if genes is None:
            genes = [{"id": 1}, {"id": 2}]
        gp.get_relevant.return_value = genes
        return gp

    def test_backfills_crystallized_without_lineage(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "Summarize text", "Summarize: {{text}}", source="crystallized")
        gp = self._mock_gene_pool()

        count = store.backfill_lineage(gp)
        assert count == 1
        lineage = store.get_lineage("s1")
        assert len(lineage) == 2
        assert all(e["generation"] == 0 for e in lineage)

    def test_idempotent(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "Summarize text", "tmpl", source="crystallized")
        gp = self._mock_gene_pool()

        store.backfill_lineage(gp)
        count = store.backfill_lineage(gp)
        assert count == 0  # already has lineage

    def test_skips_non_crystallized(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="user")
        gp = self._mock_gene_pool()

        count = store.backfill_lineage(gp)
        assert count == 0

    def test_skips_skills_with_existing_lineage(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="crystallized")
        store.record_lineage("s1", [99], generation=5)
        gp = self._mock_gene_pool()

        count = store.backfill_lineage(gp)
        assert count == 0

    def test_empty_gene_pool_results(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="crystallized")
        gp = self._mock_gene_pool(genes=[])

        count = store.backfill_lineage(gp)
        assert count == 0
