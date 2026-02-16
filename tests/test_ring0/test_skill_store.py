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


class TestInstallFromHub:
    """install_from_hub() should insert or update a hub skill."""

    def test_install_new_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        data = {
            "name": "hub_skill",
            "description": "From hub",
            "prompt_template": "Do {{task}}",
            "parameters": {"task": "string"},
            "tags": ["hub"],
            "source_code": "print('hub')",
        }
        rid = store.install_from_hub(data)
        assert rid > 0
        skill = store.get_by_name("hub_skill")
        assert skill["source"] == "hub"
        assert skill["description"] == "From hub"
        assert skill["source_code"] == "print('hub')"

    def test_update_existing_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "old desc", "old tmpl", source="user")
        store.install_from_hub({
            "name": "s1",
            "description": "new desc",
            "prompt_template": "new tmpl",
            "source_code": "new code",
        })
        skill = store.get_by_name("s1")
        assert skill["source"] == "hub"
        assert skill["description"] == "new desc"
        assert skill["source_code"] == "new code"

    def test_reactivates_deactivated_skill(self, tmp_path):
        store = SkillStore(tmp_path / "skills.db")
        store.add("s1", "desc", "tmpl", source="hub")
        store.deactivate("s1")
        assert store.get_by_name("s1")["active"] is False
        store.install_from_hub({"name": "s1", "description": "desc", "prompt_template": "tmpl"})
        assert store.get_by_name("s1")["active"] is True


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
