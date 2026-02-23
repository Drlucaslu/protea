"""Tests for ring1.tools.skill — run_skill, view_skill, edit_skill tools."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from ring1.tools.skill import make_edit_skill_tool, make_run_skill_tool, make_view_skill_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_dict(name="my_skill", source_code="print('hello')"):
    return {
        "id": 1,
        "name": name,
        "description": "A test skill",
        "source_code": source_code,
    }


def _make_mocks(skill_dict=None, is_running=False, info=None):
    """Return (skill_store, skill_runner) mock pair."""
    store = MagicMock()
    store.get_by_name.return_value = skill_dict

    runner = MagicMock()
    runner.is_running.return_value = is_running
    runner.run.return_value = (12345, "Skill *my_skill* started (PID 12345).")
    runner.get_output.return_value = "server started"
    runner.get_info.return_value = info
    return store, runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunSkillTool:

    def test_skill_not_found(self):
        store, runner = _make_mocks(skill_dict=None)
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "nonexistent"})
        assert "not found" in result
        runner.run.assert_not_called()

    def test_skill_no_source_code(self):
        store, runner = _make_mocks(skill_dict=_make_skill_dict(source_code=""))
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})
        assert "no source code" in result
        runner.run.assert_not_called()

    @patch("ring1.tools.skill.time.sleep")
    def test_normal_start(self, mock_sleep):
        skill = _make_skill_dict()
        store, runner = _make_mocks(
            skill_dict=skill,
            info={"skill_name": "my_skill", "pid": 12345, "running": True, "uptime": 3.0, "port": 8080},
        )
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})

        runner.run.assert_called_once_with("my_skill", "print('hello')", dependencies=None)
        store.update_usage.assert_called_once_with("my_skill")
        mock_sleep.assert_called_once_with(3)
        assert "PID 12345" in result
        assert "8080" in result
        assert "web_fetch" in result

    @patch("ring1.tools.skill.time.sleep")
    def test_normal_start_no_port(self, mock_sleep):
        skill = _make_skill_dict()
        store, runner = _make_mocks(
            skill_dict=skill,
            info={"skill_name": "my_skill", "pid": 12345, "running": True, "uptime": 3.0, "port": None},
        )
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})

        assert "PID 12345" in result
        assert "web_fetch" not in result

    def test_same_skill_already_running(self):
        skill = _make_skill_dict()
        store, runner = _make_mocks(
            skill_dict=skill,
            is_running=True,
            info={"skill_name": "my_skill", "pid": 99, "running": True, "uptime": 10.0, "port": 5000},
        )
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})

        # Should NOT restart — just return current status.
        runner.run.assert_not_called()
        assert "already running" in result
        assert "PID 99" in result
        assert "5000" in result

    @patch("ring1.tools.skill.time.sleep")
    def test_different_skill_running_stops_old(self, mock_sleep):
        skill = _make_skill_dict(name="new_skill")
        store, runner = _make_mocks(
            skill_dict=skill,
            is_running=True,
            info={"skill_name": "old_skill", "pid": 88, "running": True, "uptime": 60.0, "port": None},
        )
        # get_info called 3 times: check name, stop-log, post-start status.
        runner.get_info.side_effect = [
            {"skill_name": "old_skill", "pid": 88, "running": True, "uptime": 60.0, "port": None},
            {"skill_name": "old_skill", "pid": 88, "running": True, "uptime": 60.0, "port": None},
            {"skill_name": "new_skill", "pid": 12345, "running": True, "uptime": 3.0, "port": None},
        ]
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "new_skill"})

        runner.stop.assert_called_once()
        runner.run.assert_called_once()
        assert "PID 12345" in result

    @patch("ring1.tools.skill.time.sleep")
    def test_run_raises_exception(self, mock_sleep):
        skill = _make_skill_dict()
        store, runner = _make_mocks(skill_dict=skill)
        runner.run.side_effect = RuntimeError("boom")
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})

        assert "Error" in result
        assert "boom" in result

    @patch("ring1.tools.skill.time.sleep")
    def test_process_exits_early(self, mock_sleep):
        skill = _make_skill_dict()
        store, runner = _make_mocks(
            skill_dict=skill,
            info={"skill_name": "my_skill", "pid": 12345, "running": False, "uptime": 0.0, "port": None},
        )
        # is_running returns False after run (process exited).
        runner.is_running.side_effect = [False, False]
        tool = make_run_skill_tool(store, runner)
        result = tool.execute({"skill_name": "my_skill"})

        assert "WARNING" in result
        assert "exited" in result

    def test_schema(self):
        store, runner = _make_mocks()
        tool = make_run_skill_tool(store, runner)
        assert tool.name == "run_skill"
        assert "skill_name" in tool.input_schema["properties"]
        assert "skill_name" in tool.input_schema["required"]
        assert tool.input_schema["type"] == "object"


# ---------------------------------------------------------------------------
# TestViewSkillTool
# ---------------------------------------------------------------------------

class TestViewSkillTool:

    def test_skill_not_found(self):
        store = MagicMock()
        store.get_by_name.return_value = None
        tool = make_view_skill_tool(store)
        result = tool.execute({"skill_name": "nonexistent"})
        assert "not found" in result

    def test_view_returns_metadata_and_source(self):
        store = MagicMock()
        store.get_by_name.return_value = {
            "name": "my_skill",
            "description": "A cool skill",
            "tags": ["web", "api"],
            "source_code": "print('hello')",
        }
        tool = make_view_skill_tool(store)
        result = tool.execute({"skill_name": "my_skill"})
        assert "my_skill" in result
        assert "A cool skill" in result
        assert "web" in result
        assert "print('hello')" in result

    def test_schema(self):
        store = MagicMock()
        tool = make_view_skill_tool(store)
        assert tool.name == "view_skill"
        assert "skill_name" in tool.input_schema["required"]


# ---------------------------------------------------------------------------
# TestEditSkillTool
# ---------------------------------------------------------------------------

class TestEditSkillTool:

    def test_skill_not_found(self):
        store = MagicMock()
        store.get_by_name.return_value = None
        tool = make_edit_skill_tool(store)
        result = tool.execute({"skill_name": "x", "old_string": "a", "new_string": "b"})
        assert "not found" in result
        store.update.assert_not_called()

    def test_old_string_not_found(self):
        store = MagicMock()
        store.get_by_name.return_value = {"source_code": "print('hello')"}
        tool = make_edit_skill_tool(store)
        result = tool.execute({"skill_name": "x", "old_string": "goodbye", "new_string": "hi"})
        assert "not found" in result.lower()
        store.update.assert_not_called()

    def test_old_string_multiple_matches(self):
        store = MagicMock()
        store.get_by_name.return_value = {"source_code": "aaa"}
        tool = make_edit_skill_tool(store)
        result = tool.execute({"skill_name": "x", "old_string": "a", "new_string": "b"})
        assert "3 times" in result
        store.update.assert_not_called()

    def test_successful_edit(self):
        store = MagicMock()
        store.get_by_name.return_value = {"source_code": "print('hello')"}
        tool = make_edit_skill_tool(store)
        result = tool.execute({"skill_name": "my_skill", "old_string": "hello", "new_string": "world"})
        assert "updated successfully" in result
        store.update.assert_called_once_with("my_skill", source_code="print('world')")

    def test_schema(self):
        store = MagicMock()
        tool = make_edit_skill_tool(store)
        assert tool.name == "edit_skill"
        assert "skill_name" in tool.input_schema["required"]
        assert "old_string" in tool.input_schema["required"]
        assert "new_string" in tool.input_schema["required"]


