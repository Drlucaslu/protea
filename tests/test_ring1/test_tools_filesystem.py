"""Tests for ring1.tools.filesystem."""

from __future__ import annotations

import os
import pathlib

import pytest

from ring1.tools.filesystem import (
    _check_write_allowed,
    _resolve_safe,
    _route_output_path,
    make_filesystem_tools,
)


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory with sample files."""
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content\n")
    return tmp_path


@pytest.fixture
def tools(workspace):
    """Create filesystem tools bound to the workspace."""
    tool_list = make_filesystem_tools(str(workspace))
    return {t.name: t for t in tool_list}


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class TestResolveSafe:
    def test_normal_path(self, workspace):
        result = _resolve_safe(workspace, "hello.txt")
        assert result == workspace / "hello.txt"

    def test_subdir_path(self, workspace):
        result = _resolve_safe(workspace, "sub/nested.txt")
        assert result == workspace / "sub" / "nested.txt"

    def test_dotdot_escape_outside_home_blocked(self, workspace):
        """Paths that escape the user's home directory are blocked."""
        with pytest.raises(ValueError, match="outside home"):
            _resolve_safe(workspace, "/etc/passwd")

    def test_absolute_path_within_home_allowed(self, workspace):
        """Absolute paths within ~ are allowed."""
        home = pathlib.Path.home()
        result = _resolve_safe(workspace, str(home))
        assert result == home.resolve()

    def test_tilde_path_allowed(self, workspace):
        """~/… paths are resolved and allowed."""
        result = _resolve_safe(workspace, "~")
        assert result == pathlib.Path.home().resolve()

    def test_dot_path(self, workspace):
        result = _resolve_safe(workspace, ".")
        assert result == workspace.resolve()

    def test_sensitive_ssh_blocked(self, workspace):
        with pytest.raises(ValueError, match="protected"):
            _resolve_safe(workspace, "~/.ssh/id_rsa")

    def test_sensitive_gnupg_blocked(self, workspace):
        with pytest.raises(ValueError, match="protected"):
            _resolve_safe(workspace, "~/.gnupg/secring.gpg")

    def test_sensitive_aws_blocked(self, workspace):
        with pytest.raises(ValueError, match="protected"):
            _resolve_safe(workspace, "~/.aws/credentials")

    def test_sensitive_env_file_blocked(self, workspace):
        with pytest.raises(ValueError, match="protected"):
            _resolve_safe(workspace, "~/.env")

    def test_env_in_subdir_allowed(self, workspace):
        """~/.env is blocked but ~/project/.env is fine."""
        home = pathlib.Path.home()
        # Create a fake path — _resolve_safe only checks, doesn't require existence
        result = _resolve_safe(workspace, str(home / "project" / ".env"))
        assert result == (home / "project" / ".env").resolve()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_basic(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt"})
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_line_numbers(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt"})
        assert "\t" in result  # line number tab separator

    def test_read_with_offset_limit(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt", "offset": 1, "limit": 1})
        assert "line2" in result
        assert "line1" not in result
        assert "line3" not in result

    def test_read_nonexistent(self, tools):
        result = tools["read_file"].execute({"path": "nope.txt"})
        assert "Error" in result

    def test_read_outside_home_blocked(self, tools):
        result = tools["read_file"].execute({"path": "/etc/passwd"})
        assert "Error" in result
        assert "outside home" in result

    def test_read_nested(self, tools):
        result = tools["read_file"].execute({"path": "sub/nested.txt"})
        assert "nested content" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_new_file(self, tools, workspace):
        result = tools["write_file"].execute({"path": "new.txt", "content": "hello"})
        assert "Written" in result
        assert (workspace / "new.txt").read_text() == "hello"

    def test_write_creates_parent_dirs(self, tools, workspace):
        result = tools["write_file"].execute(
            {"path": "deep/nested/dir/file.txt", "content": "data"}
        )
        assert "Written" in result
        assert (workspace / "deep" / "nested" / "dir" / "file.txt").read_text() == "data"

    def test_write_overwrites(self, tools, workspace):
        tools["write_file"].execute({"path": "hello.txt", "content": "new content"})
        assert (workspace / "hello.txt").read_text() == "new content"

    def test_write_outside_home_blocked(self, tools):
        result = tools["write_file"].execute(
            {"path": "/etc/evil.txt", "content": "bad"}
        )
        assert "Error" in result
        assert "outside home" in result

    def test_write_ring0_blocked(self, tools):
        result = tools["write_file"].execute(
            {"path": "ring0/memory.py", "content": "hacked"}
        )
        assert "Error" in result
        assert "protected source" in result

    def test_write_ring1_blocked(self, tools):
        result = tools["write_file"].execute(
            {"path": "ring1/sentinel.py", "content": "hacked"}
        )
        assert "Error" in result
        assert "protected source" in result

    def test_write_tests_blocked(self, tools):
        result = tools["write_file"].execute(
            {"path": "tests/test_foo.py", "content": "hacked"}
        )
        assert "Error" in result
        assert "protected source" in result

    def test_write_ring2_main_blocked(self, tools, workspace):
        """write_file should reject writes to ring2/main.py."""
        (workspace / "ring2").mkdir(exist_ok=True)
        result = tools["write_file"].execute(
            {"path": "ring2/main.py", "content": "hacked"}
        )
        assert "Error" in result
        assert "auto-evolved" in result

    def test_write_output_allowed(self, tools, workspace):
        """output/ and other dirs are not protected."""
        result = tools["write_file"].execute(
            {"path": "output/report.txt", "content": "data"}
        )
        assert "Written" in result


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

class TestEditFile:
    def test_edit_basic(self, tools, workspace):
        result = tools["edit_file"].execute({
            "path": "hello.txt",
            "old_string": "line2",
            "new_string": "LINE_TWO",
        })
        assert "successfully" in result
        content = (workspace / "hello.txt").read_text()
        assert "LINE_TWO" in content
        assert "line2" not in content

    def test_edit_not_found(self, tools):
        result = tools["edit_file"].execute({
            "path": "hello.txt",
            "old_string": "nonexistent text",
            "new_string": "replacement",
        })
        assert "not found" in result

    def test_edit_multiple_matches(self, tools, workspace):
        (workspace / "dup.txt").write_text("aaa\naaa\n")
        result = tools["edit_file"].execute({
            "path": "dup.txt",
            "old_string": "aaa",
            "new_string": "bbb",
        })
        assert "matches 2 times" in result

    def test_edit_nonexistent_file(self, tools):
        result = tools["edit_file"].execute({
            "path": "nope.txt",
            "old_string": "x",
            "new_string": "y",
        })
        assert "Error" in result

    def test_edit_outside_home_blocked(self, tools):
        result = tools["edit_file"].execute({
            "path": "/etc/hosts",
            "old_string": "localhost",
            "new_string": "evil",
        })
        assert "Error" in result
        assert "outside home" in result

    def test_edit_ring2_main_blocked(self, tools, workspace):
        """edit_file should reject edits to ring2/main.py."""
        (workspace / "ring2").mkdir(exist_ok=True)
        (workspace / "ring2" / "main.py").write_text("original")
        result = tools["edit_file"].execute({
            "path": "ring2/main.py",
            "old_string": "original",
            "new_string": "modified",
        })
        assert "Error" in result
        assert "auto-evolved" in result
        # File unchanged
        assert (workspace / "ring2" / "main.py").read_text() == "original"

    def test_edit_ring0_blocked(self, tools, workspace):
        (workspace / "ring0").mkdir(exist_ok=True)
        (workspace / "ring0" / "memory.py").write_text("original")
        result = tools["edit_file"].execute({
            "path": "ring0/memory.py",
            "old_string": "original",
            "new_string": "modified",
        })
        assert "Error" in result
        assert "protected source" in result
        # File unchanged
        assert (workspace / "ring0" / "memory.py").read_text() == "original"


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

class TestListDir:
    def test_list_root(self, tools, workspace):
        result = tools["list_dir"].execute({})
        assert "hello.txt" in result
        assert "sub/" in result

    def test_list_subdir(self, tools):
        result = tools["list_dir"].execute({"path": "sub"})
        assert "nested.txt" in result

    def test_list_nonexistent(self, tools):
        result = tools["list_dir"].execute({"path": "nope"})
        assert "Error" in result

    def test_list_outside_home_blocked(self, tools):
        result = tools["list_dir"].execute({"path": "/etc"})
        assert "Error" in result

    def test_list_empty_dir(self, tools, workspace):
        (workspace / "empty").mkdir()
        result = tools["list_dir"].execute({"path": "empty"})
        assert "empty" in result.lower()

    def test_dirs_sorted_first(self, tools, workspace):
        """Directories should appear before files."""
        result = tools["list_dir"].execute({})
        lines = result.strip().splitlines()
        dir_indices = [i for i, l in enumerate(lines) if l.endswith("/")]
        file_indices = [i for i, l in enumerate(lines) if not l.endswith("/")]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)


# ---------------------------------------------------------------------------
# Output routing
# ---------------------------------------------------------------------------

class TestOutputRouting:
    """Test _route_output_path and write_file integration."""

    def test_json_routed_to_data(self, workspace):
        target = workspace / "output" / "result.json"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "result.json"

    def test_csv_routed_to_data(self, workspace):
        target = workspace / "output" / "table.csv"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "table.csv"

    def test_xml_routed_to_data(self, workspace):
        target = workspace / "output" / "feed.xml"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "feed.xml"

    def test_yaml_routed_to_data(self, workspace):
        target = workspace / "output" / "config.yaml"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "config.yaml"

    def test_yml_routed_to_data(self, workspace):
        target = workspace / "output" / "config.yml"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "config.yml"

    def test_pdf_routed_to_reports(self, workspace):
        target = workspace / "output" / "report.pdf"
        assert _route_output_path(workspace, target) == workspace / "output" / "reports" / "report.pdf"

    def test_py_routed_to_scripts(self, workspace):
        target = workspace / "output" / "analyze.py"
        assert _route_output_path(workspace, target) == workspace / "output" / "scripts" / "analyze.py"

    def test_sh_routed_to_scripts(self, workspace):
        target = workspace / "output" / "run.sh"
        assert _route_output_path(workspace, target) == workspace / "output" / "scripts" / "run.sh"

    def test_md_routed_to_docs(self, workspace):
        target = workspace / "output" / "notes.md"
        assert _route_output_path(workspace, target) == workspace / "output" / "docs" / "notes.md"

    def test_txt_routed_to_logs(self, workspace):
        target = workspace / "output" / "debug.txt"
        assert _route_output_path(workspace, target) == workspace / "output" / "logs" / "debug.txt"

    def test_log_routed_to_logs(self, workspace):
        target = workspace / "output" / "app.log"
        assert _route_output_path(workspace, target) == workspace / "output" / "logs" / "app.log"

    def test_unknown_extension_unchanged(self, workspace):
        target = workspace / "output" / "image.png"
        assert _route_output_path(workspace, target) == target

    def test_no_extension_unchanged(self, workspace):
        target = workspace / "output" / "Makefile"
        assert _route_output_path(workspace, target) == target

    def test_already_in_type_subdir(self, workspace):
        """Files already in a type subdir should not be double-nested."""
        target = workspace / "output" / "data" / "existing.json"
        assert _route_output_path(workspace, target) == target

    def test_already_in_reports_subdir(self, workspace):
        target = workspace / "output" / "reports" / "q1.pdf"
        assert _route_output_path(workspace, target) == target

    def test_non_output_path_unchanged(self, workspace):
        """Paths outside output/ are never routed."""
        target = workspace / "config" / "settings.json"
        assert _route_output_path(workspace, target) == target

    def test_task_subdir_preserved(self, workspace):
        """output/bitcoin/data.json → output/data/bitcoin/data.json"""
        target = workspace / "output" / "bitcoin" / "data.json"
        assert _route_output_path(workspace, target) == workspace / "output" / "data" / "bitcoin" / "data.json"

    def test_write_file_routes_output(self, tools, workspace):
        """Integration: write_file should auto-route output files."""
        result = tools["write_file"].execute(
            {"path": "output/report.json", "content": '{"key": "value"}'}
        )
        assert "Written" in result
        assert "output/data/report.json" in result
        assert (workspace / "output" / "data" / "report.json").read_text() == '{"key": "value"}'

    def test_write_file_non_output_unchanged(self, tools, workspace):
        """write_file for paths outside output/ should not route."""
        result = tools["write_file"].execute(
            {"path": "scratch/notes.json", "content": "{}"}
        )
        assert "Written" in result
        assert (workspace / "scratch" / "notes.json").read_text() == "{}"
