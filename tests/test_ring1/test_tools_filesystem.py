"""Tests for ring1.tools.filesystem."""

from __future__ import annotations

import os
import pathlib

import pytest

from ring1.tools.filesystem import (
    _check_write_allowed,
    _resolve_safe,
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
