"""Filesystem tools — read, write, edit, list directory.

Paths can be absolute (anywhere within allowed boundaries) or relative
(resolved from *workspace*).  Access is allowed within the user's home
directory, excluding sensitive locations like ``~/.ssh``.

Pure stdlib.
"""

from __future__ import annotations

import logging
import os
import pathlib

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.filesystem")

# Paths under ~ that are never accessible (read or write).
_SENSITIVE_DIRS = frozenset({
    ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".config/gcloud",
    "Library/Keychains",
})

# Sensitive file names (exact match, case-insensitive) blocked anywhere.
_SENSITIVE_FILES = frozenset({
    ".env", ".netrc", ".npmrc", ".pypirc",
})


def _resolve_safe(workspace: pathlib.Path, path_str: str) -> pathlib.Path:
    """Resolve *path_str*, allowing access within workspace or user home.

    - Relative paths are resolved from *workspace* (always allowed).
    - Absolute paths and ``~/`` paths are accepted if they fall within
      the user's home directory.
    - Sensitive subdirectories (~/.ssh, ~/.gnupg, etc.) are always blocked.

    Raises ValueError if the path is outside allowed boundaries or
    touches a sensitive location.
    """
    home = pathlib.Path.home().resolve()
    workspace = workspace.resolve()

    # Handle ~ expansion and absolute paths.
    if path_str.startswith("~"):
        target = pathlib.Path(path_str).expanduser().resolve()
    elif os.path.isabs(path_str):
        target = pathlib.Path(path_str).resolve()
    else:
        target = (workspace / path_str).resolve()

    # Check 1: paths within workspace are always allowed.
    ws_str = str(workspace) + os.sep
    if target == workspace or str(target).startswith(ws_str):
        return target

    # Check 2: paths within user home directory are allowed.
    home_str = str(home) + os.sep
    if not (target == home or str(target).startswith(home_str)):
        raise ValueError(f"Path outside home directory: {path_str}")

    # Check sensitive directories within home.
    try:
        rel = target.relative_to(home)
    except ValueError:
        raise ValueError(f"Path outside home directory: {path_str}")

    rel_parts = rel.parts
    for sensitive in _SENSITIVE_DIRS:
        s_parts = pathlib.PurePosixPath(sensitive).parts
        if rel_parts[:len(s_parts)] == s_parts:
            raise ValueError(f"Access denied: ~/{sensitive} is a protected location")

    # Check sensitive file names directly in home.
    if target.name.lower() in _SENSITIVE_FILES and target.parent == home:
        raise ValueError(f"Access denied: ~/{target.name} is a protected file")

    return target


def make_filesystem_tools(workspace_path: str) -> list[Tool]:
    """Create Tool instances for filesystem operations."""
    workspace = pathlib.Path(workspace_path).resolve()

    # -- read_file --------------------------------------------------------

    def _exec_read(inp: dict) -> str:
        try:
            target = _resolve_safe(workspace, inp["path"])
        except ValueError as exc:
            return f"Error: {exc}"

        if not target.is_file():
            return f"Error: not a file: {inp['path']}"

        try:
            text = target.read_text(errors="replace")
        except Exception as exc:
            return f"Error reading file: {exc}"

        lines = text.splitlines(keepends=True)
        offset = inp.get("offset", 0)
        limit = inp.get("limit", len(lines))
        selected = lines[offset : offset + limit]

        # Add line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>6}\t{line}")
        return "".join(numbered)

    read_file = Tool(
        name="read_file",
        description=(
            "Read a file's contents with line numbers.  Supports offset and "
            "limit for partial reads of large files.  Accepts relative paths "
            "(from workspace), absolute paths, or ~/… paths within the user's "
            "home directory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative, absolute, or ~/…).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset (0-based, default 0).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (default: all).",
                },
            },
            "required": ["path"],
        },
        execute=_exec_read,
    )

    # -- write_file -------------------------------------------------------

    def _exec_write(inp: dict) -> str:
        try:
            target = _resolve_safe(workspace, inp["path"])
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(inp["content"])
        except Exception as exc:
            return f"Error writing file: {exc}"

        return f"Written {len(inp['content'])} bytes to {inp['path']}"

    write_file = Tool(
        name="write_file",
        description=(
            "Write content to a file (creates parent directories if needed). "
            "Paths are relative to workspace. Generated files (scripts, reports, "
            "data) should be written to output/ subdirectory, not the root. "
            "Overwrites existing content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative, absolute, or ~/…).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["path", "content"],
        },
        execute=_exec_write,
    )

    # -- edit_file --------------------------------------------------------

    def _exec_edit(inp: dict) -> str:
        try:
            target = _resolve_safe(workspace, inp["path"])
        except ValueError as exc:
            return f"Error: {exc}"

        if not target.is_file():
            return f"Error: not a file: {inp['path']}"

        try:
            text = target.read_text(errors="replace")
        except Exception as exc:
            return f"Error reading file: {exc}"

        old = inp["old_string"]
        new = inp["new_string"]

        count = text.count(old)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1:
            return f"Error: old_string matches {count} times (must be unique)"

        updated = text.replace(old, new, 1)
        try:
            target.write_text(updated)
        except Exception as exc:
            return f"Error writing file: {exc}"

        return "Edit applied successfully"

    edit_file = Tool(
        name="edit_file",
        description=(
            "Replace a unique string in a file.  The old_string must appear "
            "exactly once.  Accepts relative, absolute, or ~/… paths."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative, absolute, or ~/…).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find (must be unique).",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        execute=_exec_edit,
    )

    # -- list_dir ---------------------------------------------------------

    def _exec_list(inp: dict) -> str:
        path_str = inp.get("path", ".")
        try:
            target = _resolve_safe(workspace, path_str)
        except ValueError as exc:
            return f"Error: {exc}"

        if not target.is_dir():
            return f"Error: not a directory: {path_str}"

        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except Exception as exc:
            return f"Error listing directory: {exc}"

        lines = []
        for entry in entries:
            try:
                rel = entry.relative_to(workspace)
            except ValueError:
                rel = entry  # absolute path outside workspace
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{rel}{suffix}")

        if not lines:
            return "(empty directory)"
        return "\n".join(lines)

    list_dir = Tool(
        name="list_dir",
        description="List files and subdirectories.  Directories have a trailing /.  Accepts relative, absolute, or ~/… paths.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (relative, absolute, or ~/…).  Default '.'.",
                },
            },
            "required": [],
        },
        execute=_exec_list,
    )

    return [read_file, write_file, edit_file, list_dir]
