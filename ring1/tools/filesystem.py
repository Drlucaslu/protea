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
    # Personal directories — privacy protection.
    "Documents", "Downloads", "Desktop", "Pictures", "Movies", "Music",
})

# Source directories that are read-only (relative to workspace).
# Task executor LLM must not modify core source code.
_READONLY_SOURCE_DIRS = frozenset({"ring0", "ring1", "tests"})

# Specific files that are never writable via task tools.
# ring2/main.py is auto-evolved code — only the evolution engine may modify it.
_PROTECTED_FILES = frozenset({
    "ring2/main.py",
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


# Auto-routing: map file extensions to output/ subdirectories.
_OUTPUT_TYPE_DIRS = {
    ".json": "data", ".csv": "data", ".xml": "data",
    ".yaml": "data", ".yml": "data",
    ".pdf": "reports",
    ".py": "scripts", ".sh": "scripts",
    ".md": "docs",
    ".txt": "logs", ".log": "logs",
}
_OUTPUT_TYPE_SUBDIRS = frozenset(_OUTPUT_TYPE_DIRS.values())


def _route_output_path(workspace: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Re-route files under workspace/output/ into type-based subdirectories.

    - Only applies to paths under ``workspace/output/``.
    - If already inside a known type subdir (e.g. ``output/data/``), return
      unchanged to avoid double-nesting.
    - Unknown or missing extensions are left unchanged.
    - Task subdirs are preserved: ``output/bitcoin/data.json`` →
      ``output/data/bitcoin/data.json``.
    """
    output_dir = workspace / "output"
    try:
        rel = target.relative_to(output_dir)
    except ValueError:
        return target  # not under output/

    # Already in a known type subdir?  e.g. output/data/foo.json
    if rel.parts and rel.parts[0] in _OUTPUT_TYPE_SUBDIRS:
        return target

    ext = target.suffix.lower()
    subdir = _OUTPUT_TYPE_DIRS.get(ext)
    if not subdir:
        return target  # unknown extension — leave in place

    return output_dir / subdir / rel


def _check_write_allowed(workspace: pathlib.Path, target: pathlib.Path) -> str | None:
    """Return an error message if *target* is inside a read-only source dir."""
    try:
        rel = target.relative_to(workspace)
    except ValueError:
        return None  # outside workspace — handled by _resolve_safe
    # Check specific protected files.
    rel_posix = rel.as_posix()
    if rel_posix in _PROTECTED_FILES:
        return f"Error: {rel} is auto-evolved code and cannot be modified by tasks"
    # Check read-only source dirs.
    top_dir = rel.parts[0] if rel.parts else ""
    if top_dir in _READONLY_SOURCE_DIRS:
        return f"Error: {rel} is in protected source directory '{top_dir}/' (read-only)"
    return None


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

        err = _check_write_allowed(workspace, target)
        if err:
            return err

        # Auto-route output files into type-based subdirectories.
        target = _route_output_path(workspace, target)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(inp["content"])
        except Exception as exc:
            return f"Error writing file: {exc}"

        # Report the actual path (may differ from input due to routing).
        try:
            actual = target.relative_to(workspace)
        except ValueError:
            actual = target
        return f"Written {len(inp['content'])} bytes to {actual}"

    write_file = Tool(
        name="write_file",
        description=(
            "Write content to a file (creates parent directories if needed). "
            "Paths are relative to workspace. Generated files should be written "
            "to output/ — they are auto-routed by extension into subdirectories: "
            "data/ (.json .csv .xml .yaml), reports/ (.pdf), scripts/ (.py .sh), "
            "docs/ (.md), logs/ (.txt .log). Overwrites existing content."
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

        err = _check_write_allowed(workspace, target)
        if err:
            return err

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
