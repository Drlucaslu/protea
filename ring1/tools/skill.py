"""Skill tool — let the LLM start a crystallized skill by name.

Pure stdlib.
"""

from __future__ import annotations

import logging
import time

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.skill")


def make_run_skill_tool(skill_store, skill_runner) -> Tool:
    """Create a Tool that starts a stored skill by name."""

    def _exec_run_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]

        # 1. Look up the skill in the store.
        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found."

        source_code = skill.get("source_code", "")
        if not source_code:
            return f"Error: skill '{skill_name}' has no source code."

        # 2. If the same skill is already running, return its current status.
        if skill_runner.is_running():
            info = skill_runner.get_info()
            if info and info.get("skill_name") == skill_name:
                output = skill_runner.get_output(max_lines=30)
                # Re-detect port in case it appeared after startup.
                parts = [f"Skill '{skill_name}' is already running (PID {info['pid']})."]
                if info.get("port"):
                    parts.append(f"HTTP port: {info['port']}")
                    parts.append(
                        f"Use view_skill to read the source code and find the correct API "
                        f"endpoints before calling web_fetch on http://localhost:{info['port']}."
                    )
                if output:
                    parts.append(f"\nRecent output:\n{output}")
                return "\n".join(parts)

            # Different skill running — stop it first.
            old_info = skill_runner.get_info()
            old_name = old_info["skill_name"] if old_info else "unknown"
            skill_runner.stop()
            log.info("Stopped previous skill '%s' to start '%s'", old_name, skill_name)

        # 3. Start the skill.
        dependencies = skill.get("dependencies") or None
        try:
            pid, message = skill_runner.run(skill_name, source_code, dependencies=dependencies)
        except Exception as exc:
            return f"Error starting skill '{skill_name}': {exc}"

        # 4. Update usage count.
        try:
            skill_store.update_usage(skill_name)
        except Exception:
            pass  # non-critical

        # 5. Wait for initialization and port detection.
        time.sleep(3)

        # 6. Collect status.
        info = skill_runner.get_info()
        output = skill_runner.get_output(max_lines=30)

        parts = [f"Skill '{skill_name}' started (PID {pid})."]

        if info and info.get("port"):
            parts.append(f"HTTP port: {info['port']}")
            parts.append(
                f"IMPORTANT: Use view_skill to read the source code and find the "
                f"correct API endpoints before calling web_fetch on http://localhost:{info['port']}."
            )

        if not skill_runner.is_running():
            parts.append("WARNING: Process exited shortly after starting.")

        if output:
            parts.append(f"\nInitial output:\n{output}")

        return "\n".join(parts)

    return Tool(
        name="run_skill",
        description=(
            "Start a stored Protea skill by name. Skills are standalone programs "
            "crystallized from successful evolution. Returns status, output, and "
            "HTTP port if available. Use web_fetch to interact with the skill's "
            "API after starting it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to start.",
                },
            },
            "required": ["skill_name"],
        },
        execute=_exec_run_skill,
    )


def _extract_skeleton(source: str) -> str:
    """Extract module docstring, imports, constants, and class/function signatures.

    Skips function bodies to keep the output compact while preserving
    all the information needed to understand the skill's API.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:3000] + "\n... (parse error, truncated)"

    lines = source.splitlines()
    parts: list[str] = []

    # Module docstring
    ds = ast.get_docstring(tree, clean=True)
    if ds:
        parts.append('"""' + ds + '"""')
        parts.append("")

    def _func_sig(node) -> str:
        start = node.lineno - 1
        sig_lines = []
        for i in range(start, min(start + 5, len(lines))):
            sig_lines.append(lines[i])
            if lines[i].rstrip().endswith(":"):
                break
        sig = "\n".join(sig_lines)
        fds = ast.get_docstring(node, clean=True)
        if fds:
            indent = "    " * (1 + (node.col_offset // 4))
            sig += f"\n{indent}" + '"""' + fds.split("\n")[0] + '"""'
        sig += "\n" + "    " * (1 + (node.col_offset // 4)) + "..."
        return sig

    def _class_sig(node) -> str:
        start = node.lineno - 1
        sig_lines = []
        for i in range(start, min(start + 3, len(lines))):
            sig_lines.append(lines[i])
            if lines[i].rstrip().endswith(":"):
                break
        sig = "\n".join(sig_lines)
        cds = ast.get_docstring(node, clean=True)
        if cds:
            sig += '\n    """' + cds.split("\n")[0] + '"""'
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig += "\n" + _func_sig(child)
        return sig

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(source, node)
            if seg:
                parts.append(seg)
        elif isinstance(node, ast.Assign):
            seg = ast.get_source_segment(source, node)
            if seg and len(seg) < 120:
                parts.append(seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(_func_sig(node))
        elif isinstance(node, ast.ClassDef):
            parts.append(_class_sig(node))

    return "\n".join(parts)


def make_view_skill_tool(skill_store) -> Tool:
    """Create a Tool that reads a stored skill's signature skeleton and metadata."""

    def _exec_view_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]
        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found."

        source = skill.get("source_code", "")
        skeleton = _extract_skeleton(source)

        parts = [
            f"Name: {skill.get('name', '')}",
            f"Description: {skill.get('description', '')}",
            f"Tags: {skill.get('tags', [])}",
            "",
            f"Source skeleton ({len(source)} chars full, use read_file for complete code):",
            "```python",
            skeleton,
            "```",
        ]
        return "\n".join(parts)

    return Tool(
        name="view_skill",
        description=(
            "View a skill's API skeleton: docstrings, imports, class/function signatures. "
            "Use read_file on the skill file for full source code."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to view.",
                },
            },
            "required": ["skill_name"],
        },
        execute=_exec_view_skill,
    )


def make_edit_skill_tool(skill_store) -> Tool:
    """Create a Tool that edits a stored skill's source code via search-and-replace."""

    def _exec_edit_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]
        old_string = inp["old_string"]
        new_string = inp["new_string"]

        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found."

        source_code = skill.get("source_code", "")

        count = source_code.count(old_string)
        if count == 0:
            return "Error: old_string not found in skill source code."
        if count > 1:
            return f"Error: old_string matches {count} times (must be unique)."

        updated = source_code.replace(old_string, new_string, 1)
        skill_store.update(skill_name, source_code=updated)
        return f"Skill '{skill_name}' updated successfully. Use run_skill to restart it with the new code."

    return Tool(
        name="edit_skill",
        description=(
            "Edit a stored skill's source code using search-and-replace. "
            "The old_string must appear exactly once in the source. "
            "After editing, use run_skill to restart the skill with updated code."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find (must be unique in the source).",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["skill_name", "old_string", "new_string"],
        },
        execute=_exec_edit_skill,
    )
