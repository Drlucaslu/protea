"""Skill tool — let the LLM start a crystallized skill by name.

Pure stdlib.
"""

from __future__ import annotations

import logging
import time

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.skill")


def _try_hub_fallback(skill_store, registry_client, skill_name: str) -> dict | None:
    """Search the hub for a skill and install it locally if found."""
    if registry_client is None:
        return None
    try:
        results = registry_client.search(query=skill_name, limit=5)
        # Find exact match first, then prefix match.
        match = None
        for r in results:
            if r.get("name") == skill_name:
                match = r
                break
        if match is None:
            return None
        # Download full skill data.
        data = registry_client.download(match["node_id"], match["name"])
        if data is None:
            return None
        skill_store.install_from_hub(data)
        log.info("Installed skill '%s' from hub (node=%s)", skill_name, match["node_id"])
        return skill_store.get_by_name(skill_name)
    except Exception as exc:
        log.debug("Hub fallback for '%s' failed: %s", skill_name, exc)
        return None


def make_run_skill_tool(skill_store, skill_runner, registry_client=None) -> Tool:
    """Create a Tool that starts a stored skill by name."""

    def _exec_run_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]

        # 1. Look up the skill in the store, fallback to hub.
        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            skill = _try_hub_fallback(skill_store, registry_client, skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found (checked local store and hub)."

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
        try:
            pid, message = skill_runner.run(skill_name, source_code)
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


def make_view_skill_tool(skill_store, registry_client=None) -> Tool:
    """Create a Tool that reads a stored skill's source code and metadata."""

    def _exec_view_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]
        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            skill = _try_hub_fallback(skill_store, registry_client, skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found (checked local store and hub)."

        parts = [
            f"Name: {skill.get('name', '')}",
            f"Description: {skill.get('description', '')}",
            f"Tags: {skill.get('tags', [])}",
            "",
            "Source code:",
            "```python",
            skill.get("source_code", ""),
            "```",
        ]
        return "\n".join(parts)

    return Tool(
        name="view_skill",
        description=(
            "Read the source code and metadata of a stored Protea skill. "
            "Use this to inspect a skill before editing or debugging it."
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


def make_edit_skill_tool(skill_store, registry_client=None) -> Tool:
    """Create a Tool that edits a stored skill's source code via search-and-replace."""

    def _exec_edit_skill(inp: dict) -> str:
        skill_name = inp["skill_name"]
        old_string = inp["old_string"]
        new_string = inp["new_string"]

        skill = skill_store.get_by_name(skill_name)
        if skill is None:
            skill = _try_hub_fallback(skill_store, registry_client, skill_name)
        if skill is None:
            return f"Error: skill '{skill_name}' not found (checked local store and hub)."

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
