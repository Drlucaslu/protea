"""Tool package — factory for creating the default ToolRegistry.

Pure stdlib.
"""

from __future__ import annotations

from ring1.tool_registry import ToolRegistry


def create_default_registry(
    *,
    workspace_path: str = ".",
    shell_timeout: int = 30,
    reply_fn=None,
    subagent_manager=None,
    skill_store=None,
    skill_runner=None,
    registry_client=None,
    scheduled_store=None,
    send_file_fn=None,
) -> ToolRegistry:
    """Build a ToolRegistry with all standard tools.

    Args:
        workspace_path: Root directory for file/shell tools.
        shell_timeout: Default timeout in seconds for shell exec.
        reply_fn: Callable(text) for the message tool.  None disables it.
        subagent_manager: SubagentManager for the spawn tool.  None disables it.

    Returns:
        A fully populated ToolRegistry.
    """
    from ring1.tools.filesystem import make_filesystem_tools
    from ring1.tools.message import make_message_tool
    from ring1.tools.report import make_report_tool
    from ring1.tools.shell import make_shell_tool
    from ring1.tools.web import make_web_tools

    registry = ToolRegistry()

    for tool in make_web_tools():
        registry.register(tool)

    for tool in make_filesystem_tools(workspace_path):
        registry.register(tool)

    registry.register(make_shell_tool(workspace_path, timeout=shell_timeout))
    registry.register(make_report_tool(workspace_path))

    if reply_fn is not None:
        registry.register(make_message_tool(reply_fn))

    if subagent_manager is not None:
        from ring1.tools.spawn import make_spawn_tool
        registry.register(make_spawn_tool(subagent_manager))

    if skill_store is not None:
        from ring1.tools.skill import make_edit_skill_tool, make_view_skill_tool
        registry.register(make_view_skill_tool(skill_store, registry_client))
        registry.register(make_edit_skill_tool(skill_store, registry_client))

        if skill_runner is not None:
            from ring1.tools.skill import make_run_skill_tool
            registry.register(make_run_skill_tool(skill_store, skill_runner, registry_client))

    if scheduled_store is not None:
        from ring1.tools.schedule import make_schedule_tool
        registry.register(make_schedule_tool(scheduled_store))

    if send_file_fn is not None:
        from ring1.tools.send_file import make_send_file_tool
        registry.register(make_send_file_tool(send_file_fn))

    return registry
