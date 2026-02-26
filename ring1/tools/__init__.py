"""Tool package — factory for creating the default ToolRegistry.

Pure stdlib.
"""

from __future__ import annotations

from ring1.tool_registry import ToolRegistry


def create_default_registry(
    *,
    workspace: str = ".",
    ring2_path = None,
    reply_fn = None,
    reply_fn_factory = None,
    skill_store = None,
    skill_runner = None,
    memory_store = None,
    task_store = None,
    spawn_fn = None,
    scheduled_store = None,
    send_file_fn = None,
) -> ToolRegistry:
    """Create a ToolRegistry with the default Protea tools.

    Args:
        workspace: Path to workspace directory for file operations.
        ring2_path: Path to Ring 2 code directory.
        reply_fn: Default function to send Telegram messages.
        reply_fn_factory: Optional factory(chat_id, reply_to_id) -> reply_fn for task-specific routing.
        skill_store: SkillStore for skill tools.
        skill_runner: SkillRunner for skill tools.
        memory_store: MemoryStore for memory recall.
        task_store: TaskStore for schedule management.
        spawn_fn: Callable for spawning background tasks.
        scheduled_store: ScheduledTaskStore for schedule tools.
        send_file_fn: Callable for sending files via Telegram.

    Returns:
        A ToolRegistry with all default tools registered.
    """
    registry = ToolRegistry()

    # Register file system tools
    from ring1.tools.filesystem import make_filesystem_tools
    for tool in make_filesystem_tools(workspace):
        registry.register(tool)

    # Register message tool with reply_fn_factory support
    if reply_fn is not None:
        from ring1.tools.message import make_message_tool
        registry.register(make_message_tool(reply_fn, reply_fn_factory))

    # Register shell tool
    from ring1.tools.shell import make_shell_tool
    registry.register(make_shell_tool(workspace))

    # Register web tools
    from ring1.tools.web import make_web_tools
    for tool in make_web_tools():
        registry.register(tool)

    # Register spawn tool
    if spawn_fn is not None:
        from ring1.tools.spawn import make_spawn_tool
        registry.register(make_spawn_tool(spawn_fn))

    # Register skill tools
    if skill_store is not None and skill_runner is not None:
        from ring1.tools.skill import (
            make_run_skill_tool,
            make_view_skill_tool,
            make_edit_skill_tool,
        )
        registry.register(make_run_skill_tool(skill_store, skill_runner))
        registry.register(make_view_skill_tool(skill_store))
        registry.register(make_edit_skill_tool(skill_store))

    # Register schedule tools
    if scheduled_store is not None:
        from ring1.tools.schedule import make_schedule_tool
        registry.register(make_schedule_tool(scheduled_store))

    # Register send_file tool
    if send_file_fn is not None:
        from ring1.tools.send_file import make_send_file_tool
        registry.register(make_send_file_tool(send_file_fn))

    return registry
