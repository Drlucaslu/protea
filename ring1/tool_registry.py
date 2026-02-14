"""Tool Registry — modular tool framework for LLM tool_use.

Defines a Tool dataclass and ToolRegistry that manages registration,
schema generation, and dispatch of tool calls.

Pure stdlib.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("protea.tool_registry")


@dataclass
class Tool:
    """A single tool that the LLM can call."""

    name: str
    description: str
    input_schema: dict  # Anthropic JSON Schema
    execute: Callable[[dict], str]


class ToolRegistry:
    """Registry of available tools for LLM tool_use."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.  Overwrites if name already exists."""
        self._tools[tool.name] = tool
        log.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name.  No-op if not found."""
        if name in self._tools:
            del self._tools[name]
            log.debug("Unregistered tool: %s", name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None."""
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """Return tool definitions in Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, tool_input: dict) -> str:
        """Execute a tool by name.  Returns error string for unknown tools."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        log.info("Tool call: %s(%s)", name, str(tool_input)[:120])
        result = tool.execute(tool_input)
        log.info("Tool result: %s (%d chars)", name, len(result))
        return result

    def clone_without(self, *names: str) -> ToolRegistry:
        """Return a new registry with the specified tools removed.

        Useful for subagent isolation (e.g. removing spawn/message).
        """
        new = ToolRegistry()
        for tool_name, tool in self._tools.items():
            if tool_name not in names:
                new.register(tool)
        return new

    def tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
