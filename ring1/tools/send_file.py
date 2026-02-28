"""Send-file tool — lets the LLM deliver files to the user via Telegram.

Pure stdlib.
"""

from __future__ import annotations

import logging
import pathlib
import threading

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.send_file")

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB Telegram limit


def make_send_file_tool(send_file_fn, workspace_path: str = ".") -> Tool:
    """Create a Tool that sends a local file to the user via Telegram.

    Args:
        send_file_fn: Callable(file_path: str, caption: str) -> bool
        workspace_path: Root workspace directory for resolving relative paths.
    """
    _ws = pathlib.Path(workspace_path)

    # Directories to search (in order) when a relative path isn't found as-is.
    _SEARCH_DIRS = [
        "output",
        "output/data",
        "output/reports",
        "output/scripts",
        "output/docs",
        "output/logs",
    ]

    def _exec_send_file(inp: dict) -> str:
        raw_path = inp["file_path"]
        caption = inp.get("caption", "")

        # Resolve the file — try as-is, then search output subdirectories.
        path = pathlib.Path(raw_path)
        if not path.is_absolute():
            path = _ws / raw_path
        if not path.is_file():
            for search_dir in _SEARCH_DIRS:
                alt = _ws / search_dir / raw_path
                if alt.is_file():
                    path = alt
                    break
            else:
                return f"Error: file not found: {raw_path}"

        file_size = path.stat().st_size
        if file_size > _MAX_FILE_SIZE:
            return f"Error: file too large ({file_size / 1024 / 1024:.1f} MB > 50 MB limit)"

        # Get chat_id from task context (set by TaskExecutor)
        task_chat_id = getattr(threading.current_thread(), "task_chat_id", "")
        
        try:
            ok = send_file_fn(str(path), caption, task_chat_id)
        except Exception as exc:
            log.warning("send_file tool error: %s", exc)
            return f"Error sending file: {exc}"

        if ok:
            return f"File sent: {path.name}"
        return f"Error: failed to send file {path.name}"

    return Tool(
        name="send_file",
        description=(
            "Send a file to the user via Telegram. Use this AFTER writing or "
            "generating any file (PDF, report, script, image, data) that the "
            "user needs. The user is remote and cannot access local files — "
            "you MUST use this tool to deliver files to them."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the file to send. Can be relative "
                        "(e.g. 'output/report.pdf') or absolute."
                    ),
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption for the file.",
                },
            },
            "required": ["file_path"],
        },
        execute=_exec_send_file,
    )
