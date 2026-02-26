"""Message tool — lets the LLM proactively send Telegram messages.

Pure stdlib.
"""

from __future__ import annotations

import logging
import threading

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.message")


def make_message_tool(reply_fn, reply_fn_factory=None) -> Tool:
    """Create a Tool that sends a Telegram message via *reply_fn*.

    Args:
        reply_fn: Callable(text: str) -> None to send a message (fallback).
        reply_fn_factory: Optional factory(chat_id, reply_to_id) -> reply_fn for task-specific routing.
    """

    def _exec_message(inp: dict) -> str:
        text = inp["text"]
        
        # Get chat_id from task context (set by TaskExecutor)
        task_chat_id = getattr(threading.current_thread(), "task_chat_id", "")
        
        # Use task-specific reply_fn if available
        actual_reply_fn = reply_fn
        if reply_fn_factory and task_chat_id:
            try:
                # Get reply_to_message_id from thread context if available
                reply_to_id = getattr(threading.current_thread(), "reply_to_message_id", None)
                actual_reply_fn = reply_fn_factory(task_chat_id, reply_to_id)
                log.info("Message tool: using task-specific reply_fn for chat_id=%s", task_chat_id)
            except Exception as exc:
                log.warning("Message tool: failed to create task-specific reply_fn, using fallback: %s", exc)
        else:
            if task_chat_id:
                log.warning("Message tool: no reply_fn_factory available, using fallback reply_fn for chat_id=%s", task_chat_id)
            else:
                log.warning("Message tool: no task_chat_id in thread context, using fallback reply_fn")
        
        try:
            actual_reply_fn(text)
        except Exception as exc:
            log.warning("Message tool send failed: %s", exc)
            return f"Error sending message: {exc}"
        return "Message sent"

    return Tool(
        name="message",
        description=(
            "Send a message to the user via Telegram.  Use this during "
            "multi-step tasks to report progress or intermediate results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message text to send.",
                },
            },
            "required": ["text"],
        },
        execute=_exec_message,
    )
