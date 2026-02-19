"""Claude API HTTP client — pure stdlib (urllib.request + json).

Sends messages to the Anthropic Messages API with retry + exponential backoff.
Supports both simple send_message and tool_use via send_message_with_tools.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from ring1.llm_base import LLMClient, LLMError, compress_tool_results  # noqa: F401 — re-export LLMError

log = logging.getLogger("protea.llm_client")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class ClaudeClient(LLMClient):
    """Minimal Claude Messages API client (no third-party deps)."""

    _RETRYABLE_CODES = {429, 500, 502, 503, 529}
    _LOG_PREFIX = "Claude API"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise LLMError("API key is not set")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Internal: HTTP + retry
    # ------------------------------------------------------------------

    def _call_api(self, payload: dict) -> dict:
        """Send *payload* to the Claude Messages API and return the JSON body."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
        }
        return self._call_api_with_retry(API_URL, data, headers)

    # ------------------------------------------------------------------
    # Public: simple message (no tools)
    # ------------------------------------------------------------------

    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message to Claude and return the assistant's text response."""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        body = self._call_api(payload)
        self._reset_usage()
        usage = body.get("usage", {})
        self._add_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        raise LLMError("No text content in API response")

    # ------------------------------------------------------------------
    # Public: message with tool_use loop
    # ------------------------------------------------------------------

    def send_message_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 5,
    ) -> str:
        """Send a message and handle tool_use rounds until a final text reply.

        Args:
            system_prompt: System prompt for Claude.
            user_message: Initial user message.
            tools: List of tool definitions (Claude API tool schema).
            tool_executor: Callable(name, input) -> result string.
            max_rounds: Maximum number of API round-trips.

        Returns:
            The final text response from Claude.
        """
        messages: list[dict] = [{"role": "user", "content": user_message}]
        # Track the last text seen in a tool_use round as a fallback for
        # when the loop exhausts without a final end_turn response.
        last_text_parts: list[str] = []
        self._reset_usage()

        for round_idx in range(max_rounds):
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
            }
            body = self._call_api(payload)
            usage = body.get("usage", {})
            self._add_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0))

            # Compress tool results already seen by the LLM to save input tokens.
            n = compress_tool_results(messages)
            if n:
                log.debug("Compressed %d tool results in context", n)

            content_blocks = body.get("content", [])
            stop_reason = body.get("stop_reason", "end_turn")

            # Collect text parts and tool_use blocks.
            text_parts: list[str] = []
            tool_uses: list[dict] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            # If no tool calls or stop_reason is not "tool_use", return text.
            if not tool_uses or stop_reason != "tool_use":
                if text_parts:
                    return "\n".join(text_parts)
                raise LLMError("No text content in API response")

            # Remember text from this round as fallback.
            if text_parts:
                last_text_parts = text_parts

            # Append assistant message with full content blocks.
            messages.append({"role": "assistant", "content": content_blocks})

            # Execute each tool and build tool_result blocks.
            tool_results: list[dict] = []
            for tu in tool_uses:
                tool_name = tu["name"]
                tool_input = tu.get("input", {})
                tool_id = tu["id"]
                try:
                    result_str = tool_executor(tool_name, tool_input)
                except Exception as exc:
                    log.warning("Tool %s execution failed: %s", tool_name, exc)
                    result_str = f"Error: {exc}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        # max_rounds exhausted — return last seen text or a friendly notice.
        log.warning("Tool use loop exhausted after %d rounds", max_rounds)
        if last_text_parts:
            return "\n".join(last_text_parts)
        return (
            "I ran out of tool-call budget before finishing. "
            "The task may be partially complete — please check and retry if needed."
        )
