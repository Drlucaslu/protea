"""Claude API HTTP client — pure stdlib (urllib.request + json).

Sends messages to the Anthropic Messages API with retry + exponential backoff.
Supports both simple send_message and tool_use via send_message_with_tools.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

log = logging.getLogger("protea.llm_client")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

_RETRYABLE_CODES = {429, 500, 502, 503, 529}
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


class LLMError(Exception):
    """Raised when the Claude API call fails after all retries."""


class ClaudeClient:
    """Minimal Claude Messages API client (no third-party deps)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise LLMError("CLAUDE_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Internal: HTTP + retry
    # ------------------------------------------------------------------

    def _call_api(self, payload: dict) -> dict:
        """Send *payload* to the Claude Messages API and return the JSON body.

        Retries on transient errors (429, 5xx) with exponential backoff.
        """
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    API_URL, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = exc.code
                if code in _RETRYABLE_CODES and attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "Claude API %d — retry %d/%d in %.1fs",
                        code, attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(
                    f"Claude API HTTP {code}: {exc.read().decode('utf-8', errors='replace')}"
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "Claude API network error — retry %d/%d in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"Claude API network error: {exc}") from exc

        raise LLMError(f"Claude API failed after {_MAX_RETRIES} retries") from last_error

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

        for _ in range(max_rounds):
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
            }
            body = self._call_api(payload)

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

        # max_rounds exhausted — return whatever text we have.
        if text_parts:
            return "\n".join(text_parts)
        raise LLMError("Tool use loop exhausted without final response")
