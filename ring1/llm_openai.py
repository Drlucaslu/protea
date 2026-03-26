"""OpenAI-compatible LLM client — covers OpenAI, DeepSeek, and similar APIs.

Pure stdlib (urllib.request + json).  Same retry pattern as the Anthropic client.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from ring1.llm_base import LLMClient, LLMError, compress_tool_results

log = logging.getLogger("protea.llm_openai")


def _convert_tool_schema(tool: dict) -> dict:
    """Convert an Anthropic-style tool definition to OpenAI function-calling format.

    Anthropic uses ``input_schema``; OpenAI uses ``parameters`` inside a
    ``function`` wrapper.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat completions client (no third-party deps)."""

    _LOG_PREFIX = "OpenAI-compat API"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.max_tokens = max_tokens
        self.api_url = api_url

    # ------------------------------------------------------------------
    # Internal: HTTP + retry
    # ------------------------------------------------------------------

    def _call_api(self, payload: dict, hard_timeout: float | None = None) -> dict:
        """POST *payload* to the chat completions endpoint with retry."""
        data = json.dumps(payload).encode("utf-8")
        # --- Debug: dump full payload to log file ---
        self._dump_payload(payload, len(data))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self._call_api_with_retry(self.api_url, data, headers,
                                         hard_timeout=hard_timeout)

    @staticmethod
    def _dump_payload(payload: dict, total_bytes: int) -> None:
        """Append a debug snapshot of the LLM request to data/llm_debug.jsonl."""
        import pathlib, time as _time
        debug_path = pathlib.Path("data/llm_debug.jsonl")
        try:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            messages = payload.get("messages", [])
            tools = payload.get("tools", [])
            # Measure each component
            system_msgs = [m for m in messages if m.get("role") == "system"]
            user_msgs = [m for m in messages if m.get("role") == "user"]
            asst_msgs = [m for m in messages if m.get("role") == "assistant"]
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            system_text = "\n".join(m.get("content", "") for m in system_msgs)
            user_text = "\n".join(
                m.get("content", "") if isinstance(m.get("content"), str)
                else json.dumps(m.get("content", ""), ensure_ascii=False)
                for m in user_msgs
            )
            tools_json = json.dumps(tools, ensure_ascii=False) if tools else ""
            entry = {
                "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": payload.get("model", ""),
                "max_tokens": payload.get("max_tokens", 0),
                "total_payload_bytes": total_bytes,
                "system_prompt_chars": len(system_text),
                "user_message_chars": len(user_text),
                "tools_schema_chars": len(tools_json),
                "num_tools": len(tools),
                "num_messages": len(messages),
                "msg_roles": [m.get("role", "?") for m in messages],
                "num_assistant_msgs": len(asst_msgs),
                "num_tool_result_msgs": len(tool_msgs),
                "system_prompt": system_text[:10000],
                "user_message": user_text[:20000],
                "tools_schema": tools_json[:10000] if tools_json else "",
            }
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # never break the API call

    # ------------------------------------------------------------------
    # Public: simple message (no tools)
    # ------------------------------------------------------------------

    def send_message(self, system_prompt: str, user_message: str,
                     hard_timeout: float | None = None) -> str:
        """Send a message and return the assistant's text response."""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        body = self._call_api(payload, hard_timeout=hard_timeout)
        self._reset_usage()
        usage = body.get("usage", {})
        self._add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        # Retry once on empty content (API occasionally returns no text).
        choices = body.get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            log.warning("%s empty content — retrying once", self._LOG_PREFIX)
            body = self._call_api(payload, hard_timeout=hard_timeout)
            usage = body.get("usage", {})
            self._add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        return self._extract_text(body)

    def send_message_ex(
        self, system_prompt: str, user_message: str,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        """Send a message and return (text, metadata) with stop_reason."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        body = self._call_api(payload)
        self._reset_usage()
        usage = body.get("usage", {})
        self._add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        choice = body.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content")

        # Retry once on empty content.
        if not text:
            log.warning("%s empty content — retrying once", self._LOG_PREFIX)
            body = self._call_api(payload)
            usage = body.get("usage", {})
            self._add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            choice = body.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content")
            if not text:
                raise LLMError("No text content in API response")

        finish = choice.get("finish_reason", "stop")
        stop_reason = "max_tokens" if finish == "length" else "end_turn"
        return text, {"stop_reason": stop_reason}

    # ------------------------------------------------------------------
    # Public: message with tool-call loop
    # ------------------------------------------------------------------

    def send_message_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 5,
    ) -> str:
        """Send a message and handle tool_calls rounds until a final text reply."""
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        openai_tools = [_convert_tool_schema(t) for t in tools]
        last_text: str = ""
        self._reset_usage()

        for _round_idx in range(max_rounds):
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "tools": openai_tools,
            }
            body = self._call_api(payload)
            usage = body.get("usage", {})
            self._add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

            # Compress tool results already seen by the LLM to save input tokens.
            n = compress_tool_results(messages)
            if n:
                log.debug("Compressed %d tool results in context", n)

            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Capture any text content.
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if not tool_calls or finish_reason != "tool_calls":
                # No more tool calls — return text.
                if content:
                    return content
                if last_text:
                    return last_text
                raise LLMError("No text content in API response")

            # Remember text from this round as fallback.
            if content:
                last_text = content

            # Append assistant message (must include tool_calls for the API).
            messages.append(message)

            # Execute each tool call and append results.
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_input = {}

                try:
                    result_str = tool_executor(tool_name, tool_input)
                except Exception as exc:
                    log.warning("Tool %s execution failed: %s", tool_name, exc)
                    result_str = f"Error: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        # max_rounds exhausted.
        log.warning("Tool use loop exhausted after %d rounds", max_rounds)
        if last_text:
            return last_text
        return (
            "I ran out of tool-call budget before finishing. "
            "The task may be partially complete — please check and retry if needed."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Extract text from a chat completions response."""
        choices = body.get("choices", [])
        if not choices:
            raise LLMError("No choices in API response")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise LLMError("No text content in API response")
        return content
