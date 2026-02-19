"""LLM client abstraction — ABC + factory for multi-provider support.

Defines the base interface that all LLM clients must implement,
and a factory function to instantiate the correct client by provider name.
"""

from __future__ import annotations

import abc
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

log = logging.getLogger("protea.llm_base")


class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""


class LLMClient(abc.ABC):
    """Abstract base class for LLM API clients."""

    _RETRYABLE_CODES: set[int] = {429, 500, 502, 503}
    _MAX_RETRIES: int = 3
    _BASE_DELAY: float = 2.0
    _LOG_PREFIX: str = "LLM API"

    def _call_api_with_retry(self, url: str, data: bytes, headers: dict) -> dict:
        """HTTP POST with exponential-backoff retry on transient errors."""
        last_error: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = exc.code
                if code in self._RETRYABLE_CODES and attempt < self._MAX_RETRIES - 1:
                    delay = self._BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "%s %d — retry %d/%d in %.1fs",
                        self._LOG_PREFIX, code, attempt + 1, self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(
                    f"{self._LOG_PREFIX} HTTP {code}: "
                    f"{exc.read().decode('utf-8', errors='replace')}"
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "%s network error — retry %d/%d in %.1fs",
                        self._LOG_PREFIX, attempt + 1, self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"{self._LOG_PREFIX} network error: {exc}") from exc
            except (TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "%s timeout — retry %d/%d in %.1fs",
                        self._LOG_PREFIX, attempt + 1, self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"{self._LOG_PREFIX} timeout: {exc}") from exc

        raise LLMError(
            f"{self._LOG_PREFIX} failed after {self._MAX_RETRIES} retries"
        ) from last_error

    @abc.abstractmethod
    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message and return the assistant's text response."""

    @abc.abstractmethod
    def send_message_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 5,
    ) -> str:
        """Send a message with tool-use loop and return the final text response."""


# Default API endpoints for each provider.
_DEFAULT_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
}


def create_llm_client(
    provider: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    api_url: str | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: One of "anthropic", "openai", "deepseek", "qwen".
        api_key: API key for the provider.
        model: Model name (e.g. "gpt-4o", "deepseek-chat").
        max_tokens: Maximum tokens for responses.
        api_url: Optional override for the API base URL.

    Returns:
        An LLMClient instance.

    Raises:
        LLMError: If the provider is unknown.
    """
    if provider == "anthropic":
        from ring1.llm_client import ClaudeClient

        return ClaudeClient(api_key=api_key, model=model, max_tokens=max_tokens)

    if provider in ("openai", "deepseek", "qwen"):
        from ring1.llm_openai import OpenAIClient

        url = api_url or _DEFAULT_URLS[provider]
        return OpenAIClient(
            api_key=api_key, model=model, max_tokens=max_tokens, api_url=url,
        )

    raise LLMError(f"Unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Tool result compression — reduces input tokens on multi-round tool loops.
# ---------------------------------------------------------------------------

TOOL_RESULT_COMPRESS_THRESHOLD = 1500  # chars; below this, keep as-is
_COMPRESS_HEAD = 300
_COMPRESS_TAIL = 200


def _compress_content(text: str, threshold: int = TOOL_RESULT_COMPRESS_THRESHOLD) -> str:
    """Truncate a tool result string if it exceeds *threshold*."""
    if len(text) <= threshold:
        return text
    omitted = len(text) - _COMPRESS_HEAD - _COMPRESS_TAIL
    return (
        text[:_COMPRESS_HEAD]
        + f"\n\n[... {omitted} chars omitted ...]\n\n"
        + text[-_COMPRESS_TAIL:]
    )


def compress_tool_results(
    messages: list[dict],
    threshold: int = TOOL_RESULT_COMPRESS_THRESHOLD,
) -> int:
    """Compress tool result content in-place. Returns count of results compressed.

    Handles both Anthropic format (role=user, content=[{type: tool_result}])
    and OpenAI format (role=tool, content=str).
    """
    compressed = 0
    for msg in messages:
        role = msg.get("role", "")
        # Anthropic: role=user with tool_result blocks
        if role == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        orig = block.get("content", "")
                        if isinstance(orig, str) and len(orig) > threshold:
                            block["content"] = _compress_content(orig, threshold)
                            compressed += 1
        # OpenAI: role=tool with content string
        elif role == "tool":
            orig = msg.get("content", "")
            if isinstance(orig, str) and len(orig) > threshold:
                msg["content"] = _compress_content(orig, threshold)
                compressed += 1
    return compressed
