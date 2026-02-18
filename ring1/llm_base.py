"""LLM client abstraction — ABC + factory for multi-provider support.

Defines the base interface that all LLM clients must implement,
shared retry logic, and a factory function to instantiate the correct
client by provider name.
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

    def _call_api_with_retry(self, url: str, data: bytes, headers: dict) -> dict:
        """HTTP POST with exponential backoff retry.

        Retries on transient errors (status codes in ``_RETRYABLE_CODES``)
        and network/timeout errors.  Subclasses customize behaviour via
        ``_RETRYABLE_CODES``, ``_MAX_RETRIES``, ``_BASE_DELAY``, and
        ``_LOG_PREFIX`` class attributes.
        """
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
