"""LLM client abstraction — ABC + factory for multi-provider support.

Defines the base interface that all LLM clients must implement,
and a factory function to instantiate the correct client by provider name.
"""

from __future__ import annotations

import abc
from typing import Callable


class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""


class LLMClient(abc.ABC):
    """Abstract base class for LLM API clients."""

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
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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
