"""Tests for ring1.llm_base — ABC and factory function."""

import pytest

from ring1.llm_base import LLMClient, LLMError, create_llm_client


class TestLLMClientABC:
    def test_cannot_instantiate(self):
        """LLMClient is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


class TestCreateLLMClient:
    def test_anthropic_returns_claude_client(self):
        from ring1.llm_client import ClaudeClient

        client = create_llm_client(
            provider="anthropic", api_key="sk-test", model="claude-test",
        )
        assert isinstance(client, ClaudeClient)
        assert isinstance(client, LLMClient)

    def test_openai_returns_openai_client(self):
        from ring1.llm_openai import OpenAIClient

        client = create_llm_client(
            provider="openai", api_key="sk-test", model="gpt-4o",
        )
        assert isinstance(client, OpenAIClient)
        assert isinstance(client, LLMClient)
        assert client.api_url == "https://api.openai.com/v1/chat/completions"

    def test_deepseek_returns_openai_client(self):
        from ring1.llm_openai import OpenAIClient

        client = create_llm_client(
            provider="deepseek", api_key="sk-test", model="deepseek-chat",
        )
        assert isinstance(client, OpenAIClient)
        assert client.api_url == "https://api.deepseek.com/v1/chat/completions"

    def test_qwen_returns_openai_client(self):
        from ring1.llm_openai import OpenAIClient

        client = create_llm_client(
            provider="qwen", api_key="sk-test", model="qwen3.5-plus",
        )
        assert isinstance(client, OpenAIClient)
        assert client.api_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def test_custom_api_url(self):
        client = create_llm_client(
            provider="openai",
            api_key="sk-test",
            model="gpt-4o",
            api_url="http://localhost:8080/v1/chat/completions",
        )
        from ring1.llm_openai import OpenAIClient

        assert isinstance(client, OpenAIClient)
        assert client.api_url == "http://localhost:8080/v1/chat/completions"

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            create_llm_client(
                provider="gemini", api_key="sk-test", model="gemini-pro",
            )

    def test_llm_error_importable_from_llm_client(self):
        """LLMError should still be importable from ring1.llm_client."""
        from ring1.llm_client import LLMError as LLMErrorCompat

        assert LLMErrorCompat is LLMError
