"""Tests for ring1.llm_base — ABC and factory function."""

import pytest

from ring1.llm_base import (
    LLMClient,
    LLMError,
    TOOL_RESULT_COMPRESS_THRESHOLD,
    _compress_content,
    compress_tool_results,
    create_llm_client,
)


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
        assert client.api_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

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


# ---------------------------------------------------------------------------
# Tool result compression
# ---------------------------------------------------------------------------


class TestCompressContent:
    def test_short_text_unchanged(self):
        short = "x" * 1000
        assert _compress_content(short) == short

    def test_at_threshold_unchanged(self):
        text = "x" * TOOL_RESULT_COMPRESS_THRESHOLD
        assert _compress_content(text) == text

    def test_long_text_truncated(self):
        text = "A" * 300 + "B" * 2000 + "C" * 200
        result = _compress_content(text)
        assert result.startswith("A" * 300)
        assert result.endswith("C" * 200)
        assert "[... 2000 chars omitted ...]" in result
        assert len(result) < len(text)

    def test_custom_threshold(self):
        text = "x" * 100
        result = _compress_content(text, threshold=50)
        assert "[..." in result


class TestCompressToolResults:
    def test_empty_list(self):
        assert compress_tool_results([]) == 0

    def test_no_tool_results(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert compress_tool_results(messages) == 0

    def test_anthropic_format(self):
        big = "x" * 3000
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": big},
            ]},
        ]
        n = compress_tool_results(messages)
        assert n == 1
        compressed = messages[0]["content"][0]["content"]
        assert len(compressed) < len(big)
        assert "[..." in compressed

    def test_anthropic_format_short_unchanged(self):
        short = "x" * 100
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": short},
            ]},
        ]
        assert compress_tool_results(messages) == 0
        assert messages[0]["content"][0]["content"] == short

    def test_openai_format(self):
        big = "x" * 3000
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": big},
        ]
        n = compress_tool_results(messages)
        assert n == 1
        assert len(messages[0]["content"]) < len(big)
        assert "[..." in messages[0]["content"]

    def test_openai_format_short_unchanged(self):
        short = "x" * 100
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": short},
        ]
        assert compress_tool_results(messages) == 0
        assert messages[0]["content"] == short

    def test_mixed_messages(self):
        big = "x" * 3000
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": big},
                {"type": "tool_result", "tool_use_id": "t2", "content": "short"},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": big},
        ]
        n = compress_tool_results(messages)
        assert n == 2  # one anthropic + one openai

    def test_custom_threshold(self):
        text = "x" * 200
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": text},
        ]
        assert compress_tool_results(messages, threshold=100) == 1
        assert "[..." in messages[0]["content"]

    def test_idempotent(self):
        """Compressing already-compressed results should not compress again."""
        big = "x" * 3000
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": big},
        ]
        compress_tool_results(messages)
        first = messages[0]["content"]
        # Second pass — already below threshold, should be no-op.
        assert compress_tool_results(messages) == 0
        assert messages[0]["content"] == first


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


class TestUsageTracking:
    """LLMClient._reset_usage / _add_usage / last_usage."""

    def _make_client(self):
        """Create a concrete subclass for testing."""
        class DummyClient(LLMClient):
            def send_message(self, system_prompt, user_message):
                return ""
            def send_message_with_tools(self, system_prompt, user_message, tools, tool_executor, max_rounds=5):
                return ""
        return DummyClient()

    def test_last_usage_default_zero(self):
        client = self._make_client()
        assert client.last_usage == {"input_tokens": 0, "output_tokens": 0}

    def test_reset_usage(self):
        client = self._make_client()
        client._add_usage(100, 50)
        client._reset_usage()
        assert client.last_usage == {"input_tokens": 0, "output_tokens": 0}

    def test_add_usage_accumulates(self):
        client = self._make_client()
        client._reset_usage()
        client._add_usage(100, 50)
        client._add_usage(200, 75)
        assert client.last_usage == {"input_tokens": 300, "output_tokens": 125}

    def test_last_usage_returns_copy(self):
        client = self._make_client()
        client._reset_usage()
        client._add_usage(10, 20)
        u = client.last_usage
        u["input_tokens"] = 999
        assert client.last_usage["input_tokens"] == 10
