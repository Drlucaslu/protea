"""Tests for ring1.embeddings — EmbeddingProvider."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from ring1.embeddings import (
    NoOpEmbedding,
    OpenAIEmbedding,
    create_embedding_provider,
)


class TestNoOpEmbedding:
    def test_embed_returns_empty(self):
        provider = NoOpEmbedding()
        assert provider.embed(["hello"]) == []

    def test_dimension_is_zero(self):
        provider = NoOpEmbedding()
        assert provider.dimension() == 0

    def test_embed_empty_list(self):
        provider = NoOpEmbedding()
        assert provider.embed([]) == []


class TestOpenAIEmbedding:
    def test_dimension(self):
        provider = OpenAIEmbedding(api_key="test", dimensions=256)
        assert provider.dimension() == 256

    def test_embed_empty_input(self):
        provider = OpenAIEmbedding(api_key="test")
        assert provider.embed([]) == []

    @patch("ring1.embeddings.urllib.request.urlopen")
    def test_embed_success(self, mock_urlopen):
        # Mock the API response
        response_data = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        provider = OpenAIEmbedding(api_key="test-key")
        result = provider.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("ring1.embeddings.urllib.request.urlopen")
    def test_embed_api_failure_returns_empty(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")
        provider = OpenAIEmbedding(api_key="test-key")
        result = provider.embed(["hello"])
        assert result == []

    @patch("ring1.embeddings.urllib.request.urlopen")
    def test_embed_preserves_order(self, mock_urlopen):
        # API returns items out of order
        response_data = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        provider = OpenAIEmbedding(api_key="test-key")
        result = provider.embed(["first", "second"])
        assert result[0] == [0.1, 0.2]  # index 0 first
        assert result[1] == [0.4, 0.5]  # index 1 second


class TestCreateEmbeddingProvider:
    def test_default_none(self):
        provider = create_embedding_provider({})
        assert isinstance(provider, NoOpEmbedding)

    def test_explicit_none(self):
        cfg = {"ring1": {"embeddings": {"provider": "none"}}}
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, NoOpEmbedding)

    def test_openai_without_key(self):
        cfg = {"ring1": {"embeddings": {"provider": "openai", "api_key_env": "NONEXISTENT_KEY_XYZ"}}}
        with patch.dict("os.environ", {}, clear=False):
            provider = create_embedding_provider(cfg)
            assert isinstance(provider, NoOpEmbedding)

    def test_openai_with_key(self):
        cfg = {"ring1": {"embeddings": {"provider": "openai", "api_key_env": "TEST_OPENAI_KEY"}}}
        with patch.dict("os.environ", {"TEST_OPENAI_KEY": "sk-test"}):
            provider = create_embedding_provider(cfg)
            assert isinstance(provider, OpenAIEmbedding)
