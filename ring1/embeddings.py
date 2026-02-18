"""Embedding providers for semantic vector search.

Abstracts embedding generation behind a provider interface.
OpenAI implementation uses stdlib urllib (no pip packages).
"""

from __future__ import annotations

import abc
import json
import logging
import os
import urllib.request

log = logging.getLogger("protea.embeddings")


class EmbeddingProvider(abc.ABC):
    """Abstract base for embedding providers."""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of texts into embedding vectors."""

    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""


class NoOpEmbedding(EmbeddingProvider):
    """Placeholder when no embedding provider is configured.

    embed() always returns an empty list.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    def dimension(self) -> int:
        return 0


class OpenAIEmbedding(EmbeddingProvider):
    """Generate embeddings via OpenAI API (text-embedding-3-small).

    Uses stdlib urllib.request — no openai package needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 256,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._url = "https://api.openai.com/v1/embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embedding API and return vectors.

        Returns empty list on failure (non-fatal).
        """
        if not texts:
            return []

        payload = json.dumps({
            "model": self._model,
            "input": texts,
            "dimensions": self._dimensions,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log.warning("OpenAI embedding API call failed: %s", exc)
            return []

        # Sort by index to preserve input order.
        data = body.get("data", [])
        data.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]

    def dimension(self) -> int:
        return self._dimensions


def create_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory: create an EmbeddingProvider from config dict.

    Config keys (under [ring1.embeddings]):
        provider: "openai" | "none" (default "none")
        api_key_env: environment variable name for API key
        model: embedding model name
        dimensions: vector dimensions
    """
    emb_cfg = config.get("ring1", {}).get("embeddings", {})
    provider = emb_cfg.get("provider", "none")

    if provider == "openai":
        key_env = emb_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_env, "")
        if not api_key:
            log.warning("Embedding provider 'openai' configured but %s not set", key_env)
            return NoOpEmbedding()
        model = emb_cfg.get("model", "text-embedding-3-small")
        dimensions = emb_cfg.get("dimensions", 256)
        return OpenAIEmbedding(api_key=api_key, model=model, dimensions=dimensions)

    return NoOpEmbedding()
