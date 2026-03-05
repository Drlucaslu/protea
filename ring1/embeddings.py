"""Embedding providers for semantic search and context ranking.

Supports local sentence-transformers and a no-op fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbedding(EmbeddingProvider):
    """sentence-transformers local model, supports Chinese + English."""

    MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL)
        log.info("LocalEmbedding loaded: %s (dim=%d)", self.MODEL, self.dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def dimension(self) -> int:
        return 384


class NoOpEmbedding(EmbeddingProvider):
    """Fallback when no embedding model is available. Returns zero vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384] * len(texts)

    def dimension(self) -> int:
        return 384


def create_embedding_provider(config) -> EmbeddingProvider:
    """Create an embedding provider from config.

    Args:
        config: Object with optional `embedding_provider` attribute.
                Values: "local" (default), "none".
    """
    provider = getattr(config, "embedding_provider", "local")
    if provider == "local":
        try:
            return LocalEmbedding()
        except ImportError:
            log.warning("sentence-transformers not installed — using NoOpEmbedding")
            return NoOpEmbedding()
        except Exception:
            log.warning("Failed to load local embedding model", exc_info=True)
            return NoOpEmbedding()
    return NoOpEmbedding()
