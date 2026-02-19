"""Embedding providers for semantic vector search.

Abstracts embedding generation behind a provider interface.
OpenAI implementation uses stdlib urllib (no pip packages).
LocalHash implementation uses pure stdlib (no API, no pip).
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import os
import re
import struct
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


# Regex: CJK Unified Ideographs range.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class LocalHashEmbedding(EmbeddingProvider):
    """Local token-hash embedding — pure stdlib, zero API cost.

    Tokenises text into Chinese character bigrams and English words,
    hashes each token to a position in a fixed-dimension vector,
    accumulates counts, and L2-normalises.  Good enough for
    deduplication (cosine > 0.85/0.95) since similar texts share tokens.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        # Normalise: lowercase, collapse whitespace.
        t = re.sub(r"\s+", " ", text.lower().strip())
        # Character-level n-grams (2 and 3) — language-agnostic,
        # captures partial overlap between paraphrases.
        for n in (2, 3):
            for i in range(len(t) - n + 1):
                gram = t[i : i + n]
                if not gram.isspace():
                    tokens.append(gram)
        # CJK individual characters (unigrams) — boost single-char overlap.
        tokens.extend(_CJK_RE.findall(t))
        # English/numeric whole words — capture exact word matches.
        tokens.extend(w for w in _WORD_RE.findall(t) if len(w) >= 2)
        return tokens

    def _hash_to_index(self, token: str) -> tuple[int, float]:
        """Hash token to (index, sign) pair using double-hashing."""
        h = hashlib.md5(token.encode("utf-8")).digest()
        idx = struct.unpack("<H", h[:2])[0] % self._dimensions
        sign = 1.0 if h[2] & 1 else -1.0
        return idx, sign

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx, sign = self._hash_to_index(token)
            vec[idx] += sign
        # L2 normalise.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def dimension(self) -> int:
        return self._dimensions


def create_embedding_provider(config: dict) -> EmbeddingProvider:
    """Factory: create an EmbeddingProvider from config dict.

    Config keys (under [ring1.embeddings]):
        provider: "openai" | "local" | "none" (default "none")
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

    if provider == "local":
        dimensions = emb_cfg.get("dimensions", 256)
        log.info("Using LocalHashEmbedding (dim=%d)", dimensions)
        return LocalHashEmbedding(dimensions=dimensions)

    return NoOpEmbedding()
