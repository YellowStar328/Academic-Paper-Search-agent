"""Embedding encoders for text-to-vector conversion.

FakeEncoder: Deterministic hash-based pseudo-embedding for testing.
ApiEncoder: Real semantic embedding via OpenAI-compatible API (DashScope).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Encoder that converts text to a fixed-dimensional vector."""

    @property
    def dimension(self) -> int:
        ...

    def encode(self, text: str) -> np.ndarray:
        ...

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        ...


class FakeEncoder:
    """Deterministic hash-based embedding encoder.

    Generates a fixed-dimensional vector from SHA-256 hashing.  Same input
    always produces the same output vector.  The vector is L2-normalized so
    that cosine similarity can be computed via dot product.
    """

    def __init__(self, dimension: int | None = None):
        settings = get_settings()
        self._dimension = dimension or settings.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a deterministic vector."""
        # Generate enough bytes to fill the dimension
        bytes_needed = self._dimension
        collected: list[float] = []
        counter = 0
        while len(collected) < bytes_needed:
            data = f"{text}:{counter}".encode("utf-8")
            h = hashlib.sha256(data).digest()
            for b in h:
                if len(collected) < bytes_needed:
                    # Convert byte [0,255] to [-1, 1]
                    collected.append((b / 127.5) - 1.0)
            counter += 1

        vec = np.array(collected[: self._dimension], dtype=np.float32)

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]


class ApiEncoder:
    """Real semantic embedding via OpenAI-compatible API.

    Calls the /embeddings endpoint to get real semantic vectors.
    Falls back to FakeEncoder if the API is unavailable.
    """

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.embedding_api_base_url
        self._api_key = settings.embedding_api_key
        self._model = settings.embedding_api_model
        self._timeout = 60
        self._dimension: int | None = None
        self._fallback = FakeEncoder(dimension=settings.embedding_dim)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe the API to determine dimension
            try:
                vec = self.encode("dimension probe")
                self._dimension = len(vec)
            except Exception:
                self._dimension = self._fallback.dimension
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode text via API, with cache + fallback."""
        if text in self._cache:
            return self._cache[text]

        try:
            vec = self._call_api([text])[0]
            self._cache[text] = vec
            return vec
        except Exception as e:
            logger.warning(f"ApiEncoder failed, falling back to FakeEncoder: {e}")
            vec = self._fallback.encode(text)
            self._cache[text] = vec
            return vec

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts via API batch call."""
        # Check cache first
        results: list[np.ndarray | None] = [None] * len(texts)
        to_fetch: list[int] = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                to_fetch.append(i)

        if to_fetch:
            try:
                fetched = self._call_api([texts[i] for i in to_fetch])
                for idx, vec in zip(to_fetch, fetched):
                    results[idx] = vec
                    self._cache[texts[idx]] = vec
            except Exception as e:
                logger.warning(f"ApiEncoder batch failed, falling back: {e}")
                for idx in to_fetch:
                    vec = self._fallback.encode(texts[idx])
                    results[idx] = vec
                    self._cache[texts[idx]] = vec

        return [r if r is not None else self._fallback.encode("") for r in results]

    def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        """Call the embeddings API and return vectors."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": texts,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            vectors = []
            for emb in embeddings:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
            return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


