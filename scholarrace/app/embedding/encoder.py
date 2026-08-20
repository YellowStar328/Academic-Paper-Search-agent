"""Deterministic FakeEncoder using hash-based pseudo-embedding.

The FakeEncoder uses SHA-256 hashing to deterministically generate a
fixed-dimensional vector for any input text.  Two identical strings will
always produce the same vector, which is essential for testing and for the
embedding-based deduplication/reranking pipeline.

The encoder fills the vector by:
1. Hashing the input text with SHA-256 to get a deterministic byte stream.
2. Repeating the hash (with an incrementing counter appended) until enough
   bytes are generated to fill the vector dimension.
3. Converting bytes to float values in [0, 1] and then centering to [-1, 1].
4. L2-normalizing the final vector.

This is NOT a semantically meaningful embedding — it is a stand-in that
preserves the pipeline contract.  Texts that share common tokens tend to
produce somewhat correlated hash regions, giving a crude approximation of
textual overlap.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import get_settings


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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


