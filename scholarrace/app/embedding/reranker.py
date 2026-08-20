"""EmbeddingReranker — coarse pre-ranking via embedding cosine similarity.

Given a user query and a list of candidate papers, the reranker:
1. Encodes the query into an embedding vector.
2. Encodes each paper (title + abstract) into an embedding vector.
3. Computes cosine similarity between query and each paper.
4. Returns the top-K papers sorted by similarity (descending).

This reduces the candidate pool from 1000+ to top-K (default 100) before
expensive LLM-based paper judging, saving token costs.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.embedding.encoder import EmbeddingEncoder, FakeEncoder, cosine_similarity
from app.models.paper import Paper

logger = logging.getLogger(__name__)


class EmbeddingReranker:
    """Coarse pre-ranking using embedding cosine similarity.

    Parameters
    ----------
    encoder
        Embedding encoder (default: FakeEncoder).
    top_k
        Number of top papers to return (default: from settings.embedding_top_k).
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        top_k: Optional[int] = None,
    ):
        settings = get_settings()
        self._encoder = encoder or FakeEncoder()
        self._top_k = top_k or settings.embedding_top_k

    @property
    def encoder(self) -> EmbeddingEncoder:
        return self._encoder

    def rerank(self, query: str, papers: list[Paper]) -> list[tuple[Paper, float]]:
        """Rerank papers by embedding similarity to the query.

        Returns a list of (paper, similarity_score) tuples, sorted by
        descending similarity, truncated to top_k.
        """
        if not papers:
            return []

        # Encode query
        query_vec = self._encoder.encode(query)

        # Encode papers and compute similarities
        scored: list[tuple[Paper, float]] = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract or ''}"
            paper_vec = self._encoder.encode(text)
            sim = cosine_similarity(query_vec, paper_vec)
            scored.append((paper, sim))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Truncate to top_k
        top = scored[: self._top_k]

        logger.info(
            f"EmbeddingReranker: {len(papers)} candidates → top {len(top)} "
            f"(sim range: {top[-1][1]:.4f} ~ {top[0][1]:.4f})"
            if top else "EmbeddingReranker: no candidates"
        )

        return top

    def rerank_papers(self, query: str, papers: list[Paper]) -> list[Paper]:
        """Rerank and return only the paper objects (without scores)."""
        return [p for p, _ in self.rerank(query, papers)]
