"""MMRSelector — Maximal Marginal Relevance diversity selection.

MMR iteratively selects papers that balance relevance to the query with
novelty relative to already-selected papers.  At each step, it picks:

    argmax  [ λ * sim(paper, query) - (1-λ) * max(sim(paper, selected)) ]

This reduces redundancy in the final results, ensuring diverse coverage
of the topic rather than a list of near-duplicate papers.

Reference: Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries".
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config import get_settings
from app.embedding.encoder import EmbeddingEncoder, FakeEncoder, cosine_similarity
from app.models.paper import Paper

logger = logging.getLogger(__name__)


class MMRSelector:
    """Maximal Marginal Relevance diversity selector.

    Parameters
    ----------
    encoder
        Embedding encoder for computing similarities.
    lambda_
        Trade-off between relevance and diversity.  1.0 = pure relevance,
        0.0 = pure diversity.  Default: 0.7.
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        lambda_: Optional[float] = None,
    ):
        settings = get_settings()
        self._encoder = encoder or FakeEncoder()
        self._lambda = lambda_ if lambda_ is not None else settings.mmr_lambda

    def select(
        self,
        papers: list[Paper],
        query: str,
        k: int,
        relevance_scores: Optional[list[float]] = None,
    ) -> list[Paper]:
        """Select k papers using MMR.

        Parameters
        ----------
        papers
            Candidate papers, already roughly ranked by relevance.
        query
            The user's search query.
        k
            Number of papers to select.
        relevance_scores
            Optional pre-computed relevance scores (same order as papers).
            If None, embedding similarity to query is used.
        """
        if not papers or k <= 0:
            return []

        k = min(k, len(papers))

        # Encode query
        query_vec = self._encoder.encode(query)

        # Encode all papers
        paper_vecs = [
            self._encoder.encode(f"{p.title} {p.abstract or ''}") for p in papers
        ]

        # Relevance scores
        if relevance_scores is None:
            relevance_scores = [
                cosine_similarity(query_vec, pv) for pv in paper_vecs
            ]

        # Selected indices and remaining indices
        selected: list[int] = []
        remaining = list(range(len(papers)))

        # Pick the first paper (highest relevance)
        first_idx = int(np.argmax(relevance_scores))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Iteratively pick remaining
        while len(selected) < k and remaining:
            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                # Relevance component
                rel = relevance_scores[idx]

                # Diversity component: max similarity to already selected
                max_sim = max(
                    cosine_similarity(paper_vecs[idx], paper_vecs[s])
                    for s in selected
                )

                mmr_score = self._lambda * rel - (1 - self._lambda) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        logger.info(
            f"MMRSelector: selected {len(selected)} papers from {len(papers)} "
            f"candidates (λ={self._lambda})"
        )

        return [papers[i] for i in selected]
