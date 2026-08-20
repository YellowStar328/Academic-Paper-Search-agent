"""FinalRanker — weighted scoring + MMR diversity for final paper ranking.

Combines multiple signals into a final score for each paper:
- relevance_score: from LLM paper judge
- authority_score: from AuthorityScorer
- recency_score: based on publication year
- citation_score: normalized citation count
- diversity_score: from MMR selection
- redundancy_score: penalty for near-duplicate papers

The final score is a weighted sum of all signals, configurable via Settings.
After scoring, MMR is applied to ensure diversity in the top-K results.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from app.config import get_settings
from app.embedding.encoder import EmbeddingEncoder, FakeEncoder
from app.models.candidate import PaperJudgeResult
from app.models.paper import Paper
from app.models.result import PaperWithScores
from app.ranking.authority import AuthorityScorer
from app.ranking.diversity import MMRSelector

logger = logging.getLogger(__name__)


class FinalRanker:
    """Final paper ranker combining all signals with MMR diversity.

    Parameters
    ----------
    authority_scorer
        AuthorityScorer instance (or default).
    mmr_selector
        MMRSelector instance (or default).
    encoder
        Embedding encoder (or default FakeEncoder).
    top_k
        Number of final papers to return.
    """

    def __init__(
        self,
        authority_scorer: Optional[AuthorityScorer] = None,
        mmr_selector: Optional[MMRSelector] = None,
        encoder: Optional[EmbeddingEncoder] = None,
        top_k: Optional[int] = None,
    ):
        settings = get_settings()
        self._authority = authority_scorer or AuthorityScorer()
        self._encoder = encoder or FakeEncoder()
        self._mmr = mmr_selector or MMRSelector(encoder=self._encoder)
        self._top_k = top_k or settings.final_top_k
        self._settings = settings

    def _compute_recency_score(self, paper: Paper) -> float:
        """Recency score based on publication year.

        Papers from the current year get 1.0, decreasing linearly for older
        papers, reaching 0.0 for papers >10 years old.
        """
        if paper.year is None:
            return 0.5
        current_year = datetime.now().year
        age = current_year - paper.year
        if age <= 0:
            return 1.0
        if age >= 10:
            return 0.0
        return 1.0 - (age / 10.0)

    def _compute_citation_score(self, paper: Paper) -> float:
        """Normalized citation score (log scale, saturating)."""
        if paper.citation_count <= 0:
            return 0.0
        return min(math.log(1 + paper.citation_count) / math.log(1 + 1000), 1.0)

    def _compute_redundancy_score(
        self, paper: Paper, all_papers: list[Paper]
    ) -> float:
        """Redundancy penalty: 1.0 = unique, 0.0 = fully redundant.

        Uses embedding similarity to detect near-duplicates.
        """
        from app.embedding.encoder import cosine_similarity

        if len(all_papers) <= 1:
            return 1.0

        paper_vec = self._encoder.encode(f"{paper.title} {paper.abstract or ''}")
        max_sim = 0.0
        for other in all_papers:
            if other.paper_id == paper.paper_id:
                continue
            other_vec = self._encoder.encode(
                f"{other.title} {other.abstract or ''}"
            )
            sim = cosine_similarity(paper_vec, other_vec)
            max_sim = max(max_sim, sim)

        # If max similarity is high, redundancy is high (score low)
        return max(1.0 - max_sim, 0.0)

    def rank(
        self,
        papers: list[Paper],
        query: str,
        judge_results: Optional[list[PaperJudgeResult]] = None,
    ) -> list[PaperWithScores]:
        """Rank papers and return top-K with scores.

        Parameters
        ----------
        papers
            Candidate papers (already deduplicated and embedding-reranked).
        query
            The user's search query.
        judge_results
            Optional LLM paper judge results.  If None, relevance_score
            defaults to embedding similarity.
        """
        if not papers:
            return []

        settings = self._settings
        s = settings

        # Build judge lookup
        judge_map: dict[str, PaperJudgeResult] = {}
        if judge_results:
            for jr in judge_results:
                judge_map[jr.paper_id] = jr

        # Compute embedding similarities for relevance fallback
        from app.embedding.encoder import cosine_similarity

        query_vec = self._encoder.encode(query)
        paper_vecs = {
            p.paper_id: self._encoder.encode(f"{p.title} {p.abstract or ''}")
            for p in papers
        }

        # Score all papers
        scored: list[PaperWithScores] = []
        for paper in papers:
            emb_sim = cosine_similarity(
                query_vec, paper_vecs[paper.paper_id]
            )

            # Relevance: use judge result if available, else embedding sim
            jr = judge_map.get(paper.paper_id)
            relevance = jr.relevance_score if jr else max(emb_sim, 0.0)

            authority = self._authority.score(paper, paper.source or "")
            recency = self._compute_recency_score(paper)
            citation = self._compute_citation_score(paper)
            redundancy = self._compute_redundancy_score(paper, papers)

            # Diversity score will be set after MMR selection
            diversity = 0.0

            final = (
                s.w_relevance * relevance
                + s.w_authority * authority
                + s.w_recency * recency
                + s.w_citation * citation
                + s.w_diversity * diversity
                + s.w_redundancy * redundancy
            )

            scored.append(
                PaperWithScores(
                    paper=paper,
                    relevance_score=relevance,
                    authority_score=authority,
                    recency_score=recency,
                    citation_score=citation,
                    diversity_score=diversity,
                    redundancy_score=redundancy,
                    final_score=final,
                    embedding_similarity=emb_sim,
                    judge_reasoning=jr.reasoning if jr else "",
                )
            )

        # Sort by initial final score
        scored.sort(key=lambda x: x.final_score, reverse=True)

        # Apply MMR on top candidates for diversity
        top_candidates = scored[: self._top_k * 2]  # over-select for MMR
        top_papers = [s.paper for s in top_candidates]
        relevance_scores = [s.relevance_score for s in top_candidates]

        mmr_papers = self._mmr.select(
            top_papers, query, self._top_k, relevance_scores
        )
        mmr_ids = {p.paper_id for p in mmr_papers}

        # Reorder scored list to match MMR order, set diversity score
        result: list[PaperWithScores] = []
        for i, pws in enumerate(top_candidates):
            if pws.paper.paper_id in mmr_ids:
                # Diversity score: higher for papers selected earlier by MMR
                pws.diversity_score = 1.0 - (
                    i / max(len(mmr_papers), 1)
                ) * 0.5

                # Recompute final score with diversity
                pws.final_score = (
                    s.w_relevance * pws.relevance_score
                    + s.w_authority * pws.authority_score
                    + s.w_recency * pws.recency_score
                    + s.w_citation * pws.citation_score
                    + s.w_diversity * pws.diversity_score
                    + s.w_redundancy * pws.redundancy_score
                )
                result.append(pws)

        # Sort MMR-selected papers by final score
        result.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(
            f"FinalRanker: ranked {len(papers)} papers → top {len(result)} "
            f"(score range: {result[-1].final_score:.4f} ~ {result[0].final_score:.4f})"
            if result else "FinalRanker: no papers to rank"
        )

        return result
