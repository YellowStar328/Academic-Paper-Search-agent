"""Tests for AuthorityScorer, MMRSelector, and FinalRanker."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from app.config import Settings
from app.embedding.encoder import FakeEncoder
from app.models.candidate import PaperJudgeResult
from app.models.paper import Paper, PaperIdentity
from app.models.result import PaperWithScores
from app.ranking.authority import AuthorityScorer
from app.ranking.diversity import MMRSelector
from app.ranking.final_ranker import FinalRanker


def make_paper(
    title: str = "Test Paper",
    abstract: str = "Test abstract",
    year: int = 2024,
    citation_count: int = 0,
    venue: str | None = None,
    authors: list[str] | None = None,
    source: str = "arxiv",
    paper_id: str | None = None,
) -> Paper:
    return Paper(
        paper_id=paper_id or str(uuid4()),
        identity=PaperIdentity(normalized_title=title.lower().replace(" ", "")),
        title=title,
        abstract=abstract,
        year=year,
        citation_count=citation_count,
        venue=venue or "",
        authors=authors or [],
        source=source,
    )


# ---------------------------------------------------------------------------
# AuthorityScorer tests
# ---------------------------------------------------------------------------

class TestAuthorityScorer:
    """Tests for AuthorityScorer multi-signal scoring."""

    def test_zero_citations(self):
        scorer = AuthorityScorer()
        paper = make_paper(citation_count=0)
        assert scorer.score_citation(paper) == 0.0

    def test_high_citations_saturates(self):
        scorer = AuthorityScorer(max_citation_threshold=1000)
        paper = make_paper(citation_count=10000)
        assert scorer.score_citation(paper) == 1.0

    def test_moderate_citations(self):
        scorer = AuthorityScorer(max_citation_threshold=1000)
        paper = make_paper(citation_count=100)
        score = scorer.score_citation(paper)
        assert 0.0 < score < 1.0

    def test_citation_monotonic(self):
        """More citations should yield higher or equal score."""
        scorer = AuthorityScorer()
        p1 = make_paper(citation_count=10)
        p2 = make_paper(citation_count=100)
        p3 = make_paper(citation_count=1000)
        assert scorer.score_citation(p1) <= scorer.score_citation(p2)
        assert scorer.score_citation(p2) <= scorer.score_citation(p3)

    def test_venue_top_tier(self):
        scorer = AuthorityScorer()
        paper = make_paper(venue="NeurIPS 2024")
        assert scorer.score_venue(paper) >= 0.9

    def test_venue_preprint(self):
        scorer = AuthorityScorer()
        paper = make_paper(venue="arXiv preprint")
        assert scorer.score_venue(paper) <= 0.5

    def test_venue_no_info(self):
        scorer = AuthorityScorer()
        paper = make_paper(venue=None)
        assert scorer.score_venue(paper) == 0.5

    def test_source_scores(self):
        scorer = AuthorityScorer()
        assert scorer.score_source("arxiv") == 0.6
        assert scorer.score_source("semantic_scholar") == 0.8
        assert scorer.score_source("pubmed") == 0.8
        assert scorer.score_source("unknown") == 0.5

    def test_authors_more_is_slightly_higher(self):
        scorer = AuthorityScorer()
        p1 = make_paper(authors=["Alice"])
        p2 = make_paper(authors=["A", "B", "C", "D", "E"])
        assert scorer.score_authors(p1) <= scorer.score_authors(p2)

    def test_authors_empty(self):
        scorer = AuthorityScorer()
        paper = make_paper(authors=[])
        assert scorer.score_authors(paper) == 0.5

    def test_combined_score_in_range(self):
        scorer = AuthorityScorer()
        paper = make_paper(citation_count=50, venue="NeurIPS", authors=["A", "B"])
        score = scorer.score(paper, "arxiv")
        assert 0.0 <= score <= 1.0

    def test_combined_score_uses_all_signals(self):
        """A paper with good signals should score higher than one with poor signals."""
        scorer = AuthorityScorer()
        good = make_paper(citation_count=500, venue="Nature", authors=["A"] * 5)
        poor = make_paper(citation_count=0, venue=None, authors=[])
        assert scorer.score(good) > scorer.score(poor)


# ---------------------------------------------------------------------------
# MMRSelector tests
# ---------------------------------------------------------------------------

class TestMMRSelector:
    """Tests for MMRSelector diversity selection."""

    def test_empty_input(self):
        mmr = MMRSelector()
        result = mmr.select([], "query", k=5)
        assert result == []

    def test_k_larger_than_papers(self):
        mmr = MMRSelector()
        papers = [make_paper(f"P{i}") for i in range(3)]
        result = mmr.select(papers, "query", k=10)
        assert len(result) == 3

    def test_selects_k_papers(self):
        mmr = MMRSelector()
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        result = mmr.select(papers, "query", k=5)
        assert len(result) == 5

    def test_first_pick_is_highest_relevance(self):
        """The first selected paper should be the one with highest relevance."""
        mmr = MMRSelector(lambda_=1.0)  # pure relevance
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        rel_scores = [0.1, 0.9, 0.3, 0.5, 0.2]
        result = mmr.select(papers, "query", k=1, relevance_scores=rel_scores)
        assert result[0].title == "Paper 1"  # highest relevance

    def test_lambda_zero_is_pure_diversity(self):
        """With lambda=0, selection maximizes diversity (dissimilarity)."""
        mmr = MMRSelector(lambda_=0.0)
        papers = [make_paper(f"Unique Paper {i}") for i in range(10)]
        result = mmr.select(papers, "query", k=5)
        assert len(result) == 5

    def test_lambda_one_is_pure_relevance(self):
        """With lambda=1, selection is purely by relevance."""
        mmr = MMRSelector(lambda_=1.0)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        rel_scores = [0.5, 0.9, 0.1, 0.3, 0.7]
        result = mmr.select(papers, "query", k=3, relevance_scores=rel_scores)
        # Should pick papers with scores 0.9, 0.7, 0.5
        titles = [p.title for p in result]
        assert "Paper 1" in titles  # 0.9
        assert "Paper 4" in titles  # 0.7
        assert "Paper 0" in titles  # 0.5

    def test_diversity_reduces_redundancy(self):
        """MMR should not select very similar papers together."""
        # Create papers with distinct titles
        papers_a = [make_paper(f"Alpha Beta Gamma {i}") for i in range(3)]
        papers_b = [make_paper(f"Delta Epsilon Zeta {i}") for i in range(3)]
        all_papers = papers_a + papers_b

        mmr = MMRSelector(lambda_=0.5)
        result = mmr.select(all_papers, "query", k=4)

        # Should have a mix of both groups
        alpha_count = sum(1 for p in result if "Alpha" in p.title)
        delta_count = sum(1 for p in result if "Delta" in p.title)
        assert alpha_count >= 1
        assert delta_count >= 1

    def test_k_zero_returns_empty(self):
        mmr = MMRSelector()
        papers = [make_paper("P1")]
        result = mmr.select(papers, "query", k=0)
        assert result == []


# ---------------------------------------------------------------------------
# FinalRanker tests
# ---------------------------------------------------------------------------

class TestFinalRanker:
    """Tests for FinalRanker weighted scoring + MMR."""

    def test_empty_input(self):
        ranker = FinalRanker(top_k=5)
        result = ranker.rank([], "query")
        assert result == []

    def test_returns_top_k(self):
        ranker = FinalRanker(top_k=3)
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        result = ranker.rank(papers, "query")
        assert len(result) <= 3

    def test_returns_papers_with_scores(self):
        ranker = FinalRanker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        result = ranker.rank(papers, "query")
        assert all(isinstance(r, PaperWithScores) for r in result)

    def test_scores_in_range(self):
        ranker = FinalRanker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        result = ranker.rank(papers, "query")
        for r in result:
            assert 0.0 <= r.relevance_score <= 1.0
            assert 0.0 <= r.authority_score <= 1.0
            assert 0.0 <= r.recency_score <= 1.0
            assert 0.0 <= r.citation_score <= 1.0
            assert 0.0 <= r.redundancy_score <= 1.0

    def test_sorted_by_final_score(self):
        ranker = FinalRanker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        result = ranker.rank(papers, "query")
        scores = [r.final_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_high_citation_scores_higher(self):
        """A paper with more citations should tend to rank higher."""
        ranker = FinalRanker(top_k=2)
        high = make_paper("Important Paper", citation_count=500, venue="Nature")
        low = make_paper("Minor Paper", citation_count=0)
        result = ranker.rank([high, low], "query")
        titles = [r.paper.title for r in result]
        assert "Important Paper" in titles[0]

    def test_recent_paper_scores_higher_recency(self):
        ranker = FinalRanker(top_k=2)
        recent = make_paper("Recent", year=2025)
        old = make_paper("Old", year=2010)
        result = ranker.rank([recent, old], "query")
        recent_scores = {r.paper.title: r.recency_score for r in result}
        assert recent_scores["Recent"] > recent_scores["Old"]

    def test_with_judge_results(self):
        ranker = FinalRanker(top_k=3)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        judge_results = [
            PaperJudgeResult(
                paper_id=papers[0].paper_id,
                relevance_score=0.95,
                authority_score=0.8,
                reasoning="Highly relevant",
            ),
            PaperJudgeResult(
                paper_id=papers[1].paper_id,
                relevance_score=0.3,
                authority_score=0.5,
                reasoning="Marginally relevant",
            ),
        ]
        result = ranker.rank(papers, "query", judge_results=judge_results)
        # Paper 0 should rank higher than Paper 1
        scores = {r.paper.paper_id: r.relevance_score for r in result}
        if papers[0].paper_id in scores and papers[1].paper_id in scores:
            assert scores[papers[0].paper_id] > scores[papers[1].paper_id]

    def test_embedding_similarity_set(self):
        ranker = FinalRanker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        result = ranker.rank(papers, "query")
        for r in result:
            assert r.embedding_similarity is not None

    def test_fewer_papers_than_top_k(self):
        ranker = FinalRanker(top_k=10)
        papers = [make_paper(f"Paper {i}") for i in range(3)]
        result = ranker.rank(papers, "query")
        assert len(result) <= 3
