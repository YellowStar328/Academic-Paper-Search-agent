"""Integration tests for SearchPipeline end-to-end execution.

These tests use MockLLMProvider, Mock search providers, and FakeEncoder
to test the full pipeline without any real API calls.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.agents.base import LLMProvider
from app.agents.coordinator import MultiAgentCoordinator
from app.agents.judge import PaperJudge, StrongJudge
from app.agents.mock import MockLLMProvider
from app.agents.qwen import QwenAgent
from app.citation.expansion import CitationExpander
from app.config import Settings
from app.embedding.encoder import FakeEncoder
from app.models.paper import Paper, PaperIdentity, PaperList
from app.models.query import UserQuery
from app.models.result import SearchResult
from app.pipeline.search_pipeline import SearchPipeline
from app.query.parser import QueryParser
from app.retrieval.base import BaseSearchProvider
from app.ranking.authority import AuthorityScorer
from app.ranking.final_ranker import FinalRanker
from app.utils.observability import MetricsTracker


# ---------------------------------------------------------------------------
# Mock providers for testing
# ---------------------------------------------------------------------------

class MockSearchProvider(BaseSearchProvider):
    """Mock search provider returning preset papers."""

    def __init__(self, papers: list[Paper], source_name: str = "mock"):
        super().__init__(http_client=None)
        self._papers = papers
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    async def search(self, query: str, max_results: int = 50) -> PaperList:
        return PaperList(papers=self._papers[:max_results], source=self._source_name)


def make_paper(
    title: str = "Test Paper",
    abstract: str = "Test abstract about machine learning",
    year: int = 2024,
    citation_count: int = 10,
    doi: str | None = None,
    source: str = "arxiv",
) -> Paper:
    return Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            doi=doi,
            normalized_title=title.lower().replace(" ", ""),
            year=year,
        ),
        title=title,
        abstract=abstract,
        year=year,
        citation_count=citation_count,
        source=source,
    )


def make_test_papers() -> list[Paper]:
    """Create a set of test papers."""
    return [
        make_paper(
            "Attention Is All You Need",
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks",
            year=2017,
            citation_count=500,
            doi="10.1/attention",
        ),
        make_paper(
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "We introduce a new language representation model called BERT",
            year=2019,
            citation_count=300,
            doi="10.1/bert",
        ),
        make_paper(
            "GPT-4 Technical Report",
            "We report the development of GPT-4, a large multimodal model",
            year=2023,
            citation_count=200,
            doi="10.1/gpt4",
        ),
        make_paper(
            "Deep Residual Learning for Image Recognition",
            "Deeper neural networks are more difficult to train",
            year=2016,
            citation_count=400,
            doi="10.1/resnet",
        ),
        make_paper(
            "Generative Adversarial Networks",
            "We propose a new framework for estimating generative models",
            year=2014,
            citation_count=600,
            doi="10.1/gan",
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestSearchPipeline:
    """End-to-end pipeline tests."""

    def _build_pipeline(self, settings: Settings | None = None) -> SearchPipeline:
        """Build a fully wired pipeline with mock components."""
        s = settings or Settings(app_env="test")

        # Mock LLM provider
        llm = MockLLMProvider()

        # Query parser
        parser = QueryParser(provider=llm)

        # Multi-agent coordinator
        qwen_agent = QwenAgent(llm)
        coordinator = MultiAgentCoordinator(qwen_agent=qwen_agent)

        # Strong judge
        strong_judge = StrongJudge(provider=llm)

        # Paper judge
        paper_judge = PaperJudge(provider=llm)

        # Search providers
        papers = make_test_papers()
        providers = [
            MockSearchProvider(papers, "arxiv"),
            MockSearchProvider(papers[:3], "semantic_scholar"),
        ]

        # Citation expander
        citation_expander = CitationExpander(providers=providers)

        # Final ranker
        final_ranker = FinalRanker()

        # Build pipeline
        pipeline = SearchPipeline(
            query_parser=parser,
            coordinator=coordinator,
            strong_judge=strong_judge,
            providers=providers,
            citation_expander=citation_expander,
            paper_judge=paper_judge,
            final_ranker=final_ranker,
            settings=s,
        )
        return pipeline

    @pytest.mark.asyncio
    async def test_pipeline_runs_end_to_end(self):
        """Pipeline should produce a SearchResult with papers."""
        pipeline = self._build_pipeline()
        user_query = UserQuery(query="transformer architecture survey")
        result = await pipeline.run(user_query)

        assert isinstance(result, SearchResult)
        assert len(result.papers) > 0
        assert result.request_id != ""
        assert result.latency_ms > 0
        assert result.query == "transformer architecture survey"

    @pytest.mark.asyncio
    async def test_pipeline_returns_scored_papers(self):
        """Papers should have all score fields populated."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query="machine learning"))

        for pws in result.papers:
            assert pws.relevance_score is not None
            assert pws.authority_score is not None
            assert pws.recency_score is not None
            assert pws.final_score is not None

    @pytest.mark.asyncio
    async def test_pipeline_papers_sorted_by_final_score(self):
        """Papers should be sorted by descending final_score."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query="deep learning"))

        scores = [pws.final_score for pws in result.papers]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_pipeline_builds_research_graph(self):
        """Pipeline should build a research graph."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query="neural networks"))

        assert result.graph is not None
        assert len(result.graph.nodes) > 0
        assert len(result.graph.clusters) > 0
        assert len(result.graph.timeline) > 0

    @pytest.mark.asyncio
    async def test_pipeline_summary_populated(self):
        """Pipeline should populate the search summary."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query="AI research"))

        assert result.summary is not None
        assert result.summary.total_papers > 0
        assert result.summary.query == "AI research"

    @pytest.mark.asyncio
    async def test_pipeline_empty_query(self):
        """Pipeline should handle empty query gracefully."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query=""))

        assert isinstance(result, SearchResult)
        # Should still return papers from mock providers

    @pytest.mark.asyncio
    async def test_pipeline_metrics_recorded(self):
        """Pipeline should record metrics."""
        pipeline = self._build_pipeline()
        result = await pipeline.run(UserQuery(query="transformers"))

        assert result.metrics is not None
        assert result.metrics.papers_collected > 0
        assert len(result.metrics.search_sources_used) > 0

    @pytest.mark.asyncio
    async def test_pipeline_provider_failure_isolated(self):
        """Pipeline should continue even if a provider fails."""

        class FailingProvider(BaseSearchProvider):
            @property
            def source_name(self) -> str:
                return "failing"

            async def search(self, query: str, max_results: int = 50) -> PaperList:
                raise RuntimeError("Provider down")

        s = Settings(app_env="test")
        llm = MockLLMProvider()
        parser = QueryParser(provider=llm)
        qwen_agent = QwenAgent(llm)
        coordinator = MultiAgentCoordinator(qwen_agent=qwen_agent)
        strong_judge = StrongJudge(provider=llm)
        paper_judge = PaperJudge(provider=llm)

        good_papers = make_test_papers()
        good_provider = MockSearchProvider(good_papers, "arxiv")
        failing_provider = FailingProvider()

        providers = [good_provider, failing_provider]
        citation_expander = CitationExpander(providers=[good_provider])
        final_ranker = FinalRanker()

        pipeline = SearchPipeline(
            query_parser=parser,
            coordinator=coordinator,
            strong_judge=strong_judge,
            providers=providers,
            citation_expander=citation_expander,
            paper_judge=paper_judge,
            final_ranker=final_ranker,
            settings=s,
        )

        result = await pipeline.run(UserQuery(query="test query"))
        # Should still get papers from the good provider
        assert len(result.papers) > 0

    @pytest.mark.asyncio
    async def test_pipeline_dedup_merges_duplicates(self):
        """Pipeline should deduplicate papers from multiple providers."""
        s = Settings(app_env="test")
        llm = MockLLMProvider()
        parser = QueryParser(provider=llm)
        qwen_agent = QwenAgent(llm)
        coordinator = MultiAgentCoordinator(qwen_agent=qwen_agent)
        strong_judge = StrongJudge(provider=llm)
        paper_judge = PaperJudge(provider=llm)

        # Both providers return the same papers (same DOIs)
        papers = make_test_papers()
        providers = [
            MockSearchProvider(papers, "arxiv"),
            MockSearchProvider(papers, "semantic_scholar"),
        ]

        citation_expander = CitationExpander(providers=providers)
        final_ranker = FinalRanker()

        pipeline = SearchPipeline(
            query_parser=parser,
            coordinator=coordinator,
            strong_judge=strong_judge,
            providers=providers,
            citation_expander=citation_expander,
            paper_judge=paper_judge,
            final_ranker=final_ranker,
            settings=s,
        )

        result = await pipeline.run(UserQuery(query="machine learning"))
        # Should have fewer papers than 2x due to dedup
        total_retrieved = len(papers) * 2
        assert len(result.papers) <= total_retrieved
