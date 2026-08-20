"""Tests for CitationExpander and PaperIdentityResolver deduplication."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.citation.expansion import CitationExpander
from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider
from app.retrieval.resolver import PaperIdentityResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_paper(
    title: str = "Test Paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    year: int | None = 2024,
    citation_count: int = 0,
    paper_id: str | None = None,
) -> Paper:
    return Paper(
        paper_id=paper_id or str(uuid4()),
        identity=PaperIdentity(
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=s2_id,
            normalized_title=title.lower().replace(" ", ""),
            year=year,
        ),
        title=title,
        abstract="Test abstract",
        year=year,
        citation_count=citation_count,
        source="test",
    )


class MockCitationProvider(BaseSearchProvider):
    """Mock provider that returns preset citation/reference papers."""

    def __init__(self, citations: list[Paper] | None = None,
                 references: list[Paper] | None = None,
                 source_name: str = "mock_citation"):
        super().__init__(http_client=None)
        self._citations = citations or []
        self._references = references or []
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    async def search(self, query: str, max_results: int = 50) -> PaperList:
        return PaperList(source=self.source_name)

    async def get_citations(self, paper_id: str, max_results: int = 50) -> PaperList:
        return PaperList(papers=self._citations[:max_results], source=self.source_name)

    async def get_references(self, paper_id: str, max_results: int = 50) -> PaperList:
        return PaperList(papers=self._references[:max_results], source=self.source_name)


# ---------------------------------------------------------------------------
# PaperIdentityResolver tests
# ---------------------------------------------------------------------------

class TestPaperIdentityResolver:
    """Tests for PaperIdentityResolver deduplication."""

    def test_empty_list(self):
        resolver = PaperIdentityResolver()
        assert resolver.resolve([]) == []

    def test_no_duplicates(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("Paper A", doi="10.1/aaa")
        p2 = make_paper("Paper B", doi="10.1/bbb")
        result = resolver.resolve([p1, p2])
        assert len(result) == 2

    def test_doi_dedup(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("Attention Is All You Need", doi="10.1/aaa", citation_count=100)
        p2 = make_paper("Attention is all you need", doi="10.1/AAA", citation_count=200)
        result = resolver.resolve([p1, p2])
        assert len(result) == 1
        # Should take max citation count
        assert result[0].citation_count == 200

    def test_arxiv_dedup(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("Deep Learning", arxiv_id="2401.12345")
        p2 = make_paper("Deep Learning (Extended)", arxiv_id="2401.12345",
                        citation_count=50)
        result = resolver.resolve([p1, p2])
        assert len(result) == 1
        assert result[0].citation_count == 50

    def test_s2_id_dedup(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("BERT", s2_id="s2-001")
        p2 = make_paper("BERT Pre-training", s2_id="s2-001")
        result = resolver.resolve([p1, p2])
        assert len(result) == 1

    def test_title_year_fallback_dedup(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("GPT-4 Technical Report", year=2023)
        p2 = make_paper("GPT-4 technical report", year=2023)
        result = resolver.resolve([p1, p2])
        assert len(result) == 1

    def test_different_year_no_dedup(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("Transformers", year=2017)
        p2 = make_paper("Transformers", year=2023)
        result = resolver.resolve([p1, p2])
        assert len(result) == 2

    def test_doi_takes_priority_over_arxiv(self):
        """A paper with both DOI and arxiv_id should dedup by DOI."""
        resolver = PaperIdentityResolver()
        p1 = make_paper("Paper A", doi="10.1/aaa", arxiv_id="2401.001")
        p2 = make_paper("Paper A v2", doi="10.1/AAA")  # same DOI, no arxiv
        result = resolver.resolve([p1, p2])
        assert len(result) == 1

    def test_merge_unions_authors(self):
        resolver = PaperIdentityResolver()
        p1 = Paper(
            paper_id=str(uuid4()),
            identity=PaperIdentity(doi="10.1/aaa", normalized_title="test", year=2024),
            title="Test",
            authors=["Alice", "Bob"],
            year=2024,
            doi="10.1/aaa",
        )
        p2 = Paper(
            paper_id=str(uuid4()),
            identity=PaperIdentity(doi="10.1/aaa", normalized_title="test", year=2024),
            title="Test Extended",
            authors=["Bob", "Charlie"],
            year=2024,
            doi="10.1/aaa",
        )
        result = resolver.resolve([p1, p2])
        assert len(result) == 1
        authors = result[0].authors
        assert "Alice" in authors
        assert "Bob" in authors
        assert "Charlie" in authors

    def test_merge_unions_references(self):
        resolver = PaperIdentityResolver()
        p1 = Paper(
            paper_id=str(uuid4()),
            identity=PaperIdentity(doi="10.1/aaa", normalized_title="test", year=2024),
            title="Test",
            year=2024,
            doi="10.1/aaa",
            references=["ref1", "ref2"],
        )
        p2 = Paper(
            paper_id=str(uuid4()),
            identity=PaperIdentity(doi="10.1/aaa", normalized_title="test", year=2024),
            title="Test",
            year=2024,
            doi="10.1/aaa",
            references=["ref2", "ref3"],
        )
        result = resolver.resolve([p1, p2])
        assert len(result) == 1
        assert set(result[0].references) == {"ref1", "ref2", "ref3"}

    def test_resolve_paper_list(self):
        resolver = PaperIdentityResolver()
        p1 = make_paper("Paper A", doi="10.1/aaa")
        p2 = make_paper("Paper A v2", doi="10.1/AAA")
        pl = PaperList(papers=[p1, p2], source="test")
        result = resolver.resolve_paper_list(pl)
        assert len(result) == 1
        assert result.source == "test"


# ---------------------------------------------------------------------------
# CitationExpander tests
# ---------------------------------------------------------------------------

class TestCitationExpander:
    """Tests for CitationExpander depth=1."""

    @pytest.mark.asyncio
    async def test_expand_empty_input(self):
        provider = MockCitationProvider()
        expander = CitationExpander(providers=[provider])
        result = await expander.expand([])
        assert result == []

    @pytest.mark.asyncio
    async def test_expand_no_providers(self):
        """Expander with no providers should just return deduped input."""
        p1 = make_paper("Paper A", doi="10.1/aaa", citation_count=10)
        expander = CitationExpander(providers=[])
        result = await expander.expand([p1])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_expand_adds_citations_and_references(self):
        """Expanded papers should be merged into the result."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        cited = make_paper("Cited Paper", doi="10.1/cited", citation_count=50)
        refed = make_paper("Referenced Paper", doi="10.1/refed", citation_count=30)

        provider = MockCitationProvider(
            citations=[cited], references=[refed]
        )
        expander = CitationExpander(providers=[provider], max_papers_to_expand=5)
        result = await expander.expand([original])

        titles = {p.title for p in result}
        assert "Original" in titles
        assert "Cited Paper" in titles
        assert "Referenced Paper" in titles
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_expand_deduplicates(self):
        """If a citation is a duplicate of an existing paper, it should be merged."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        # Citation with same DOI as original (duplicate)
        dup = make_paper("Original Copy", doi="10.1/orig", citation_count=200)
        # Genuinely new paper
        new_paper = make_paper("New Paper", doi="10.1/new", citation_count=10)

        provider = MockCitationProvider(citations=[dup, new_paper])
        expander = CitationExpander(providers=[provider])
        result = await expander.expand([original])

        # Should be 2 unique papers (original+dup merged, plus new)
        assert len(result) == 2
        # Merged paper should have max citation count
        merged = [p for p in result if p.identity.doi == "10.1/orig"][0]
        assert merged.citation_count == 200

    @pytest.mark.asyncio
    async def test_expand_only_top_papers(self):
        """Only top N papers by citation_count should be expanded."""
        p1 = make_paper("High Cited", doi="10.1/high", citation_count=1000)
        p2 = make_paper("Low Cited", doi="10.1/low", citation_count=1)

        cited_from_high = make_paper("Cited by High", doi="10.1/ch", citation_count=50)
        provider = MockCitationProvider(citations=[cited_from_high])

        # Only expand 1 paper (the highest cited)
        expander = CitationExpander(providers=[provider], max_papers_to_expand=1)
        result = await expander.expand([p1, p2])

        # Should have original 2 + 1 expanded = 3
        assert len(result) == 3
        titles = {p.title for p in result}
        assert "Cited by High" in titles

    @pytest.mark.asyncio
    async def test_expand_provider_failure_isolated(self):
        """If a provider fails, the expander should continue with others."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)

        class FailingProvider(BaseSearchProvider):
            @property
            def source_name(self) -> str:
                return "failing"

            async def search(self, query: str, max_results: int = 50) -> PaperList:
                return PaperList(source=self.source_name)

            async def get_citations(self, paper_id: str,
                                   max_results: int = 50) -> PaperList:
                raise RuntimeError("API down")

            async def get_references(self, paper_id: str,
                                     max_results: int = 50) -> PaperList:
                raise RuntimeError("API down")

        good_cited = make_paper("Good Cited", doi="10.1/good", citation_count=50)
        good_provider = MockCitationProvider(citations=[good_cited])

        expander = CitationExpander(
            providers=[FailingProvider(), good_provider]
        )
        result = await expander.expand([original])

        titles = {p.title for p in result}
        assert "Original" in titles
        assert "Good Cited" in titles

    @pytest.mark.asyncio
    async def test_expand_multiple_providers(self):
        """Multiple providers' results should all be merged."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        cited_a = make_paper("Cited A", doi="10.1/a", citation_count=10)
        cited_b = make_paper("Cited B", doi="10.1/b", citation_count=20)

        provider_a = MockCitationProvider(
            citations=[cited_a], source_name="source_a"
        )
        provider_b = MockCitationProvider(
            citations=[cited_b], source_name="source_b"
        )

        expander = CitationExpander(providers=[provider_a, provider_b])
        result = await expander.expand([original])

        assert len(result) == 3
        titles = {p.title for p in result}
        assert "Cited A" in titles
        assert "Cited B" in titles

    @pytest.mark.asyncio
    async def test_expand_paper_list(self):
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        cited = make_paper("Cited Paper", doi="10.1/cited", citation_count=50)

        provider = MockCitationProvider(citations=[cited])
        expander = CitationExpander(providers=[provider])
        pl = PaperList(papers=[original], source="test")
        result = await expander.expand_paper_list(pl)

        assert result.source == "test"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_expand_respects_max_citations(self):
        """max_citations_per_paper should limit fetched citations."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        papers = [make_paper(f"Cited {i}", doi=f"10.1/c{i}") for i in range(10)]

        provider = MockCitationProvider(citations=papers)
        expander = CitationExpander(
            providers=[provider], max_citations_per_paper=3
        )
        result = await expander.expand([original])

        # original + 3 citations = 4
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_expand_respects_max_references(self):
        """max_references_per_paper should limit fetched references."""
        original = make_paper("Original", doi="10.1/orig", citation_count=100)
        papers = [make_paper(f"Ref {i}", doi=f"10.1/r{i}") for i in range(10)]

        provider = MockCitationProvider(references=papers)
        expander = CitationExpander(
            providers=[provider], max_references_per_paper=2
        )
        result = await expander.expand([original])

        # original + 2 references = 3
        assert len(result) == 3
