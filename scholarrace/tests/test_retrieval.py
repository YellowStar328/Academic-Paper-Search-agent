"""Tests for search providers, HTTP client, and PaperIdentityResolver."""

import asyncio
from uuid import uuid4

import httpx
import pytest
import respx

from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.arxiv import ArxivProvider
from app.retrieval.semantic_scholar import SemanticScholarProvider
from app.retrieval.openalex import OpenAlexProvider
from app.retrieval.pubmed import PubMedProvider
from app.retrieval.crossref import CrossrefProvider
from app.retrieval.resolver import PaperIdentityResolver
from app.utils.http_client import (
    CircuitBreaker,
    CircuitBreakerOpen,
    HttpClient,
    retry_with_backoff,
)


# ---------- HTTP Client Tests ----------

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_proceed() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_proceed() is False

    def test_record_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_half_open_after_timeout(self):
        import time

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.can_proceed() is True
        assert cb.state == "half_open"


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(func, max_retries=3, initial_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection failed")
            return "ok"

        result = await retry_with_backoff(func, max_retries=5, initial_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("always fails")

        with pytest.raises(httpx.ConnectError):
            await retry_with_backoff(func, max_retries=2, initial_delay=0.01)
        assert call_count == 3  # initial + 2 retries


class TestHttpClient:
    @pytest.mark.asyncio
    async def test_get_success(self):
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://api.test.com/data").mock(
                return_value=httpx.Response(200, json={"result": "ok"})
            )
            client = HttpClient(timeout=5.0, max_retries=1)
            response = await client.get("https://api.test.com/data")
            assert response.status_code == 200
            assert response.json() == {"result": "ok"}
            await client.close()

    @pytest.mark.asyncio
    async def test_post_success(self):
        with respx.mock(assert_all_called=False) as mock:
            mock.post("https://api.test.com/submit").mock(
                return_value=httpx.Response(200, json={"status": "accepted"})
            )
            client = HttpClient(timeout=5.0, max_retries=1)
            response = await client.post("https://api.test.com/submit", json={"data": 123})
            assert response.status_code == 200
            await client.close()


# ---------- arXiv Provider Tests ----------

class TestArxivProvider:
    @pytest.mark.asyncio
    @respx.mock
    async def test_search_returns_papers(self):
        """Test arXiv search with mock XML response."""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Deep Learning for Image Classification</title>
    <summary>We propose a new method for image classification using deep learning.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/pdf/2401.12345v1" title="pdf" type="application/pdf"/>
    <arxiv:doi>10.1234/test</arxiv:doi>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.LG"/>
  </entry>
</feed>"""
        respx.route(method="GET", url__startswith="http://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=mock_xml)
        )
        provider = ArxivProvider()
        result = await provider.search("deep learning", max_results=5)
        assert len(result) == 1
        paper = result.papers[0]
        assert paper.title == "Deep Learning for Image Classification"
        assert paper.arxiv_id == "2401.12345"
        assert paper.year == 2024
        assert paper.doi == "10.1234/test"
        assert "Alice Smith" in paper.authors
        assert "cs.CV" in paper.fields_of_study
        assert paper.pdf_url is not None
        await provider.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_empty_results(self):
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        respx.route(method="GET", url__startswith="http://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=mock_xml)
        )
        provider = ArxivProvider()
        result = await provider.search("nonexistent topic", max_results=5)
        assert len(result) == 0
        await provider.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_http_error_returns_empty(self):
        respx.route(method="GET", url__startswith="http://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(500, text="Server Error")
        )
        provider = ArxivProvider()
        result = await provider.search("test", max_results=5)
        assert len(result) == 0
        await provider.close()

    def test_extract_arxiv_id(self):
        provider = ArxivProvider()
        assert provider._extract_arxiv_id("http://arxiv.org/abs/2401.12345v1") == "2401.12345"
        assert provider._extract_arxiv_id("http://arxiv.org/abs/2401.12345") == "2401.12345"


# ---------- Mock Provider Tests ----------

class TestSemanticScholarProvider:
    @pytest.mark.asyncio
    async def test_search(self):
        provider = SemanticScholarProvider()
        result = await provider.search("attention", max_results=10)
        assert len(result) > 0
        assert all(p.source == "semantic_scholar" for p in result)
        await provider.close()

    @pytest.mark.asyncio
    async def test_search_no_match_returns_all(self):
        provider = SemanticScholarProvider()
        result = await provider.search("nonexistent_xyz", max_results=10)
        # Should return all mock papers (fallback behavior)
        assert len(result) > 0
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_paper(self):
        provider = SemanticScholarProvider()
        paper = await provider.get_paper("mock-s2-001")
        assert paper is not None
        assert paper.title == "Attention Is All You Need"
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_paper_missing(self):
        provider = SemanticScholarProvider()
        paper = await provider.get_paper("nonexistent-id")
        assert paper is None
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_citations(self):
        provider = SemanticScholarProvider()
        result = await provider.get_citations("mock-s2-001")
        assert len(result) > 0
        await provider.close()


class TestOpenAlexProvider:
    @pytest.mark.asyncio
    async def test_search(self):
        provider = OpenAlexProvider()
        result = await provider.search("residual", max_results=10)
        assert len(result) > 0
        assert all(p.source == "openalex" for p in result)
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_paper(self):
        provider = OpenAlexProvider()
        paper = await provider.get_paper("mock-openalex-001")
        assert paper is not None
        assert "ImageNet" in paper.title
        await provider.close()


class TestPubMedProvider:
    @pytest.mark.asyncio
    async def test_search(self):
        provider = PubMedProvider()
        result = await provider.search("medical", max_results=10)
        assert len(result) > 0
        assert all(p.source == "pubmed" for p in result)
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_paper(self):
        provider = PubMedProvider()
        paper = await provider.get_paper("mock-pubmed-001")
        assert paper is not None
        assert "medical imaging" in paper.title.lower()
        await provider.close()


class TestCrossrefProvider:
    @pytest.mark.asyncio
    async def test_search(self):
        provider = CrossrefProvider()
        result = await provider.search("graph neural", max_results=10)
        assert len(result) > 0
        assert all(p.source == "crossref" for p in result)
        await provider.close()

    @pytest.mark.asyncio
    async def test_get_paper(self):
        provider = CrossrefProvider()
        paper = await provider.get_paper("10.1000/mock-crossref-001")
        assert paper is not None
        assert "Graph Neural" in paper.title
        await provider.close()


# ---------- PaperIdentityResolver Tests ----------

class TestPaperIdentityResolver:
    def test_no_duplicates(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(paper_id="p1", identity=PaperIdentity(doi="10.1/a"), title="A"),
            Paper(paper_id="p2", identity=PaperIdentity(doi="10.1/b"), title="B"),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 2

    def test_dedup_by_doi(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(paper_id="p1", identity=PaperIdentity(doi="10.1/a"), title="Paper A", abstract="Short"),
            Paper(paper_id="p2", identity=PaperIdentity(doi="10.1/a"), title="Paper A Full", abstract="Longer abstract here"),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 1
        # Should merge: take longer title and abstract
        assert result[0].title == "Paper A Full"
        assert result[0].abstract == "Longer abstract here"

    def test_dedup_by_arxiv_id(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(arxiv_id="2401.12345", normalized_title="test", year=2024),
                title="Test Paper",
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(arxiv_id="2401.12345", normalized_title="test", year=2024),
                title="Test Paper Duplicate",
            ),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 1

    def test_dedup_by_s2_id(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(semantic_scholar_id="s2-123", normalized_title="test", year=2023),
                title="Test",
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(semantic_scholar_id="s2-123", normalized_title="test", year=2023),
                title="Test Better Title",
            ),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 1
        assert result[0].title == "Test Better Title"

    def test_dedup_by_title_year(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(normalized_title="sametitle", year=2020),
                title="Same Title",
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(normalized_title="sametitle", year=2020),
                title="Same Title",
            ),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 1

    def test_dedup_merges_authors(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                authors=["Alice", "Bob"],
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                authors=["Bob", "Charlie"],
            ),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 1
        assert set(result[0].authors) == {"Alice", "Bob", "Charlie"}

    def test_dedup_merges_fields(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                fields_of_study=["CS", "AI"],
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                fields_of_study=["AI", "ML"],
            ),
        ]
        result = resolver.resolve(papers)
        assert set(result[0].fields_of_study) == {"CS", "AI", "ML"}

    def test_dedup_takes_max_citations(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                citation_count=100,
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(doi="10.1/a"),
                title="A",
                citation_count=200,
            ),
        ]
        result = resolver.resolve(papers)
        assert result[0].citation_count == 200

    def test_resolve_empty(self):
        resolver = PaperIdentityResolver()
        assert resolver.resolve([]) == []

    def test_resolve_paper_list(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(paper_id="p1", identity=PaperIdentity(doi="10.1/a"), title="A"),
            Paper(paper_id="p2", identity=PaperIdentity(doi="10.1/a"), title="A"),
        ]
        plist = PaperList(papers=papers, source="test")
        result = resolver.resolve_paper_list(plist)
        assert len(result) == 1
        assert result.source == "test"

    def test_no_identity_key_uses_paper_id(self):
        """Papers with no identity should not be merged."""
        resolver = PaperIdentityResolver()
        papers = [
            Paper(paper_id="p1", identity=PaperIdentity(), title="A"),
            Paper(paper_id="p2", identity=PaperIdentity(), title="B"),
        ]
        result = resolver.resolve(papers)
        assert len(result) == 2

    def test_dedup_prefers_longer_title(self):
        resolver = PaperIdentityResolver()
        papers = [
            Paper(
                paper_id="p1",
                identity=PaperIdentity(doi="10.1/a"),
                title="Short",
            ),
            Paper(
                paper_id="p2",
                identity=PaperIdentity(doi="10.1/a"),
                title="A Much Longer and More Descriptive Title",
            ),
        ]
        result = resolver.resolve(papers)
        assert "Longer" in result[0].title
