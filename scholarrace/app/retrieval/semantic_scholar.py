"""Semantic Scholar search provider — real API integration.

Semantic Scholar API is free (no key required for basic usage, but
rate-limited). API docs: https://api.semanticscholar.org/api-docs/

In test mode (APP_ENV=test), falls back to mock data for unit tests.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Optional
from uuid import uuid4

from app.config import get_settings
from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider, filter_papers_by_year
from app.utils.http_client import HttpClient

logger = logging.getLogger(__name__)

# Mock data for test mode
_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            semantic_scholar_id="mock-s2-001",
            normalized_title="attentionisallyouneed",
            year=2017,
        ),
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        venue="NeurIPS",
        semantic_scholar_id="mock-s2-001",
        url="https://www.semanticscholar.org/paper/mock-s2-001",
        citation_count=90000,
        fields_of_study=["Computer Science"],
        source="semantic_scholar",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            semantic_scholar_id="mock-s2-002",
            normalized_title="bertpretrainingofdeepbidirectionaltransformers",
            year=2019,
        ),
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        abstract="We introduce a new language representation model called BERT.",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        year=2019,
        venue="NAACL",
        semantic_scholar_id="mock-s2-002",
        url="https://www.semanticscholar.org/paper/mock-s2-002",
        citation_count=70000,
        fields_of_study=["Computer Science"],
        source="semantic_scholar",
    ),
]


class SemanticScholarProvider(BaseSearchProvider):
    """Semantic Scholar API search provider.

    Uses the Graph API endpoint: /graph/v1/paper/search
    Requires no API key for basic usage (rate-limited to ~100 req / 5 min).
    """

    # Fields to request from the API
    FIELDS = "paperId,title,abstract,year,authors,citationCount,venue,fieldsOfStudy,externalIds,openAccessPdf,url,publicationDate"

    def __init__(self, http_client: Optional[HttpClient] = None):
        settings = get_settings()
        super().__init__(
            http_client=http_client or HttpClient(timeout=settings.semantic_scholar_timeout),
            timeout=settings.semantic_scholar_timeout,
        )
        self.base_url = settings.semantic_scholar_base_url
        self.api_key = settings.semantic_scholar_api_key or None
        self.max_results = settings.semantic_scholar_max_results
        self._is_test = settings.is_test

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search Semantic Scholar for papers matching the query.

        Retries on 429 (rate limit) with exponential backoff.
        If year_start/year_end are provided, uses API-level year filtering
        (e.g. ``year=2012-2024``) and also post-filters results.
        """
        if self._is_test:
            papers = self._mock_search(query, max_results).papers
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            return PaperList(papers=papers, source=self.source_name)

        max_results = min(max_results, self.max_results)
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API max per page
            "fields": self.FIELDS,
        }
        # S2 API supports year range filtering: year=2012-2024
        if year_start is not None or year_end is not None:
            y_lo = year_start if year_start is not None else 1900
            y_hi = year_end if year_end is not None else 2100
            params["year"] = f"{y_lo}-{y_hi}"
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        max_retries = 6
        backoff = 5.0  # initial delay seconds (S2 is strict without API key)

        for attempt in range(max_retries + 1):
            try:
                response = await self._http_client.get(
                    self.base_url, params=params, headers=headers
                )
                if response.status_code == 429:
                    if attempt < max_retries:
                        retry_after = response.headers.get("retry-after")
                        delay = float(retry_after) if retry_after else backoff
                        logger.warning(
                            f"S2 API 429 rate limited, retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        await asyncio.sleep(delay)
                        backoff = min(backoff * 2, 30.0)
                        continue
                    logger.error("S2 API 429 rate limited, max retries exceeded")
                    return PaperList(source=self.source_name)

                if response.status_code in (403, 502, 503):
                    # 403 often happens when S2 blocks requests without
                    # API key under heavy rate-limiting. Retry with backoff.
                    if attempt < max_retries:
                        logger.warning(
                            f"S2 API {response.status_code}, retrying in "
                            f"{backoff:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30.0)
                        continue
                    logger.error(
                        f"Semantic Scholar API returned {response.status_code} "
                        f"after {max_retries + 1} attempts"
                    )
                    return PaperList(source=self.source_name)

                if response.status_code != 200:
                    logger.error(
                        f"Semantic Scholar API returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    return PaperList(source=self.source_name)

                import json

                data = json.loads(response.text)
                papers = []
                for item in data.get("data", []):
                    paper = self._parse_item(item)
                    if paper:
                        papers.append(paper)

                # Defensive: API-level year filter may not always work,
                # so also filter at result level.
                if year_start is not None or year_end is not None:
                    papers = filter_papers_by_year(papers, year_start, year_end)
                return PaperList(papers=papers, source=self.source_name)

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"S2 search error (attempt {attempt + 1}/{max_retries + 1}): {e}, "
                        f"retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                logger.error(f"Semantic Scholar search failed: {e}")
                return PaperList(source=self.source_name)

        return PaperList(source=self.source_name)

    def _mock_search(self, query: str, max_results: int) -> PaperList:
        """Mock search for test mode."""
        query_lower = query.lower()
        keywords = query_lower.split()
        matched = []
        for paper in _MOCK_PAPERS:
            title_lower = paper.title.lower()
            if any(k in title_lower for k in keywords):
                matched.append(paper)
        if not matched:
            matched = list(_MOCK_PAPERS)
        return PaperList(papers=matched[:max_results], source=self.source_name)

    def _parse_item(self, item: dict) -> Optional[Paper]:
        """Parse a Semantic Scholar API result item into a Paper."""
        try:
            title = item.get("title", "") or ""
            abstract = item.get("abstract", "") or ""
            year = item.get("year")

            # Publication date
            pub_date = None
            pub_date_str = item.get("publicationDate")
            if pub_date_str:
                try:
                    parts = pub_date_str.split("T")[0].split("-")
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    pub_date = date(y, m, d)
                except (ValueError, IndexError):
                    pass

            # Authors
            authors = []
            for a in item.get("authors", []):
                name = a.get("name", "")
                if name:
                    authors.append(name)

            # IDs from externalIds
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            s2_id = item.get("paperId")

            # URL
            url = item.get("url") or (
                f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""
            )

            # PDF URL
            open_access_pdf = item.get("openAccessPdf")
            pdf_url = open_access_pdf.get("url") if open_access_pdf else None

            # Citation count
            citation_count = item.get("citationCount", 0) or 0

            # Fields of study
            fields = item.get("fieldsOfStudy") or []

            # Venue
            venue = item.get("venue") or ""

            identity = PaperIdentity(
                doi=doi,
                arxiv_id=arxiv_id,
                semantic_scholar_id=s2_id,
                normalized_title=title.lower().replace(" ", ""),
                year=year,
            )

            return Paper(
                paper_id=str(uuid4()),
                identity=identity,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                publication_date=pub_date,
                doi=doi,
                arxiv_id=arxiv_id,
                semantic_scholar_id=s2_id,
                url=url,
                pdf_url=pdf_url,
                citation_count=citation_count,
                venue=venue,
                fields_of_study=fields,
                source=self.source_name,
            )
        except Exception as e:
            logger.error(f"Failed to parse Semantic Scholar item: {e}")
            return None

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by Semantic Scholar paper ID."""
        if self._is_test:
            for paper in _MOCK_PAPERS:
                if paper.semantic_scholar_id == paper_id:
                    return paper
            return None

        url = f"{self.base_url.rstrip('/search')}/{paper_id}"
        params = {"fields": self.FIELDS}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await self._http_client.get(url, params=params, headers=headers)
            if response.status_code != 200:
                return None
            import json

            item = json.loads(response.text)
            return self._parse_item(item)
        except Exception as e:
            logger.error(f"Semantic Scholar get_paper failed: {e}")
            return None

    async def get_citations(self, paper, max_results: int = 50) -> PaperList:
        """Get papers that cite this paper via Semantic Scholar API.

        Args:
            paper: a Paper object (or paper_id string for backward compat).
        """
        if self._is_test:
            pid = paper if isinstance(paper, str) else paper.paper_id
            return PaperList(
                papers=[p for p in _MOCK_PAPERS if p.semantic_scholar_id != pid],
                source=self.source_name,
            )

        # Extract S2 paperId from Paper object
        s2_id = None
        if isinstance(paper, str):
            s2_id = paper  # assume caller passed S2 ID directly
        else:
            s2_id = paper.semantic_scholar_id or getattr(paper, "identity", None) and paper.identity.semantic_scholar_id

        if not s2_id:
            logger.debug(f"S2 get_citations: no semantic_scholar_id for paper {getattr(paper, 'paper_id', paper)}")
            return PaperList(source=self.source_name)

        return await self._fetch_citations_or_refs(s2_id, "citations", max_results)

    async def get_references(self, paper, max_results: int = 50) -> PaperList:
        """Get papers referenced by this paper via Semantic Scholar API.

        Args:
            paper: a Paper object (or paper_id string for backward compat).
        """
        if self._is_test:
            pid = paper if isinstance(paper, str) else paper.paper_id
            return PaperList(
                papers=[p for p in _MOCK_PAPERS if p.semantic_scholar_id != pid],
                source=self.source_name,
            )

        s2_id = None
        if isinstance(paper, str):
            s2_id = paper
        else:
            s2_id = paper.semantic_scholar_id or getattr(paper, "identity", None) and paper.identity.semantic_scholar_id

        if not s2_id:
            logger.debug(f"S2 get_references: no semantic_scholar_id for paper {getattr(paper, 'paper_id', paper)}")
            return PaperList(source=self.source_name)

        return await self._fetch_citations_or_refs(s2_id, "references", max_results)

    async def _fetch_citations_or_refs(
        self, s2_id: str, direction: str, max_results: int
    ) -> PaperList:
        """Fetch citations or references from S2 Graph API.

        direction: "citations" or "references"
        """
        url = f"{self.base_url.rstrip('/search')}/{s2_id}/{direction}"
        params = {
            "fields": self.FIELDS,
            "limit": min(max_results, 100),
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = await self._http_client.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = backoff
                        logger.warning(f"S2 {direction} 429, retry in {delay:.1f}s")
                        await asyncio.sleep(delay)
                        backoff = min(backoff * 2, 30.0)
                        continue
                    return PaperList(source=self.source_name)

                if response.status_code != 200:
                    logger.error(f"S2 {direction} API {response.status_code}: {response.text[:200]}")
                    return PaperList(source=self.source_name)

                import json
                data = json.loads(response.text)
                papers = []
                for item in data.get("data", []):
                    # citations: each item has a "citingPaper" key
                    # references: each item has a "citedPaper" key
                    paper_data = item.get("citingPaper") or item.get("citedPaper") or item
                    if paper_data:
                        paper = self._parse_item(paper_data)
                        if paper:
                            papers.append(paper)

                return PaperList(papers=papers, source=self.source_name)

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"S2 {direction} error (attempt {attempt+1}): {e}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                logger.error(f"S2 {direction} failed: {e}")
                return PaperList(source=self.source_name)

        return PaperList(source=self.source_name)
