"""CrossRef search provider — real API integration.

CrossRef API is free and requires no authentication (but including
a mailto in the User-Agent gives access to the polite pool with
higher rate limits). API docs: https://api.crossref.org

In test mode (APP_ENV=test), falls back to mock data for unit tests.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional
from uuid import uuid4

from app.config import get_settings
from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider
from app.utils.http_client import HttpClient

logger = logging.getLogger(__name__)

# Mock data for test mode
_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
        doi="10.1000/mock-crossref-001",
        normalized_title="graphneuralnetworkframeworkforfeaturelearning",
            year=2023,
        ),
        title="Graph Neural Network Framework for Feature Learning",
        abstract="We propose a novel framework for graph neural networks.",
        authors=["Alice Zhang", "Bob Li"],
        year=2023,
        venue="Nature",
        doi="10.1000/mock-crossref-001",
        url="https://doi.org/10.1000/mock-crossref-001",
        citation_count=120,
        fields_of_study=["Computer Science"],
        source="crossref",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            doi="10.1126/science.abc1234",
            normalized_title="transformersurveycomprehensivereview",
            year=2024,
        ),
        title="Transformer Survey: A Comprehensive Review",
        abstract="A comprehensive survey of transformer architectures.",
        authors=["Charlie Wang", "Diana Chen"],
        year=2024,
        venue="Science",
        doi="10.1126/science.abc1234",
        url="https://doi.org/10.1126/science.abc1234",
        citation_count=89,
        fields_of_study=["Computer Science"],
        source="crossref",
    ),
]


class CrossrefProvider(BaseSearchProvider):
    """CrossRef API search provider.

    Uses the Works endpoint: https://api.crossref.org/works
    Free, no API key (mailto in User-Agent for polite pool).
    """

    def __init__(self, http_client: Optional[HttpClient] = None):
        settings = get_settings()
        super().__init__(
            http_client=http_client or HttpClient(timeout=settings.crossref_timeout),
            timeout=settings.crossref_timeout,
        )
        self.base_url = settings.crossref_base_url
        self.email = settings.crossref_email or "research@example.com"
        self.max_results = settings.crossref_max_results
        self._is_test = settings.is_test

    @property
    def source_name(self) -> str:
        return "crossref"

    async def search(self, query: str, max_results: int = 50) -> PaperList:
        """Search CrossRef for papers matching the query."""
        if self._is_test:
            return self._mock_search(query, max_results)

        max_results = min(max_results, self.max_results)
        params = {
            "query": query,
            "rows": min(max_results, 100),
            "select": "DOI,title,abstract,author,published,container-title,subject,URL,link,is-referenced-by-count,ISSN,type,publisher",
        }
        headers = {
            "User-Agent": f"ScholarRace/1.0 (mailto:{self.email})",
        }

        try:
            response = await self._http_client.get(
                self.base_url, params=params, headers=headers
            )
            if response.status_code != 200:
                logger.error(
                    f"CrossRef API returned {response.status_code}: {response.text[:200]}"
                )
                return PaperList(source=self.source_name)

            import json

            data = json.loads(response.text)
            message = data.get("message", {})
            items = message.get("items", [])
            papers = []
            for item in items:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)

            return PaperList(papers=papers, source=self.source_name)
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
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
        """Parse a CrossRef API result item into a Paper."""
        try:
            # Title (list in CrossRef, take first)
            titles = item.get("title") or []
            title = titles[0] if titles else ""

            # Abstract (CrossRef has HTML in abstract, strip tags)
            abstract = item.get("abstract", "") or ""
            if abstract:
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()

            # Date
            year = None
            pub_date = None
            date_parts = item.get("published", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                parts = date_parts[0]
                if len(parts) >= 3:
                    y, m, d = parts[0], parts[1], parts[2]
                    year = y
                    try:
                        pub_date = date(y, m, d)
                    except ValueError:
                        pass
                elif len(parts) >= 1:
                    year = parts[0]

            # Authors
            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                full = f"{given} {family}".strip()
                if full:
                    authors.append(full)

            # DOI
            doi = item.get("DOI", "")
            url = item.get("URL", "") or (f"https://doi.org/{doi}" if doi else "")

            # PDF URL from link
            pdf_url = None
            links = item.get("link", []) or []
            for link in links:
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break
            if not pdf_url and links:
                pdf_url = links[0].get("URL")

            # Citation count
            citation_count = item.get("is-referenced-by-count", 0) or 0

            # Venue
            venue = ""
            containers = item.get("container-title", [])
            if containers:
                venue = containers[0]

            # Fields of study (subjects in CrossRef)
            fields = item.get("subject", []) or []

            # arXiv ID extraction from DOI
            arxiv_id = None
            if doi and "arxiv" in doi.lower():
                m = re.search(r"arxiv\.(\d+\.\d+|[a-z\-]+/\d+\.\d+)", doi, re.IGNORECASE)
                if m:
                    arxiv_id = m.group(1)

            identity = PaperIdentity(
                doi=doi or None,
                arxiv_id=arxiv_id,
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
                doi=doi or None,
                arxiv_id=arxiv_id,
                url=url,
                pdf_url=pdf_url,
                citation_count=citation_count,
                venue=venue,
                fields_of_study=fields,
                source=self.source_name,
            )
        except Exception as e:
            logger.error(f"Failed to parse CrossRef item: {e}")
            return None

    async def get_paper(self, doi: str) -> Optional[Paper]:
        """Get a single paper by DOI."""
        if self._is_test:
            for paper in _MOCK_PAPERS:
                if paper.doi == doi:
                    return paper
            return None

        url = f"{self.base_url.rstrip('/')}/{doi}"
        headers = {"User-Agent": f"ScholarRace/1.0 (mailto:{self.email})"}
        try:
            response = await self._http_client.get(url, headers=headers)
            if response.status_code != 200:
                return None
            import json

            data = json.loads(response.text)
            item = data.get("message", {})
            return self._parse_item(item)
        except Exception as e:
            logger.error(f"CrossRef get_paper failed: {e}")
            return None
