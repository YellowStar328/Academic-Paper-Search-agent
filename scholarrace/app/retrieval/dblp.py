"""DBLP search provider — real API integration.

DBLP provides free, no-key access to computer science bibliographic data.
API docs: https://dblp.org/faq/How+to+use+the+dblp+search+API.html

In test mode (APP_ENV=test), falls back to mock data for unit tests.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional
from uuid import uuid4
from xml.etree import ElementTree as ET

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
            normalized_title="attentionisallyouneed",
            year=2017,
        ),
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        venue="NeurIPS",
        url="https://dblp.org/rec/conf/nips/vaswani-2017",
        citation_count=0,
        fields_of_study=["Computer Science"],
        source="dblp",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            normalized_title="bertpretrainingofdeepbidirectionaltransformers",
            year=2019,
        ),
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        abstract="We introduce a new language representation model called BERT.",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        year=2019,
        venue="NAACL",
        url="https://dblp.org/rec/conf/naacl/devlin-2019",
        citation_count=0,
        fields_of_study=["Computer Science"],
        source="dblp",
    ),
]


class DblpProvider(BaseSearchProvider):
    """DBLP API search provider — computer science bibliography."""

    def __init__(self, http_client: Optional[HttpClient] = None):
        settings = get_settings()
        super().__init__(
            http_client=http_client or HttpClient(timeout=settings.dblp_timeout),
            timeout=settings.dblp_timeout,
        )
        self.base_url = settings.dblp_base_url
        self.max_results = settings.dblp_max_results

    @property
    def source_name(self) -> str:
        return "dblp"

    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search DBLP for papers matching the query.

        Uses the DBLP publication search API:
        https://dblp.org/search/publ/api?q=...&format=json&h=...

        DBLP API has no server-side year filter, so when year_start/year_end
        are provided, results are filtered at the client level.
        """
        max_results = min(max_results, self.max_results)

        # Test mode: return mock data
        if get_settings().is_test:
            papers = _MOCK_PAPERS[:max_results]
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            return PaperList(
                papers=papers,
                source=self.source_name,
            )

        params = {
            "q": query,
            "format": "json",
            "h": max_results,
        }

        try:
            response = await self._http_client.get(
                self.base_url, params=params
            )
            data = response.json()
            papers = self._parse_results(data)
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            logger.info(
                f"DBLP: returned {len(papers)} results for '{query}'"
            )
            return PaperList(papers=papers, source=self.source_name)
        except Exception as e:
            logger.error(f"DBLP search failed: {e}")
            return PaperList(papers=[], source=self.source_name)

    def _parse_results(self, response: dict) -> list[Paper]:
        """Parse DBLP JSON response into Paper objects."""
        hits = (
            response.get("result", {}).get("hits", {}).get("hit", [])
        )
        papers = []
        for hit in hits:
            info = hit.get("info", {})
            if not info:
                continue
            title = info.get("title", "").strip()
            if not title:
                continue

            # Authors: DBLP returns a string or list
            authors_raw = info.get("authors", {}).get("author", [])
            if isinstance(authors_raw, dict):
                # Single author
                authors_raw = [authors_raw]
            authors = [
                a.get("text", "") if isinstance(a, dict) else str(a)
                for a in authors_raw
            ] if authors_raw else []

            year_str = info.get("year", "")
            year = int(year_str) if year_str and year_str.isdigit() else None

            venue = info.get("venue", "") or ""
            doi = info.get("doi", "") or None
            url = info.get("url", "") or None

            paper = Paper(
                paper_id=str(uuid4()),
                identity=PaperIdentity(
                    doi=doi,
                    normalized_title=title.lower().replace(" ", ""),
                    year=year,
                ),
                title=title,
                abstract="",  # DBLP doesn't provide abstracts
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                citation_count=0,  # DBLP doesn't provide citation counts
                fields_of_study=["Computer Science"],
                source=self.source_name,
            )
            papers.append(paper)

        return papers
