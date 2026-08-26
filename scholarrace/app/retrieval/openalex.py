"""OpenAlex search provider — real API integration.

OpenAlex is a fully-open catalog of scholarly works (no API key required).
API docs: https://docs.openalex.org/

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
from app.retrieval.base import BaseSearchProvider, filter_papers_by_year
from app.utils.http_client import HttpClient

logger = logging.getLogger(__name__)

# Mock data for test mode
_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            openalex_id="mock-openalex-001",
            normalized_title="imagenetclassificationwithdeepconvolutionalneuralnetworks",
            year=2012,
        ),
        title="ImageNet Classification with Deep Convolutional Neural Networks",
        abstract="We trained a large, deep convolutional neural network to classify images.",
        authors=["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
        year=2012,
        venue="NeurIPS",
        openalex_id="mock-openalex-001",
        url="https://openalex.org/mock-openalex-001",
        citation_count=80000,
        fields_of_study=["Computer Science"],
        source="openalex",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            openalex_id="mock-openalex-002",
            normalized_title="residualnetworksforimageclassification",
            year=2016,
        ),
        title="Residual Networks for Image Classification",
        abstract="We present a residual learning framework for training deep networks.",
        authors=["Kaiming He", "Xiangyu Zhang"],
        year=2016,
        venue="CVPR",
        openalex_id="mock-openalex-002",
        url="https://openalex.org/mock-openalex-002",
        citation_count=60000,
        fields_of_study=["Computer Science"],
        source="openalex",
    ),
]


class OpenAlexProvider(BaseSearchProvider):
    """OpenAlex API search provider.

    Uses the Works API: https://api.openalex.org/works
    Free, no authentication required (email in User-Agent for polite pool).
    """

    def __init__(self, http_client: Optional[HttpClient] = None):
        settings = get_settings()
        super().__init__(
            http_client=http_client or HttpClient(timeout=settings.openalex_timeout),
            timeout=settings.openalex_timeout,
        )
        self.base_url = settings.openalex_base_url
        self.email = settings.openalex_email or "research@example.com"
        self.max_results = settings.openalex_max_results
        self._is_test = settings.is_test

    @property
    def source_name(self) -> str:
        return "openalex"

    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search OpenAlex for papers matching the query.

        If year_start/year_end are provided, uses API-level date filtering
        via ``from_publication_date`` / ``to_publication_date``.
        """
        if self._is_test:
            papers = self._mock_search(query, max_results).papers
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            return PaperList(papers=papers, source=self.source_name)

        max_results = min(max_results, self.max_results)
        params = {
            "search": query,
            "per-page": min(max_results, 200),
            "mailto": self.email,
        }
        # OpenAlex requires date-range filters inside the `filter` param,
        # e.g. filter=from_publication_date:2020-01-01,to_publication_date:2025-12-31
        filters = []
        if year_start is not None:
            filters.append(f"from_publication_date:{year_start}-01-01")
        if year_end is not None:
            filters.append(f"to_publication_date:{year_end}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        try:
            response = await self._http_client.get(
                self.base_url, params=params
            )
            if response.status_code != 200:
                logger.error(
                    f"OpenAlex API returned {response.status_code}: {response.text[:200]}"
                )
                return PaperList(source=self.source_name)

            import json

            data = json.loads(response.text)
            papers = []
            for item in data.get("results", []):
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)

            # Defensive: also filter at result level
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            return PaperList(papers=papers, source=self.source_name)
        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
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
        """Parse an OpenAlex API result item into a Paper."""
        try:
            title = item.get("display_name", "") or item.get("title", "") or ""
            abstract_inverted = item.get("abstract_inverted_index")
            abstract = self._reconstruct_abstract(abstract_inverted)

            year = item.get("publication_year")

            # Publication date
            pub_date = None
            pub_date_str = item.get("publication_date")
            if pub_date_str:
                try:
                    parts = pub_date_str.split("-")
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    pub_date = date(y, m, d)
                except (ValueError, IndexError):
                    pass

            # Authors
            authors = []
            authorships = item.get("authorships", [])
            for a in authorships:
                author = a.get("author") or {}
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            # IDs
            doi_url = item.get("doi") or ""
            doi = doi_url.replace("https://doi.org/", "") if doi_url else None

            # OpenAlex ID -> raw id
            openalex_id_full = item.get("id", "")
            openalex_id = (
                openalex_id_full.replace("https://openalex.org/", "")
                if openalex_id_full
                else None
            )

            # URL (landing page)
            url = item.get("id") or openalex_id_full or ""

            # PDF URL from best_oa_location
            pdf_url = None
            best_oa = item.get("best_oa_location")
            if best_oa:
                pdf_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

            # Citation count
            citation_count = item.get("cited_by_count", 0) or 0

            # Venue
            venue = ""
            primary_location = item.get("primary_location")
            if primary_location:
                source_info = primary_location.get("source") or {}
                venue = source_info.get("display_name", "") or ""

            # Fields of study
            fields = []
            concepts = item.get("concepts", [])
            for c in concepts[:5]:
                name = c.get("display_name", "")
                if name:
                    fields.append(name)
            # Also try topics (newer OpenAlex API)
            topics = item.get("topics", [])
            for t in topics[:3]:
                name = t.get("display_name", "")
                if name and name not in fields:
                    fields.append(name)

            # arXiv ID — extract from any arXiv URL across locations /
            # best_oa_location / pdf_url.  Never use the OpenAlex ID as a
            # fallback (different identifier space).
            arxiv_id = None
            arxiv_url_pattern = re.compile(
                r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{4}\.\d{1,5})"
            )

            def _extract_arxiv_id(url_or_path: str) -> Optional[str]:
                if not url_or_path:
                    return None
                m = arxiv_url_pattern.search(url_or_path)
                return m.group(1) if m else None

            # 1) best_oa_location
            if best_oa:
                for key in ("landing_page_url", "pdf_url"):
                    aid = _extract_arxiv_id(best_oa.get(key, ""))
                    if aid:
                        arxiv_id = aid
                        break

            # 2) locations list
            if not arxiv_id:
                for loc in item.get("locations", []):
                    for key in ("landing_page_url", "pdf_url"):
                        aid = _extract_arxiv_id(loc.get(key, ""))
                        if aid:
                            arxiv_id = aid
                            break
                    if arxiv_id:
                        break

            # 3) DOI itself may be an arXiv DOI (10.48550/arXiv.2401.12345)
            if not arxiv_id and doi:
                m = re.search(
                    r"arxiv\.(\d{4}\.\d{4,5}|[a-z\-]+/\d{4}\.\d{1,5})",
                    doi,
                    re.IGNORECASE,
                )
                if m:
                    arxiv_id = m.group(1)

            identity = PaperIdentity(
                doi=doi,
                arxiv_id=arxiv_id,
                openalex_id=openalex_id,
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
                openalex_id=openalex_id,
                url=url,
                pdf_url=pdf_url,
                citation_count=citation_count,
                venue=venue,
                fields_of_study=fields,
                source=self.source_name,
            )
        except Exception as e:
            logger.error(f"Failed to parse OpenAlex item: {e}")
            return None

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        positions = []
        for word, indices in inverted_index.items():
            for idx in indices:
                positions.append((idx, word))
        positions.sort()
        return " ".join(w for _, w in positions)

    async def get_paper(self, openalex_id: str) -> Optional[Paper]:
        """Get a single paper by OpenAlex ID."""
        if self._is_test:
            for paper in _MOCK_PAPERS:
                if paper.openalex_id == openalex_id:
                    return paper
            return None

        url = f"{self.base_url.rstrip('/')}/{openalex_id}"
        params = {"mailto": self.email}
        try:
            response = await self._http_client.get(url, params=params)
            if response.status_code != 200:
                return None
            import json

            item = json.loads(response.text)
            return self._parse_item(item)
        except Exception as e:
            logger.error(f"OpenAlex get_paper failed: {e}")
            return None
