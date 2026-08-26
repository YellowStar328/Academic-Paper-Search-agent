"""arXiv search provider — real httpx integration with arXiv API.

arXiv API is free and requires no API key.
API docs: https://info.arxiv.org/help/api/index.html

NOTE: arXiv API expects field-specific search syntax (e.g. ``all:"keyword" AND ti:"keyword"``),
not natural-language sentences. The pipeline is expected to use a strong LLM to refine
queries before calling this provider. If a raw sentence is passed, we fall back to a
simple keyword extraction to avoid garbage results.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional
from uuid import uuid4
from xml.etree import ElementTree as ET

from app.config import get_settings
from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider, filter_papers_by_year
from app.utils.http_client import HttpClient

logger = logging.getLogger(__name__)

# arXiv Atom XML namespaces
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# Common English stop words — only used as a fallback when no LLM refinement is available
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "on", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "out", "off",
        "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "just", "don", "now", "i", "me", "my", "myself", "we",
        "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "if", "or", "and", "as", "until",
        "while", "because", "though", "although", "even", "give",
        "gives", "gave", "given", "giving", "show", "shows", "showed",
        "shown", "showing", "paper", "papers", "find", "finds", "found",
        "finding", "use", "uses", "using", "result", "results",
        "better", "best", "good", "well", "also", "but",
    }
)


def _fallback_keyword_query(query: str) -> str:
    """Fallback: simple keyword extraction when no LLM refinement is available.

    This is a last-resort heuristic — the pipeline should normally use
    QueryRefiner (strong LLM) to generate proper arXiv search syntax.
    """
    cleaned = re.sub(r"[^\w\s\-]", " ", query)
    tokens = cleaned.split()
    keywords = []
    for t in tokens:
        t_lower = t.lower().strip("-")
        if len(t_lower) >= 3 and t_lower not in _STOP_WORDS and t_lower.isalpha():
            keywords.append(t_lower)
        elif len(t_lower) >= 4 and any(c.isdigit() for c in t):
            keywords.append(t_lower)

    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)

    if not unique:
        return f"all:{query}"

    top = unique[:6]
    return " AND ".join(f'all:"{k}"' for k in top)


def _looks_like_arxiv_syntax(query: str) -> bool:
    """Check if the query is already in arXiv field-specific syntax.

    Examples of arXiv syntax: all:"keyword" AND ti:"keyword", abs:transformer, etc.
    """
    # Contains field operators like all:, ti:, abs:, au:, cat:
    return bool(re.search(r'\b(all|ti|abs|au|cat|sr):\s*"?', query))


def _build_arxiv_search_query(query: str) -> str:
    """Build an arXiv API search_query string.

    If the query is already in arXiv field-specific syntax (from LLM refinement),
    use it directly. Otherwise, fall back to simple keyword extraction.
    """
    if _looks_like_arxiv_syntax(query):
        return query
    return _fallback_keyword_query(query)


class ArxivProvider(BaseSearchProvider):
    """arXiv API search provider."""

    def __init__(self, http_client: Optional[HttpClient] = None):
        settings = get_settings()
        super().__init__(
            http_client=http_client or HttpClient(timeout=settings.arxiv_timeout),
            timeout=settings.arxiv_timeout,
        )
        self.base_url = settings.arxiv_base_url
        self.max_results = settings.arxiv_max_results

    @property
    def source_name(self) -> str:
        return "arxiv"

    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search arXiv for papers matching the query.

        Uses the arXiv API: http://export.arxiv.org/api/query
        Converts natural-language queries into arXiv field-specific syntax.
        If year_start/year_end are provided, results are filtered to that range.
        """
        max_results = min(max_results, self.max_results)
        search_query = _build_arxiv_search_query(query)
        logger.debug(f"arXiv search_query: {search_query[:200]}")
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = await self._http_client.get(self.base_url, params=params)
            if response.status_code != 200:
                logger.error(
                    f"arXiv API returned {response.status_code}: {response.text[:200]}"
                )
                return PaperList(source=self.source_name)

            papers = self._parse_atom_feed(response.text)
            if year_start is not None or year_end is not None:
                papers = filter_papers_by_year(papers, year_start, year_end)
            return PaperList(papers=papers, source=self.source_name)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return PaperList(source=self.source_name)

    def _parse_atom_feed(self, xml_text: str) -> list[Paper]:
        """Parse arXiv Atom XML feed into Paper objects."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return papers

        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Paper]:
        """Parse a single arXiv entry element into a Paper."""
        try:
            # Title
            title_el = entry.find(f"{{{ATOM_NS}}}title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""

            # Abstract (summary)
            summary_el = entry.find(f"{{{ATOM_NS}}}summary")
            abstract = (
                summary_el.text.strip()
                if summary_el is not None and summary_el.text
                else ""
            )

            # Published date
            published_el = entry.find(f"{{{ATOM_NS}}}published")
            year = None
            pub_date = None
            if published_el is not None and published_el.text:
                date_str = published_el.text.strip()
                # Parse "2024-01-15T..." format
                try:
                    parts = date_str.split("T")[0].split("-")
                    year = int(parts[0])
                    pub_date = date(year, int(parts[1]), int(parts[2]))
                except (ValueError, IndexError):
                    pass

            # Authors
            authors = []
            for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
                name_el = author_el.find(f"{{{ATOM_NS}}}name")
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # arXiv ID (from id element, e.g. http://arxiv.org/abs/2401.12345v1)
            id_el = entry.find(f"{{{ATOM_NS}}}id")
            arxiv_id = ""
            if id_el is not None and id_el.text:
                arxiv_id = self._extract_arxiv_id(id_el.text.strip())

            # DOI
            doi_el = entry.find(f"{{{ARXIV_NS}}}doi")
            doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

            # PDF link
            pdf_url = None
            for link in entry.findall(f"{{{ATOM_NS}}}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            # Comments (may contain keywords)
            comment_el = entry.find(f"{{{ARXIV_NS}}}comment")
            comments = (
                comment_el.text.strip()
                if comment_el is not None and comment_el.text
                else ""
            )

            # Primary category
            primary_cat_el = entry.find(f"{{{ARXIV_NS}}}primary_category")
            fields_of_study = []
            if primary_cat_el is not None:
                term = primary_cat_el.get("term", "")
                if term:
                    fields_of_study.append(term)

            # All categories
            for cat_el in entry.findall(f"{{{ATOM_NS}}}category"):
                term = cat_el.get("term", "")
                if term and term not in fields_of_study:
                    fields_of_study.append(term)

            identity = PaperIdentity(
                doi=doi,
                arxiv_id=arxiv_id or None,
                normalized_title=title.lower().replace(" ", ""),
                year=year,
            )

            paper = Paper(
                paper_id=str(uuid4()),
                identity=identity,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                publication_date=pub_date,
                doi=doi,
                arxiv_id=arxiv_id or None,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                pdf_url=pdf_url,
                fields_of_study=fields_of_study,
                keywords=[],
                source=self.source_name,
            )
            return paper
        except Exception as e:
            logger.error(f"Failed to parse arXiv entry: {e}")
            return None

    def _extract_arxiv_id(self, id_url: str) -> str:
        """Extract arXiv ID from URL like http://arxiv.org/abs/2401.12345v1."""
        # Match patterns like 2401.12345 or cs.AI/0701001 (old format)
        match = re.search(r"abs/(.+?)(?:v\d+)?$", id_url)
        if match:
            return match.group(1)
        return id_url

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by arXiv ID."""
        params = {"id_list": paper_id}

        try:
            response = await self._http_client.get(self.base_url, params=params)
            if response.status_code != 200:
                return None
            papers = self._parse_atom_feed(response.text)
            return papers[0] if papers else None
        except Exception as e:
            logger.error(f"arXiv get_paper failed: {e}")
            return None
