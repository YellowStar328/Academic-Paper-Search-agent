"""Paper data models (Pydantic for API + SQLAlchemy ORM for persistence)."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


def normalize_title(title: str) -> str:
    """Normalize a paper title for deduplication.

    Lowercases, strips punctuation/whitespace, removes accents.
    """
    if not title:
        return ""
    # Normalize unicode (e.g., é -> e)
    title = unicodedata.normalize("NFKD", title)
    # Remove non-ASCII (accents leftovers)
    title = title.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    title = title.lower()
    # Remove all non-alphanumeric characters
    title = re.sub(r"[^a-z0-9]", "", title)
    return title


class PaperIdentity(BaseModel):
    """Canonical identity for deduplication. Priority: DOI > arXiv ID > S2 ID > title+year."""

    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    normalized_title: str = ""
    year: Optional[int] = None

    def identity_key(self) -> str:
        """Return the highest-priority identity key for dedup."""
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.semantic_scholar_id:
            return f"s2:{self.semantic_scholar_id}"
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        if self.pubmed_id:
            return f"pubmed:{self.pubmed_id}"
        # Fallback: normalized title + year
        if self.normalized_title and self.year:
            return f"title_year:{self.normalized_title}:{self.year}"
        if self.normalized_title:
            return f"title:{self.normalized_title}"
        return ""

    @classmethod
    def from_paper_data(
        cls,
        title: str,
        year: Optional[int] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        semantic_scholar_id: Optional[str] = None,
        openalex_id: Optional[str] = None,
        pubmed_id: Optional[str] = None,
    ) -> PaperIdentity:
        return cls(
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            openalex_id=openalex_id,
            pubmed_id=pubmed_id,
            normalized_title=normalize_title(title),
            year=year,
        )


class Paper(BaseModel):
    """A single academic paper."""

    paper_id: str = Field(..., description="Internal UUID")
    identity: PaperIdentity = Field(default_factory=PaperIdentity)
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    publication_date: Optional[date] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    url: str = ""
    pdf_url: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    fields_of_study: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source: str = "unknown"
    references: list[str] = Field(
        default_factory=list, description="List of paper_ids referenced by this paper"
    )
    citations: list[str] = Field(
        default_factory=list, description="List of paper_ids citing this paper"
    )

    def identity_key(self) -> str:
        return self.identity.identity_key()


class PaperList(BaseModel):
    """A collection of papers with convenience methods."""

    papers: list[Paper] = Field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.papers)

    def __iter__(self):
        return iter(self.papers)

    def extend(self, other: PaperList) -> PaperList:
        self.papers.extend(other.papers)
        return self
