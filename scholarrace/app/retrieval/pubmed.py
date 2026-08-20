"""PubMed search provider — Mock implementation."""

from __future__ import annotations

from uuid import uuid4
from typing import Optional

from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider

_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            pubmed_id="mock-pubmed-001",
            normalized_title="asurveyondeeplearningappliedtomedicalimaging",
            year=2021,
        ),
        title="A survey on deep learning applied to medical imaging",
        abstract="Deep learning has revolutionized medical imaging analysis in recent years.",
        authors=["John Smith", "Jane Doe"],
        year=2021,
        venue="Medical Image Analysis",
        pubmed_id="mock-pubmed-001",
        url="https://pubmed.ncbi.nlm.nih.gov/mock-pubmed-001",
        citation_count=500,
        fields_of_study=["Medicine"],
        source="pubmed",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            pubmed_id="mock-pubmed-002",
            normalized_title="transformerbasedmodelsforclinicalnlp",
            year=2023,
        ),
        title="Transformer-based models for clinical NLP",
        abstract="We explore transformer architectures for clinical text analysis.",
        authors=["Alice Chen", "Bob Wilson"],
        year=2023,
        venue="JAMIA",
        pubmed_id="mock-pubmed-002",
        url="https://pubmed.ncbi.nlm.nih.gov/mock-pubmed-002",
        citation_count=120,
        fields_of_study=["Medicine"],
        source="pubmed",
    ),
]


class PubMedProvider(BaseSearchProvider):
    """Mock PubMed search provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "pubmed"

    async def search(self, query: str, max_results: int = 50) -> PaperList:
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

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        for paper in _MOCK_PAPERS:
            if paper.pubmed_id == paper_id:
                return paper
        return None
