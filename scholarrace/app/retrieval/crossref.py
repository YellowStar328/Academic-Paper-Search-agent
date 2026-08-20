"""Crossref search provider — Mock implementation."""

from __future__ import annotations

from uuid import uuid4
from typing import Optional

from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider

_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            doi="10.1000/mock-crossref-001",
            normalized_title="areviewofgraphneuralnetworks",
            year=2022,
        ),
        title="A Review of Graph Neural Networks",
        abstract="Graph neural networks (GNNs) have emerged as a powerful tool for learning on graph-structured data.",
        authors=["Michael Brown", "Emily Davis"],
        year=2022,
        venue="ACM Computing Surveys",
        doi="10.1000/mock-crossref-001",
        url="https://doi.org/10.1000/mock-crossref-001",
        citation_count=800,
        fields_of_study=["Computer Science"],
        source="crossref",
    ),
]


class CrossrefProvider(BaseSearchProvider):
    """Mock Crossref search provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "crossref"

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
            if paper.doi == paper_id:
                return paper
        return None
