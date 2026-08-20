"""OpenAlex search provider — Mock implementation."""

from __future__ import annotations

from uuid import uuid4
from typing import Optional

from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider

_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            openalex_id="mock-openalex-001",
            normalized_title="imageneretclassificationwithdeeepconvolutionalneuralnetworks",
            year=2012,
        ),
        title="ImageNet Classification with Deep Convolutional Neural Networks",
        abstract="We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest.",
        authors=["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
        year=2012,
        venue="NeurIPS",
        openalex_id="mock-openalex-001",
        url="https://openalex.org/mock-openalex-001",
        citation_count=40000,
        fields_of_study=["Computer Science"],
        source="openalex",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            openalex_id="mock-openalex-002",
            normalized_title="residuallearningforimagerecognition",
            year=2016,
        ),
        title="Deep Residual Learning for Image Recognition",
        abstract="We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
        authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
        year=2016,
        venue="CVPR",
        openalex_id="mock-openalex-002",
        url="https://openalex.org/mock-openalex-002",
        citation_count=50000,
        fields_of_study=["Computer Science"],
        source="openalex",
    ),
]


class OpenAlexProvider(BaseSearchProvider):
    """Mock OpenAlex search provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "openalex"

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
            if paper.openalex_id == paper_id:
                return paper
        return None
