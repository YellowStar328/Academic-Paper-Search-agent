"""Semantic Scholar search provider — Mock implementation.

The Semantic Scholar API requires no key for basic usage but has rate limits.
This is a Mock implementation that returns preset papers for testing.
The interface is ready for real integration.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

from app.models.paper import Paper, PaperIdentity, PaperList
from app.retrieval.base import BaseSearchProvider

logger = logging.getLogger(__name__)

# Preset mock papers for testing
_MOCK_PAPERS = [
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            semantic_scholar_id="mock-s2-001",
            normalized_title="attentionisallyouneed",
            year=2017,
        ),
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=2017,
        venue="NeurIPS",
        semantic_scholar_id="mock-s2-001",
        url="https://www.semanticscholar.org/paper/mock-s2-001",
        citation_count=100000,
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
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
        year=2019,
        venue="NAACL",
        semantic_scholar_id="mock-s2-002",
        url="https://www.semanticscholar.org/paper/mock-s2-002",
        citation_count=50000,
        fields_of_study=["Computer Science"],
        source="semantic_scholar",
    ),
    Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            semantic_scholar_id="mock-s2-003",
            normalized_title="deeplearning",
            year=2015,
        ),
        title="Deep Learning",
        abstract="Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction.",
        authors=["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"],
        year=2015,
        venue="Nature",
        semantic_scholar_id="mock-s2-003",
        url="https://www.semanticscholar.org/paper/mock-s2-003",
        citation_count=30000,
        fields_of_study=["Computer Science"],
        source="semantic_scholar",
    ),
]


class SemanticScholarProvider(BaseSearchProvider):
    """Mock Semantic Scholar search provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_papers = list(_MOCK_PAPERS)

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    async def search(self, query: str, max_results: int = 50) -> PaperList:
        """Search mock papers. Returns papers that match the query keywords."""
        query_lower = query.lower()
        keywords = query_lower.split()

        matched = []
        for paper in self._mock_papers:
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            if any(k in title_lower or k in abstract_lower for k in keywords):
                matched.append(paper)

        # If no matches, return all (for testing pipeline flow)
        if not matched:
            matched = list(self._mock_papers)

        return PaperList(papers=matched[:max_results], source=self.source_name)

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        for paper in self._mock_papers:
            if paper.semantic_scholar_id == paper_id:
                return paper
        return None

    async def get_citations(self, paper_id: str, max_results: int = 50) -> PaperList:
        """Mock: return subset of papers as citations."""
        return PaperList(
            papers=self._mock_papers[:2], source=self.source_name
        )

    async def get_references(self, paper_id: str, max_results: int = 50) -> PaperList:
        """Mock: return subset of papers as references."""
        return PaperList(
            papers=self._mock_papers[:1], source=self.source_name
        )
