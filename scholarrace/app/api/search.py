"""POST /api/search — main search endpoint.

Accepts a user query and returns ranked papers with a research graph.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.query import SearchOptions, UserQuery
from app.models.result import SearchResult

router = APIRouter()
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    max_results: int = Field(10, ge=1, le=100, description="Max papers to return")
    year_from: Optional[int] = Field(None, description="Earliest publication year")
    year_to: Optional[int] = Field(None, description="Latest publication year")
    enable_citation_expansion: bool = Field(
        True, description="Enable citation/reference expansion"
    )
    enable_embedding_rerank: bool = Field(
        True, description="Enable embedding-based coarse reranking"
    )


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    request_id: str
    query: str
    total_papers: int
    papers: list[dict] = Field(default_factory=list)
    summary: Optional[dict] = None
    latency_ms: float


@router.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Run the full search pipeline and return ranked papers."""
    from app.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not initialized",
        )

    user_query = UserQuery(
        query=request.query,
        options=SearchOptions(
            max_results=request.max_results,
            year_from=request.year_from,
            year_to=request.year_to,
            enable_citation_expansion=request.enable_citation_expansion,
            enable_embedding_rerank=request.enable_embedding_rerank,
        ),
    )

    try:
        result: SearchResult = await pipeline.run(user_query)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )

    # Convert to response format
    papers = [
        {
            "paper_id": pws.paper.paper_id,
            "title": pws.paper.title,
            "abstract": pws.paper.abstract,
            "year": pws.paper.year,
            "authors": pws.paper.authors,
            "citation_count": pws.paper.citation_count,
            "venue": pws.paper.venue,
            "source": pws.paper.source,
            "doi": pws.paper.identity.doi,
            "arxiv_id": pws.paper.identity.arxiv_id,
            "relevance_score": pws.relevance_score,
            "authority_score": pws.authority_score,
            "recency_score": pws.recency_score,
            "citation_score": pws.citation_score,
            "diversity_score": pws.diversity_score,
            "redundancy_score": pws.redundancy_score,
            "final_score": pws.final_score,
            "judge_reasoning": pws.judge_reasoning,
        }
        for pws in result.papers
    ]

    summary = None
    if result.summary:
        summary = {
            "total_papers": result.summary.total_papers,
            "query": result.summary.query,
            "domain": result.summary.domain,
            "intent": result.summary.intent,
            "top_paper_title": result.summary.top_paper_title,
            "clusters_count": result.summary.clusters_count,
        }

    return SearchResponse(
        request_id=result.request_id,
        query=result.query,
        total_papers=len(result.papers),
        papers=papers,
        summary=summary,
        latency_ms=result.latency_ms,
    )
