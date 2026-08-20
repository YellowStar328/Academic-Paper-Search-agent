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
    metrics: Optional[dict] = None
    latency_ms: float


@router.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Run the full search pipeline and return ranked papers."""
    from app.main import get_pipeline, create_pipeline, set_pipeline

    pipeline = get_pipeline()
    if pipeline is None:
        # Fallback: lazily initialize on first request if startup missed
        try:
            pipeline = create_pipeline()
            set_pipeline(pipeline)
        except Exception as e:
            logger.error(f"Failed to init pipeline on demand: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search pipeline not initialized: {e}",
            )

    # Build effective query text: append year-range hint so the rule-based
    # parser can extract hard_filters (year_start/year_end).
    query_text = request.query
    if request.year_from is not None and request.year_to is not None:
        query_text += f" {request.year_from}-{request.year_to}"
    elif request.year_from is not None:
        query_text += f" after {request.year_from}"
    elif request.year_to is not None:
        query_text += f" before {request.year_to}"

    user_query = UserQuery(
        query=query_text,
        options=SearchOptions(
            top_k=request.max_results,
            enable_citation_expansion=request.enable_citation_expansion,
            enable_embedding_rerank=request.enable_embedding_rerank,
            strategy="full_pipeline",
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
            "url": pws.paper.url,
            "pdf_url": pws.paper.pdf_url,
            "fields_of_study": pws.paper.fields_of_study,
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

    metrics = None
    if result.metrics:
        metrics = {
            "request_id": result.metrics.request_id,
            "query": result.metrics.query,
            "total_latency_ms": result.metrics.total_latency_ms,
            "stage_latencies": result.metrics.stage_latencies,
            "llm_calls": result.metrics.llm_calls,
            "token_usage": result.metrics.token_usage,
            "papers_collected": result.metrics.papers_collected,
            "papers_after_dedup": result.metrics.papers_after_dedup,
            "papers_after_rerank": result.metrics.papers_after_rerank,
            "papers_final": result.metrics.papers_final,
            "models_used": result.metrics.models_used,
            "search_sources_used": result.metrics.search_sources_used,
            "thompson_allocations": result.metrics.thompson_allocations,
        }

    return SearchResponse(
        request_id=result.request_id,
        query=result.query,
        total_papers=len(result.papers),
        papers=papers,
        summary=summary,
        metrics=metrics,
        latency_ms=result.latency_ms,
    )
