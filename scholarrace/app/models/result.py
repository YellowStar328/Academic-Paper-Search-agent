"""Pipeline result models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.candidate import CandidateQuery
from app.models.paper import Paper


class PaperWithScores(BaseModel):
    """A paper enriched with ranking scores."""

    paper: Paper
    relevance_score: float = 0.0
    authority_score: float = 0.0
    recency_score: float = 0.0
    citation_score: float = 0.0
    diversity_score: float = 0.0
    redundancy_score: float = 0.0
    final_score: float = 0.0
    embedding_similarity: Optional[float] = None
    judge_reasoning: str = ""


class GraphNode(BaseModel):
    """A node in the research graph."""

    paper_id: str
    title: str
    year: Optional[int] = None
    cluster_id: Optional[int] = None


class GraphEdge(BaseModel):
    """An edge in the research graph (citation or reference)."""

    source_id: str
    target_id: str
    edge_type: str = "citation"
    weight: float = 1.0


class GraphCluster(BaseModel):
    """A cluster of related papers."""

    cluster_id: int
    label: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    centroid_title: str = ""


class TimelineEntry(BaseModel):
    """A timeline entry for temporal analysis."""

    year: int
    paper_ids: list[str] = Field(default_factory=list)
    count: int = 0


class ResearchGraph(BaseModel):
    """The structured research output graph."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    clusters: list[GraphCluster] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)


class PipelineMetrics(BaseModel):
    """Observability metrics for a pipeline run."""

    request_id: str
    query: str
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    token_usage: int = 0
    papers_collected: int = 0
    papers_after_dedup: int = 0
    papers_after_rerank: int = 0
    papers_final: int = 0
    models_used: list[str] = Field(default_factory=list)
    search_sources_used: list[str] = Field(default_factory=list)
    thompson_allocations: dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SearchResult(BaseModel):
    """Final output of a search pipeline run."""

    query: str
    semantic_core: str = ""
    papers: list[PaperWithScores] = Field(default_factory=list)
    graph: ResearchGraph = Field(default_factory=ResearchGraph)
    metrics: Optional[PipelineMetrics] = None
    raw_candidates: list[CandidateQuery] = Field(default_factory=list)
    judged_candidates: list[Any] = Field(default_factory=list)
