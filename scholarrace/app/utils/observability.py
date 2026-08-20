"""Observability utilities: metrics tracking for pipeline runs."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from uuid import uuid4

from app.models.result import PipelineMetrics

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks metrics during a pipeline run."""

    def __init__(self, query: str = ""):
        self.request_id: str = str(uuid4())
        self.query: str = query
        self.start_time: float = time.time()
        self.stage_start: float = 0.0
        self.stage_latencies: dict[str, float] = {}
        self.llm_calls: int = 0
        self.token_usage: int = 0
        self.papers_collected: int = 0
        self.papers_after_dedup: int = 0
        self.papers_after_rerank: int = 0
        self.papers_final: int = 0
        self.models_used: list[str] = []
        self.search_sources_used: list[str] = []
        self.thompson_allocations: dict[str, int] = {}

    def start_stage(self, stage_name: str) -> None:
        """Start timing a stage."""
        self.stage_start = time.time()
        logger.debug(f"Pipeline stage started: {stage_name}")

    def end_stage(self, stage_name: str) -> None:
        """End timing a stage and record latency."""
        latency = (time.time() - self.stage_start) * 1000
        self.stage_latencies[stage_name] = latency
        logger.debug(f"Pipeline stage completed: {stage_name} ({latency:.1f}ms)")

    def record_llm_call(self, tokens: int = 0) -> None:
        """Record an LLM API call."""
        self.llm_calls += 1
        self.token_usage += tokens

    def record_papers_collected(self, count: int) -> None:
        self.papers_collected += count

    def record_papers_after_dedup(self, count: int) -> None:
        self.papers_after_dedup = count

    def record_papers_after_rerank(self, count: int) -> None:
        self.papers_after_rerank = count

    def record_papers_final(self, count: int) -> None:
        self.papers_final = count

    def record_model_used(self, model: str) -> None:
        if model not in self.models_used:
            self.models_used.append(model)

    def record_source_used(self, source: str) -> None:
        if source not in self.search_sources_used:
            self.search_sources_used.append(source)

    def record_thompson_allocation(self, allocations: dict[str, int]) -> None:
        self.thompson_allocations = allocations

    def get_metrics(self) -> PipelineMetrics:
        """Build the final metrics object."""
        total_latency = (time.time() - self.start_time) * 1000
        return PipelineMetrics(
            request_id=self.request_id,
            query=self.query,
            total_latency_ms=total_latency,
            stage_latencies=dict(self.stage_latencies),
            llm_calls=self.llm_calls,
            token_usage=self.token_usage,
            papers_collected=self.papers_collected,
            papers_after_dedup=self.papers_after_dedup,
            papers_after_rerank=self.papers_after_rerank,
            papers_final=self.papers_final,
            models_used=list(self.models_used),
            search_sources_used=list(self.search_sources_used),
            thompson_allocations=dict(self.thompson_allocations),
        )
