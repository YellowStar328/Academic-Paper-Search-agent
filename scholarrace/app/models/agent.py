"""Agent run and model confidence tracking models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentRun(BaseModel):
    """Record of a single agent's execution."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    model_name: str
    query_text: str = ""
    generated_candidates: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ModelConfidence(BaseModel):
    """Confidence entry for a (model, domain, query_type) triple."""

    model_name: str
    domain: str
    query_type: str
    alpha: float = 1.0
    beta: float = 1.0
    total_runs: int = 0
    avg_reward: float = 0.0

    def mean(self) -> float:
        """Posterior mean of the Beta distribution."""
        total = self.alpha + self.beta
        if total == 0:
            return 0.5
        return self.alpha / total
