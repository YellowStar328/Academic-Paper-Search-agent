"""Candidate query and judge result models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.paper import Paper


class CandidateQuery(BaseModel):
    """A candidate sub-query proposed by an agent."""

    query: str
    proposer_model: str
    rationale: str = ""
    keywords: list[str] = Field(default_factory=list)
    logic: str = "OR"
    sub_queries: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    """Result of the strong model judging a candidate query."""

    candidate: CandidateQuery
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)


class PaperJudgeResult(BaseModel):
    """Result of the strong model judging a single paper's relevance."""

    paper_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    key_findings: list[str] = Field(default_factory=list)


class AgentPaperReport(BaseModel):
    """A single paper entry in an agent's procurement report.

    The agent (Qwen/DeepSeek/GLM) judges each paper it retrieved
    and reports its assessment to the strong model.
    """

    paper: Paper
    agent_relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    agent_reasoning: str = ""
    agent_key_findings: list[str] = Field(default_factory=list)
    search_query_used: str = ""
    source: str = ""


class AgentReport(BaseModel):
    """Procurement report from one agent to the strong model.

    Each agent independently:
    1. Generates search keywords
    2. Dispatches retrieval providers
    3. Judges paper abstracts
    4. Writes this report

    The strong model then reviews all three reports and makes the
    final selection.
    """

    agent_model: str
    search_queries: list[str] = Field(default_factory=list)
    rationale: str = ""
    paper_reports: list[AgentPaperReport] = Field(default_factory=list)
    token_usage: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
