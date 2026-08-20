"""Candidate query and judge result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
