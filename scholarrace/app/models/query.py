"""Query models: structured representation of user search intent."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """The high-level intent of a user's query."""

    SURVEY = "survey"  # Broad overview of a field
    COMPARISON = "comparison"  # Compare methods/approaches
    METHOD = "method"  # Find methodology details
    APPLICATION = "application"  # Find applications of a technique
    DEFINITION = "definition"  # Understand a concept
    RECENT = "recent"  # Latest developments
    REPRODUCTION = "reproduction"  # Reproduce results


class LogicOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class HardFilter(BaseModel):
    """Hard constraints (must be satisfied)."""

    year_start: Optional[int] = None
    year_end: Optional[int] = None
    venue: Optional[str] = None
    open_access_only: bool = False
    min_citations: Optional[int] = None
    fields_of_study: list[str] = Field(default_factory=list)
    has_code: bool = False
    language: Optional[str] = None


class SearchOptions(BaseModel):
    """User-facing search options."""

    top_k: int = Field(default=20, ge=1, le=200)
    mode: str = Field(default="auto", description="auto | human_review")
    enable_citation_expansion: bool = True
    enable_multi_agent: bool = True
    enable_thompson_sampling: bool = True
    enable_embedding_rerank: bool = True
    enable_mmr: bool = True
    strategy: str = Field(
        default="full_pipeline",
        description="single_model|random_multi_agent|greedy_multi_agent|thompson|thompson_plus_citation|full_pipeline",
    )


class SearchQuery(BaseModel):
    """Structured query object output by QueryParser.

    Separates hard constraints from semantic core.
    """

    original_query: str
    semantic_core: str = Field(..., description="The distilled semantic intent")
    domain: str = Field(default="general", description="Academic domain")
    intent: QueryIntent = QueryIntent.SURVEY
    sub_queries: list[str] = Field(default_factory=list)
    hard_filters: HardFilter = Field(default_factory=HardFilter)
    logic: list[LogicOperator] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    options: SearchOptions = Field(default_factory=SearchOptions)

    @property
    def query_type(self) -> str:
        """Compact string key for confidence tracking."""
        return f"{self.domain}:{self.intent.value}"


class UserQuery(BaseModel):
    """Raw user input."""

    query: str
    options: SearchOptions = Field(default_factory=SearchOptions)
