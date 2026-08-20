"""ORM models for persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.orm_base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid4())


class PaperORM(Base):
    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    title: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    venue: Mapped[str] = mapped_column(Text, default="")
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    semantic_scholar_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    openalex_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    pubmed_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    url: Mapped[str] = mapped_column(Text, default="")
    pdf_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    reference_count: Mapped[int] = mapped_column(Integer, default=0)
    fields_of_study: Mapped[list] = mapped_column(JSON, default=list)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(50), default="unknown")
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:
        return f"<PaperORM {self.paper_id} {self.title[:50]}>"


class QueryLogORM(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    request_id: Mapped[str] = mapped_column(String(36), default=_new_uuid)
    original_query: Mapped[str] = mapped_column(Text)
    semantic_core: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str] = mapped_column(String(100), default="general")
    intent: Mapped[str] = mapped_column(String(50), default="survey")
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class AgentRunORM(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    model_name: Mapped[str] = mapped_column(String(100))
    query_text: Mapped[str] = mapped_column(Text, default="")
    generated_candidates: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    token_usage: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Integer, default=1)  # SQLite bool
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class ModelConfidenceORM(Base):
    __tablename__ = "model_confidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    domain: Mapped[str] = mapped_column(String(100), index=True)
    query_type: Mapped[str] = mapped_column(String(100), index=True)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    total_runs: Mapped[int] = mapped_column(Integer, default=0)
    avg_reward: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class FeedbackORM(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    paper_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    rating: Mapped[int] = mapped_column(Integer)  # 1-5
    comment: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
