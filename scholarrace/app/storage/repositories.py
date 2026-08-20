"""Repository pattern for database operations."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select, update, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper, PaperIdentity
from app.models.agent import AgentRun, ModelConfidence
from app.storage.orm_models import (
    AgentRunORM,
    FeedbackORM,
    ModelConfidenceORM,
    PaperORM,
    QueryLogORM,
)


class PaperRepository:
    """Repository for Paper CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, paper: Paper) -> PaperORM:
        orm = PaperORM(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            semantic_scholar_id=paper.semantic_scholar_id,
            openalex_id=paper.openalex_id,
            pubmed_id=paper.pubmed_id,
            url=paper.url,
            pdf_url=paper.pdf_url,
            citation_count=paper.citation_count,
            reference_count=paper.reference_count,
            fields_of_study=paper.fields_of_study,
            keywords=paper.keywords,
            source=paper.source,
        )
        await self.session.merge(orm)
        await self.session.commit()
        return orm

    async def get_by_id(self, paper_id: str) -> Optional[PaperORM]:
        result = await self.session.execute(
            select(PaperORM).where(PaperORM.paper_id == paper_id)
        )
        return result.scalar_one_or_none()

    async def get_by_doi(self, doi: str) -> Optional[PaperORM]:
        result = await self.session.execute(
            select(PaperORM).where(PaperORM.doi == doi)
        )
        return result.scalar_one_or_none()

    async def get_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperORM]:
        result = await self.session.execute(
            select(PaperORM).where(PaperORM.arxiv_id == arxiv_id)
        )
        return result.scalar_one_or_none()

    async def search_by_title(self, title: str, limit: int = 10) -> list[PaperORM]:
        result = await self.session.execute(
            select(PaperORM)
            .where(PaperORM.title.ilike(f"%{title}%"))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_all(self) -> None:
        await self.session.execute(delete(PaperORM))
        await self.session.commit()


class AgentRunRepository:
    """Repository for agent run records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, run: AgentRun) -> AgentRunORM:
        orm = AgentRunORM(
            request_id=run.request_id,
            model_name=run.model_name,
            query_text=run.query_text,
            generated_candidates=run.generated_candidates,
            latency_ms=run.latency_ms,
            token_usage=run.token_usage,
            success=run.success,
            error=run.error,
        )
        self.session.add(orm)
        await self.session.commit()
        return orm

    async def get_by_request(self, request_id: str) -> list[AgentRunORM]:
        result = await self.session.execute(
            select(AgentRunORM)
            .where(AgentRunORM.request_id == request_id)
            .order_by(AgentRunORM.timestamp)
        )
        return list(result.scalars().all())


class ModelConfidenceRepository:
    """Repository for Thompson Sampling confidence state."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(
        self, model: str, domain: str, query_type: str
    ) -> Optional[ModelConfidenceORM]:
        result = await self.session.execute(
            select(ModelConfidenceORM).where(
                and_(
                    ModelConfidenceORM.model_name == model,
                    ModelConfidenceORM.domain == domain,
                    ModelConfidenceORM.query_type == query_type,
                )
            )
        )
        return result.scalar_one_or_none()

    async def upsert(self, confidence: ModelConfidence) -> ModelConfidenceORM:
        existing = await self.get(
            confidence.model_name, confidence.domain, confidence.query_type
        )
        if existing:
            existing.alpha = confidence.alpha
            existing.beta = confidence.beta
            existing.total_runs = confidence.total_runs
            existing.avg_reward = confidence.avg_reward
        else:
            existing = ModelConfidenceORM(
                model_name=confidence.model_name,
                domain=confidence.domain,
                query_type=confidence.query_type,
                alpha=confidence.alpha,
                beta=confidence.beta,
                total_runs=confidence.total_runs,
                avg_reward=confidence.avg_reward,
            )
            self.session.add(existing)
        await self.session.commit()
        return existing

    async def get_all_for_domain(
        self, domain: str
    ) -> list[ModelConfidenceORM]:
        result = await self.session.execute(
            select(ModelConfidenceORM).where(
                ModelConfidenceORM.domain == domain
            )
        )
        return list(result.scalars().all())


class QueryLogRepository:
    """Repository for query logs."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(
        self,
        original_query: str,
        semantic_core: str,
        domain: str,
        intent: str,
        options: dict,
        result_count: int,
        latency_ms: float,
        request_id: str = "",
    ) -> QueryLogORM:
        orm = QueryLogORM(
            request_id=request_id,
            original_query=original_query,
            semantic_core=semantic_core,
            domain=domain,
            intent=intent,
            options=options,
            result_count=result_count,
            latency_ms=latency_ms,
        )
        self.session.add(orm)
        await self.session.commit()
        return orm

    async def get_recent(self, limit: int = 20) -> list[QueryLogORM]:
        result = await self.session.execute(
            select(QueryLogORM)
            .order_by(QueryLogORM.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class FeedbackRepository:
    """Repository for user feedback."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(
        self,
        request_id: str,
        rating: int,
        comment: str = "",
        paper_id: Optional[str] = None,
    ) -> FeedbackORM:
        orm = FeedbackORM(
            request_id=request_id,
            paper_id=paper_id,
            rating=rating,
            comment=comment,
        )
        self.session.add(orm)
        await self.session.commit()
        return orm

    async def get_by_request(self, request_id: str) -> list[FeedbackORM]:
        result = await self.session.execute(
            select(FeedbackORM).where(FeedbackORM.request_id == request_id)
        )
        return list(result.scalars().all())
