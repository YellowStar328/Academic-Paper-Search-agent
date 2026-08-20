"""Tests for storage layer: database, ORM, repositories, and cache."""

import asyncio
from uuid import uuid4

import pytest

from app.models.paper import Paper, PaperIdentity
from app.models.agent import AgentRun, ModelConfidence
from app.storage.cache import Cache
from app.storage.database import async_session, close_db, init_db, reset_engine
from app.storage.repositories import (
    AgentRunRepository,
    FeedbackRepository,
    ModelConfidenceRepository,
    PaperRepository,
    QueryLogRepository,
)


@pytest.fixture(scope="function")
async def db_session():
    """Provide a fresh in-memory DB for each test."""
    await reset_engine()
    await init_db()
    async with async_session() as session:
        yield session
    await close_db()
    await reset_engine()


@pytest.fixture(scope="function")
async def cache():
    """Provide a test cache."""
    c = Cache(is_test=True)
    await c.connect()
    yield c
    await c.disconnect()


# ---------- Cache Tests ----------

class TestCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("key1", {"data": "value"}, ttl=60)
        result = await cache.get("key1")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_missing(self, cache):
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        await cache.set("key2", "value", ttl=60)
        await cache.delete("key2")
        result = await cache.get("key2")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        await cache.set("key3", "val", ttl=60)
        assert await cache.exists("key3") is True
        assert await cache.exists("missing") is False

    @pytest.mark.asyncio
    async def test_make_key(self, cache):
        key = cache.make_key("source", "query", "filter")
        assert key == "source:query:filter"

    @pytest.mark.asyncio
    async def test_flush(self, cache):
        await cache.set("key4", "val", ttl=60)
        await cache.flush()
        assert await cache.get("key4") is None

    @pytest.mark.asyncio
    async def test_json_serialization(self, cache):
        data = {"list": [1, 2, 3], "nested": {"a": "b"}}
        await cache.set("json_key", data, ttl=60)
        result = await cache.get("json_key")
        assert result == data


# ---------- ORM / Repository Tests ----------

class TestPaperRepository:
    @pytest.mark.asyncio
    async def test_save_and_get(self, db_session):
        repo = PaperRepository(db_session)
        paper = Paper(
            paper_id=str(uuid4()),
            title="Attention Is All You Need",
            abstract="We propose a new architecture...",
            authors=["Vaswani", "Shazeer"],
            year=2017,
            doi="10.1234/test",
            arxiv_id="1706.03762",
            source="arxiv",
        )
        await repo.save(paper)

        result = await repo.get_by_id(paper.paper_id)
        assert result is not None
        assert result.title == "Attention Is All You Need"
        assert result.year == 2017

    @pytest.mark.asyncio
    async def test_get_by_doi(self, db_session):
        repo = PaperRepository(db_session)
        paper = Paper(
            paper_id=str(uuid4()),
            title="Test Paper",
            doi="10.9999/test",
        )
        await repo.save(paper)

        result = await repo.get_by_doi("10.9999/test")
        assert result is not None
        assert result.title == "Test Paper"

    @pytest.mark.asyncio
    async def test_get_by_arxiv_id(self, db_session):
        repo = PaperRepository(db_session)
        paper = Paper(
            paper_id=str(uuid4()),
            title="ArXiv Paper",
            arxiv_id="2401.12345",
        )
        await repo.save(paper)

        result = await repo.get_by_arxiv_id("2401.12345")
        assert result is not None
        assert result.title == "ArXiv Paper"

    @pytest.mark.asyncio
    async def test_search_by_title(self, db_session):
        repo = PaperRepository(db_session)
        paper = Paper(
            paper_id=str(uuid4()),
            title="Deep Learning for NLP",
        )
        await repo.save(paper)

        results = await repo.search_by_title("Deep Learning")
        assert len(results) >= 1
        assert "Deep Learning" in results[0].title

    @pytest.mark.asyncio
    async def test_delete_all(self, db_session):
        repo = PaperRepository(db_session)
        paper = Paper(paper_id=str(uuid4()), title="To Delete")
        await repo.save(paper)
        await repo.delete_all()

        result = await repo.get_by_id(paper.paper_id)
        assert result is None


class TestAgentRunRepository:
    @pytest.mark.asyncio
    async def test_save_and_get(self, db_session):
        repo = AgentRunRepository(db_session)
        run = AgentRun(
            request_id="req-123",
            model_name="qwen",
            query_text="transformer survey",
            generated_candidates=["q1", "q2"],
            latency_ms=1500.0,
            token_usage=500,
            success=True,
        )
        await repo.save(run)

        results = await repo.get_by_request("req-123")
        assert len(results) == 1
        assert results[0].model_name == "qwen"
        assert results[0].token_usage == 500


class TestModelConfidenceRepository:
    @pytest.mark.asyncio
    async def test_upsert_new(self, db_session):
        repo = ModelConfidenceRepository(db_session)
        mc = ModelConfidence(
            model_name="qwen",
            domain="cs",
            query_type="cs:survey",
            alpha=2.0,
            beta=1.0,
            total_runs=3,
            avg_reward=0.67,
        )
        await repo.upsert(mc)

        result = await repo.get("qwen", "cs", "cs:survey")
        assert result is not None
        assert result.alpha == 2.0
        assert result.total_runs == 3

    @pytest.mark.asyncio
    async def test_upsert_existing(self, db_session):
        repo = ModelConfidenceRepository(db_session)
        mc = ModelConfidence(
            model_name="deepseek",
            domain="cs",
            query_type="cs:survey",
            alpha=1.0,
            beta=1.0,
        )
        await repo.upsert(mc)

        mc_updated = ModelConfidence(
            model_name="deepseek",
            domain="cs",
            query_type="cs:survey",
            alpha=5.0,
            beta=2.0,
            total_runs=6,
            avg_reward=0.83,
        )
        await repo.upsert(mc_updated)

        result = await repo.get("deepseek", "cs", "cs:survey")
        assert result is not None
        assert result.alpha == 5.0
        assert result.total_runs == 6

    @pytest.mark.asyncio
    async def test_get_all_for_domain(self, db_session):
        repo = ModelConfidenceRepository(db_session)
        for model in ["qwen", "deepseek", "glm"]:
            mc = ModelConfidence(
                model_name=model, domain="cs", query_type="cs:survey"
            )
            await repo.upsert(mc)

        results = await repo.get_all_for_domain("cs")
        assert len(results) == 3


class TestQueryLogRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_recent(self, db_session):
        repo = QueryLogRepository(db_session)
        await repo.save(
            original_query="machine learning",
            semantic_core="ML survey",
            domain="cs",
            intent="survey",
            options={"top_k": 20},
            result_count=15,
            latency_ms=3000.0,
            request_id="req-1",
        )
        await repo.save(
            original_query="transformers",
            semantic_core="transformer architectures",
            domain="cs",
            intent="survey",
            options={"top_k": 10},
            result_count=8,
            latency_ms=2500.0,
            request_id="req-2",
        )

        recent = await repo.get_recent(limit=10)
        assert len(recent) == 2


class TestFeedbackRepository:
    @pytest.mark.asyncio
    async def test_save_and_get(self, db_session):
        repo = FeedbackRepository(db_session)
        await repo.save(
            request_id="req-1",
            rating=5,
            comment="Great results",
            paper_id="p-123",
        )
        await repo.save(
            request_id="req-1",
            rating=3,
            comment="Okay",
        )

        results = await repo.get_by_request("req-1")
        assert len(results) == 2
        assert results[0].rating == 5
