"""Tests for FastAPI API endpoints.

Uses TestClient with a mock pipeline to test API contract without
running the full pipeline.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app, set_pipeline
from app.models.paper import Paper, PaperIdentity
from app.models.result import (
    PaperWithScores,
    ResearchGraph,
    SearchResult,
    SearchSummary,
)
from app.pipeline.search_pipeline import SearchPipeline


def make_scored_paper(title: str = "Test Paper") -> PaperWithScores:
    paper = Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(
            doi="10.1/test",
            normalized_title=title.lower().replace(" ", ""),
            year=2024,
        ),
        title=title,
        abstract="Test abstract",
        year=2024,
        citation_count=50,
        authors=["Alice"],
        venue="NeurIPS",
        source="arxiv",
    )
    return PaperWithScores(
        paper=paper,
        relevance_score=0.9,
        authority_score=0.8,
        recency_score=1.0,
        citation_score=0.5,
        diversity_score=0.7,
        redundancy_score=0.9,
        final_score=0.85,
        embedding_similarity=0.9,
        judge_reasoning="Highly relevant",
    )


def make_mock_pipeline() -> MagicMock:
    """Create a mock pipeline that returns a fixed SearchResult."""
    pipeline = MagicMock(spec=SearchPipeline)
    scored = make_scored_paper()

    result = SearchResult(
        request_id="test-req-123",
        query="machine learning",
        papers=[scored],
        graph=ResearchGraph(),
        summary=SearchSummary(
            total_papers=1,
            query="machine learning",
            domain="cs",
            intent="survey",
            top_paper_title="Test Paper",
            clusters_count=1,
        ),
        latency_ms=150.0,
    )

    pipeline.run = AsyncMock(return_value=result)
    return pipeline


@pytest.fixture
def client():
    """Test client with mock pipeline."""
    pipeline = make_mock_pipeline()
    set_pipeline(pipeline)
    with TestClient(app) as c:
        yield c
    set_pipeline(None)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSearchEndpoint:
    """Tests for POST /api/search."""

    def test_search_basic(self, client):
        response = client.post("/api/search", json={"query": "machine learning"})
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "test-req-123"
        assert data["query"] == "machine learning"
        assert data["total_papers"] == 1
        assert len(data["papers"]) == 1
        assert data["papers"][0]["title"] == "Test Paper"
        assert data["papers"][0]["final_score"] == 0.85
        assert data["latency_ms"] == 150.0

    def test_search_with_options(self, client):
        response = client.post(
            "/api/search",
            json={
                "query": "deep learning",
                "max_results": 5,
                "year_from": 2020,
                "year_to": 2024,
                "enable_citation_expansion": False,
                "enable_embedding_rerank": True,
            },
        )
        assert response.status_code == 200

    def test_search_empty_query_rejected(self, client):
        response = client.post("/api/search", json={"query": ""})
        assert response.status_code == 422

    def test_search_missing_query_rejected(self, client):
        response = client.post("/api/search", json={})
        assert response.status_code == 422

    def test_search_max_results_validation(self, client):
        response = client.post(
            "/api/search", json={"query": "test", "max_results": 0}
        )
        assert response.status_code == 422
        response = client.post(
            "/api/search", json={"query": "test", "max_results": 101}
        )
        assert response.status_code == 422

    def test_search_summary_returned(self, client):
        response = client.post("/api/search", json={"query": "transformers"})
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] is not None
        assert data["summary"]["total_papers"] == 1
        assert data["summary"]["top_paper_title"] == "Test Paper"

    def test_search_paper_scores_included(self, client):
        response = client.post("/api/search", json={"query": "AI"})
        data = response.json()
        paper = data["papers"][0]
        assert "relevance_score" in paper
        assert "authority_score" in paper
        assert "recency_score" in paper
        assert "final_score" in paper

    def test_search_paper_metadata_included(self, client):
        response = client.post("/api/search", json={"query": "neural networks"})
        data = response.json()
        paper = data["papers"][0]
        assert "title" in paper
        assert "abstract" in paper
        assert "year" in paper
        assert "authors" in paper
        assert "citation_count" in paper
        assert "venue" in paper
        assert "doi" in paper

    def test_search_no_pipeline_returns_503(self):
        set_pipeline(None)
        client = TestClient(app)
        with patch("app.main.create_pipeline", side_effect=RuntimeError("init failed")):
            response = client.post("/api/search", json={"query": "test"})
            assert response.status_code == 503


class TestFeedbackEndpoint:
    """Tests for POST /api/feedback."""

    def test_feedback_basic(self, client):
        response = client.post(
            "/api/feedback",
            json={
                "request_id": "req-123",
                "paper_id": "paper-456",
                "rating": 5,
                "is_relevant": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["request_id"] == "req-123"
        assert data["paper_id"] == "paper-456"
        assert data["recorded"] is True

    def test_feedback_with_comment(self, client):
        response = client.post(
            "/api/feedback",
            json={
                "request_id": "req-123",
                "paper_id": "paper-456",
                "rating": 3,
                "comment": "Somewhat relevant",
                "is_relevant": True,
            },
        )
        assert response.status_code == 200

    def test_feedback_rating_validation(self, client):
        response = client.post(
            "/api/feedback",
            json={
                "request_id": "req-123",
                "paper_id": "paper-456",
                "rating": 0,
                "is_relevant": True,
            },
        )
        assert response.status_code == 422

        response = client.post(
            "/api/feedback",
            json={
                "request_id": "req-123",
                "paper_id": "paper-456",
                "rating": 6,
                "is_relevant": True,
            },
        )
        assert response.status_code == 422

    def test_feedback_missing_fields(self, client):
        response = client.post(
            "/api/feedback",
            json={"request_id": "req-123"},
        )
        assert response.status_code == 422
