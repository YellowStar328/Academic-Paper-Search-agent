"""Tests for LLM providers and agents."""

import asyncio
import json

import pytest

from app.agents.base import (
    BaseOpenAIProvider,
    LLMProvider,
    LLMResponse,
    create_qwen_provider,
    create_deepseek_provider,
    create_glm_provider,
    create_strong_judge_provider,
)
from app.agents.mock import MockLLMProvider
from app.agents.qwen import QwenAgent
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.judge import StrongJudge, PaperJudge
from app.models.query import SearchQuery
from app.models.candidate import CandidateQuery
from app.models.paper import Paper


# ---------- MockLLMProvider Tests ----------

class TestMockLLMProvider:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self):
        provider = MockLLMProvider(model_name="mock-test")
        response = await provider.generate(prompt="What is machine learning?")
        assert isinstance(response, LLMResponse)
        assert response.model == "mock-test"
        assert response.success is True
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_deterministic_for_same_prompt(self):
        provider = MockLLMProvider()
        r1 = await provider.generate(prompt="test prompt 123")
        r2 = await provider.generate(prompt="test prompt 123")
        assert r1.content == r2.content

    @pytest.mark.asyncio
    async def test_query_understanding_response(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            prompt='query understanding for "transformer architectures"'
        )
        data = json.loads(response.content)
        assert "semantic_core" in data
        assert "domain" in data
        assert "sub_queries" in data

    @pytest.mark.asyncio
    async def test_candidate_generation_response(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            prompt="generate candidate sub-queries for machine learning"
        )
        data = json.loads(response.content)
        assert "candidates" in data
        assert len(data["candidates"]) > 0

    @pytest.mark.asyncio
    async def test_judge_response(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            prompt="Evaluate this candidate query. Score coverage specificity novelty"
        )
        data = json.loads(response.content)
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_paper_relevance_response(self):
        provider = MockLLMProvider()
        response = await provider.generate(
            prompt="evaluate paper relevance to query. paper title and abstract"
        )
        data = json.loads(response.content)
        assert "relevance_score" in data

    @pytest.mark.asyncio
    async def test_token_usage_nonzero(self):
        provider = MockLLMProvider()
        response = await provider.generate(prompt="some prompt here")
        assert response.token_usage > 0

    @pytest.mark.asyncio
    async def test_latency_nonzero(self):
        provider = MockLLMProvider()
        response = await provider.generate(prompt="test")
        assert response.latency_ms > 0


# ---------- Provider Factory Tests ----------

class TestProviderFactories:
    def test_create_qwen_provider(self):
        provider = create_qwen_provider()
        assert isinstance(provider, BaseOpenAIProvider)
        assert "qwen" in provider.model_name or provider.model_name

    def test_create_deepseek_provider(self):
        provider = create_deepseek_provider()
        assert isinstance(provider, BaseOpenAIProvider)

    def test_create_glm_provider(self):
        provider = create_glm_provider()
        assert isinstance(provider, BaseOpenAIProvider)

    def test_create_strong_judge_provider(self):
        provider = create_strong_judge_provider()
        assert isinstance(provider, BaseOpenAIProvider)

    def test_provider_implements_protocol(self):
        provider = create_qwen_provider()
        # Should have generate method and model_name attribute
        assert hasattr(provider, "generate")
        assert hasattr(provider, "model_name")
        assert callable(getattr(provider, "generate"))


# ---------- Agent Tests ----------

class TestQwenAgent:
    @pytest.mark.asyncio
    async def test_generate_queries(self):
        mock_provider = MockLLMProvider(model_name="qwen")
        agent = QwenAgent(provider=mock_provider)
        query = SearchQuery(
            original_query="transformer architectures for NLP",
            semantic_core="transformer architecture survey",
            domain="cs",
        )
        candidates = await agent.generate_queries(query)
        assert len(candidates) > 0
        assert all(c.proposer_model == "qwen" for c in candidates)
        assert all(c.query for c in candidates)

    @pytest.mark.asyncio
    async def test_failed_provider_returns_empty(self):
        class FailingProvider:
            model_name = "qwen"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(
                    content="", model="qwen", success=False, error="API error"
                )

        agent = QwenAgent(provider=FailingProvider())
        query = SearchQuery(original_query="test", semantic_core="test")
        candidates = await agent.generate_queries(query)
        assert candidates == []


class TestDeepSeekAgent:
    @pytest.mark.asyncio
    async def test_generate_queries(self):
        mock_provider = MockLLMProvider(model_name="deepseek")
        agent = DeepSeekAgent(provider=mock_provider)
        query = SearchQuery(
            original_query="graph neural networks",
            semantic_core="GNN survey",
            domain="cs",
        )
        candidates = await agent.generate_queries(query)
        assert len(candidates) > 0
        assert all(c.proposer_model == "deepseek" for c in candidates)


class TestGLMAgent:
    @pytest.mark.asyncio
    async def test_generate_queries(self):
        mock_provider = MockLLMProvider(model_name="glm")
        agent = GLMAgent(provider=mock_provider)
        query = SearchQuery(
            original_query="reinforcement learning",
            semantic_core="RL methods",
            domain="cs",
        )
        candidates = await agent.generate_queries(query)
        assert len(candidates) > 0
        assert all(c.proposer_model == "glm" for c in candidates)


class TestStrongJudge:
    @pytest.mark.asyncio
    async def test_evaluate_query_candidate(self):
        mock_provider = MockLLMProvider(model_name="strong_judge")
        judge = StrongJudge(provider=mock_provider)
        candidate = CandidateQuery(
            query="transformer survey", proposer_model="qwen"
        )
        result = await judge.evaluate_query_candidate(
            "transformer architectures", candidate
        )
        assert 0.0 <= result.score <= 1.0
        assert result.candidate.query == "transformer survey"

    @pytest.mark.asyncio
    async def test_evaluate_candidates_batch(self):
        mock_provider = MockLLMProvider(model_name="strong_judge")
        judge = StrongJudge(provider=mock_provider)
        candidates = [
            CandidateQuery(query="q1", proposer_model="qwen"),
            CandidateQuery(query="q2", proposer_model="deepseek"),
            CandidateQuery(query="q3", proposer_model="glm"),
        ]
        results = await judge.evaluate_candidates_batch("test query", candidates)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_failed_judge_returns_default(self):
        class FailingProvider:
            model_name = "judge"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(
                    content="", model="judge", success=False, error="error"
                )

        judge = StrongJudge(provider=FailingProvider())
        candidate = CandidateQuery(query="test", proposer_model="qwen")
        result = await judge.evaluate_query_candidate("orig", candidate)
        assert result.score == 0.5


class TestPaperJudge:
    @pytest.mark.asyncio
    async def test_evaluate_paper(self):
        mock_provider = MockLLMProvider(model_name="paper_judge")
        judge = PaperJudge(provider=mock_provider)
        paper = Paper(
            paper_id="p1",
            title="Attention Is All You Need",
            abstract="We propose a new architecture based on attention mechanism.",
        )
        result = await judge.evaluate_paper("transformer survey", paper)
        assert result.paper_id == "p1"
        assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_papers_batch(self):
        mock_provider = MockLLMProvider(model_name="paper_judge")
        judge = PaperJudge(provider=mock_provider)
        papers = [
            Paper(paper_id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
            for i in range(5)
        ]
        results = await judge.evaluate_papers_batch("query", papers)
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0


class TestMultiAgentParallel:
    """Test that multiple agents can run in parallel."""

    @pytest.mark.asyncio
    async def test_parallel_query_generation(self):
        agents = [
            QwenAgent(provider=MockLLMProvider(model_name="qwen")),
            DeepSeekAgent(provider=MockLLMProvider(model_name="deepseek")),
            GLMAgent(provider=MockLLMProvider(model_name="glm")),
        ]
        query = SearchQuery(
            original_query="large language models",
            semantic_core="LLM survey",
            domain="cs",
        )
        results = await asyncio.gather(*[a.generate_queries(query) for a in agents])
        assert len(results) == 3
        # Each agent should produce at least 1 candidate
        for r in results:
            assert len(r) > 0
        # All proposers should be different
        proposers = {r[0].proposer_model for r in results}
        assert len(proposers) == 3
