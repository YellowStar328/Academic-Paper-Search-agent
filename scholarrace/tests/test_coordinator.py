"""Tests for MultiAgentCoordinator."""

import asyncio

import pytest

from app.agents.coordinator import MultiAgentCoordinator
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.mock import MockLLMProvider
from app.agents.qwen import QwenAgent
from app.models.query import SearchQuery


@pytest.fixture
def coordinator():
    return MultiAgentCoordinator(
        qwen_agent=QwenAgent(provider=MockLLMProvider(model_name="qwen")),
        deepseek_agent=DeepSeekAgent(provider=MockLLMProvider(model_name="deepseek")),
        glm_agent=GLMAgent(provider=MockLLMProvider(model_name="glm")),
    )


@pytest.fixture
def sample_query():
    return SearchQuery(
        original_query="transformer architectures for NLP",
        semantic_core="transformer architecture survey",
        domain="cs",
    )


class TestMultiAgentCoordinator:
    def test_model_names(self, coordinator):
        names = coordinator.model_names
        assert "qwen" in names
        assert "deepseek" in names
        assert "glm" in names
        assert len(names) == 3

    @pytest.mark.asyncio
    async def test_generate_candidates(self, coordinator, sample_query):
        candidates, runs = await coordinator.generate_candidates(
            sample_query, request_id="req-1"
        )
        # Each mock agent generates 2 candidates, so 3 agents => 6
        assert len(candidates) >= 3
        assert len(runs) == 3
        # All runs should be successful
        assert all(r.success for r in runs)
        # All candidates should have proposer model set
        proposers = {c.proposer_model for c in candidates}
        assert "qwen" in proposers
        assert "deepseek" in proposers
        assert "glm" in proposers

    @pytest.mark.asyncio
    async def test_generate_candidates_single(self, coordinator, sample_query):
        candidates, runs = await coordinator.generate_candidates_single(
            sample_query, "qwen", request_id="req-2"
        )
        assert len(candidates) > 0
        assert len(runs) == 1
        assert runs[0].model_name == "qwen"
        assert all(c.proposer_model == "qwen" for c in candidates)

    @pytest.mark.asyncio
    async def test_generate_candidates_single_unknown_model(self, coordinator, sample_query):
        with pytest.raises(ValueError, match="Unknown model"):
            await coordinator.generate_candidates_single(
                sample_query, "unknown_model"
            )

    @pytest.mark.asyncio
    async def test_generate_candidates_random(self, coordinator, sample_query):
        candidates, runs = await coordinator.generate_candidates_random(
            sample_query, request_id="req-3"
        )
        assert len(runs) == 1
        assert runs[0].success

    @pytest.mark.asyncio
    async def test_agent_failure_isolated(self, sample_query):
        """Test that a single agent failure doesn't block others.

        When an LLM provider returns failure, the agent returns empty
        candidates but does not crash, allowing other agents to continue.
        """
        from app.agents.base import LLMResponse

        class FailingProvider:
            model_name = "qwen"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="", model="qwen", success=False, error="API error")

        coordinator = MultiAgentCoordinator(
            qwen_agent=QwenAgent(provider=FailingProvider()),
            deepseek_agent=DeepSeekAgent(provider=MockLLMProvider(model_name="deepseek")),
            glm_agent=GLMAgent(provider=MockLLMProvider(model_name="glm")),
        )
        candidates, runs = await coordinator.generate_candidates(
            sample_query, request_id="req-4"
        )
        # All 3 runs should complete (no crash)
        assert len(runs) == 3
        # Qwen should have no candidates (LLM failed)
        qwen_candidates = [c for c in candidates if c.proposer_model == "qwen"]
        assert len(qwen_candidates) == 0
        # But deepseek and glm should still produce candidates
        deepseek_candidates = [c for c in candidates if c.proposer_model == "deepseek"]
        assert len(deepseek_candidates) > 0
        glm_candidates = [c for c in candidates if c.proposer_model == "glm"]
        assert len(glm_candidates) > 0

    @pytest.mark.asyncio
    async def test_request_id_propagated(self, coordinator, sample_query):
        _, runs = await coordinator.generate_candidates(
            sample_query, request_id="req-test-123"
        )
        assert all(r.request_id == "req-test-123" for r in runs)

    @pytest.mark.asyncio
    async def test_latency_recorded(self, coordinator, sample_query):
        _, runs = await coordinator.generate_candidates(sample_query)
        for r in runs:
            assert r.latency_ms > 0

    @pytest.mark.asyncio
    async def test_default_agents_created(self, sample_query):
        """Test that coordinator creates default agents if none provided."""
        # In test env, defaults use MockLLMProvider
        coordinator = MultiAgentCoordinator()
        assert len(coordinator.agents) == 3
        candidates, runs = await coordinator.generate_candidates(sample_query)
        assert len(runs) == 3
