"""Tests for StrongJudge and PaperJudge — candidate evaluation and paper ranking."""

import pytest

from app.agents.judge import StrongJudge, PaperJudge
from app.agents.mock import MockLLMProvider
from app.models.candidate import CandidateQuery
from app.models.paper import Paper


class TestStrongJudgeSelection:
    @pytest.mark.asyncio
    async def test_select_top_candidates(self):
        judge = StrongJudge(provider=MockLLMProvider())
        candidates = [
            CandidateQuery(query=f"query_{i}", proposer_model=m)
            for i, m in enumerate(["qwen", "deepseek", "glm"])
        ]
        top = await judge.select_top_candidates("original", candidates, top_k=2)
        assert len(top) <= 2
        # Should be sorted by score descending
        for i in range(len(top) - 1):
            assert top[i].score >= top[i + 1].score

    @pytest.mark.asyncio
    async def test_select_top_candidates_empty(self):
        judge = StrongJudge(provider=MockLLMProvider())
        top = await judge.select_top_candidates("orig", [], top_k=5)
        assert top == []

    @pytest.mark.asyncio
    async def test_select_top_candidates_all(self):
        judge = StrongJudge(provider=MockLLMProvider())
        candidates = [
            CandidateQuery(query="q1", proposer_model="qwen"),
            CandidateQuery(query="q2", proposer_model="deepseek"),
        ]
        top = await judge.select_top_candidates("orig", candidates, top_k=10)
        assert len(top) == 2

    @pytest.mark.asyncio
    async def test_select_top_candidates_scores_in_range(self):
        judge = StrongJudge(provider=MockLLMProvider())
        candidates = [
            CandidateQuery(query="q1", proposer_model="qwen"),
            CandidateQuery(query="q2", proposer_model="deepseek"),
            CandidateQuery(query="q3", proposer_model="glm"),
        ]
        top = await judge.select_top_candidates("orig", candidates, top_k=3)
        for r in top:
            assert 0.0 <= r.score <= 1.0
            assert 0.0 <= r.coverage <= 1.0
            assert 0.0 <= r.specificity <= 1.0
            assert 0.0 <= r.novelty <= 1.0


class TestPaperJudgeBatch:
    @pytest.mark.asyncio
    async def test_evaluate_papers_large_batch(self):
        """Test batch processing with more papers than batch_size."""
        judge = PaperJudge(provider=MockLLMProvider())
        papers = [
            Paper(paper_id=f"p{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
            for i in range(40)  # exceeds default batch_size=16
        ]
        results = await judge.evaluate_papers_batch("query", papers)
        assert len(results) == 40
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0
            assert 0.0 <= r.authority_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_papers_empty(self):
        judge = PaperJudge(provider=MockLLMProvider())
        results = await judge.evaluate_papers_batch("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_evaluate_paper_preserves_id(self):
        judge = PaperJudge(provider=MockLLMProvider())
        paper = Paper(paper_id="test-id-123", title="Test", abstract="Test abstract")
        result = await judge.evaluate_paper("query", paper)
        assert result.paper_id == "test-id-123"

    @pytest.mark.asyncio
    async def test_evaluate_paper_long_abstract_truncated(self):
        """Paper abstracts longer than 500 chars are truncated."""
        judge = PaperJudge(provider=MockLLMProvider())
        long_abstract = "A" * 1000
        paper = Paper(paper_id="p1", title="Test", abstract=long_abstract)
        # Should not crash
        result = await judge.evaluate_paper("query", paper)
        assert result.paper_id == "p1"

    @pytest.mark.asyncio
    async def test_evaluate_paper_no_abstract(self):
        judge = PaperJudge(provider=MockLLMProvider())
        paper = Paper(paper_id="p1", title="Test", abstract="")
        result = await judge.evaluate_paper("query", paper)
        assert result.paper_id == "p1"


class TestJudgeFailureHandling:
    @pytest.mark.asyncio
    async def test_strong_judge_parse_error_returns_default(self):
        from app.agents.base import LLMResponse

        class BadJsonProvider:
            model_name = "judge"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="not json", model="judge", success=True)

        judge = StrongJudge(provider=BadJsonProvider())
        candidate = CandidateQuery(query="test", proposer_model="qwen")
        result = await judge.evaluate_query_candidate("orig", candidate)
        assert result.score == 0.5  # default on parse error

    @pytest.mark.asyncio
    async def test_paper_judge_parse_error_returns_default(self):
        from app.agents.base import LLMResponse

        class BadJsonProvider:
            model_name = "judge"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="not json", model="judge", success=True)

        judge = PaperJudge(provider=BadJsonProvider())
        paper = Paper(paper_id="p1", title="Test", abstract="Test")
        result = await judge.evaluate_paper("query", paper)
        assert result.relevance_score == 0.5

    @pytest.mark.asyncio
    async def test_strong_judge_provider_failure(self):
        from app.agents.base import LLMResponse

        class FailingProvider:
            model_name = "judge"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="", model="judge", success=False, error="API down")

        judge = StrongJudge(provider=FailingProvider())
        candidate = CandidateQuery(query="test", proposer_model="qwen")
        result = await judge.evaluate_query_candidate("orig", candidate)
        assert result.score == 0.5
        assert "failed" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_paper_judge_provider_failure(self):
        from app.agents.base import LLMResponse

        class FailingProvider:
            model_name = "judge"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="", model="judge", success=False, error="API down")

        judge = PaperJudge(provider=FailingProvider())
        paper = Paper(paper_id="p1", title="Test", abstract="Test")
        result = await judge.evaluate_paper("query", paper)
        assert result.relevance_score == 0.5
