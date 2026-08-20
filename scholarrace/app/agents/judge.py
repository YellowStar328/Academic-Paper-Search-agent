"""Strong model judge — evaluates candidate queries and papers.

The strong model MUST be different from the generation models (no self-evaluation).
"""

from __future__ import annotations

import json
from typing import Optional

from app.agents.base import BaseOpenAIProvider, LLMProvider, create_strong_judge_provider
from app.agents.mock import MockLLMProvider
from app.config import get_settings
from app.models.candidate import CandidateQuery, JudgeResult, PaperJudgeResult
from app.models.paper import Paper
from app.models.query import SearchQuery

JUDGE_SYSTEM_PROMPT = """You are a strong academic search evaluator. You evaluate search queries and papers.

For query evaluation, score candidates on:
- coverage: How well does the query cover the original intent?
- specificity: Is the query specific enough to find relevant papers?
- novelty: Does the query bring a unique perspective?

For paper evaluation, score on:
- relevance_score: How relevant is the paper to the query?
- authority_score: How authoritative is this paper (venue, citations, authors)?

Return JSON as specified in each prompt."""

JUDGE_QUERY_TEMPLATE = """Original query: {original}
Candidate query: {candidate}
Proposer model: {model}

Evaluate this candidate query. Return JSON:
{{"score": 0.0-1.0, "reasoning": "...", "coverage": 0.0-1.0, "specificity": 0.0-1.0, "novelty": 0.0-1.0}}"""

JUDGE_PAPER_TEMPLATE = """Query: {query}
Paper title: {title}
Paper abstract: {abstract}

Evaluate this paper's relevance. Return JSON:
{{"relevance_score": 0.0-1.0, "authority_score": 0.0-1.0, "reasoning": "...", "key_findings": [...]}}"""


class StrongJudge:
    """Strong model judge for evaluating candidate queries."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "strong_judge"

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            settings = get_settings()
            if settings.is_test or not settings.strong_model_api_key:
                self._provider = MockLLMProvider(model_name="strong_judge")
            else:
                self._provider = create_strong_judge_provider()
        return self._provider

    async def evaluate_query_candidate(
        self,
        original_query: str,
        candidate: CandidateQuery,
    ) -> JudgeResult:
        """Evaluate a single candidate query."""
        prompt = JUDGE_QUERY_TEMPLATE.format(
            original=original_query,
            candidate=candidate.query,
            model=candidate.proposer_model,
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )

        if not response.success:
            return JudgeResult(candidate=candidate, score=0.5, reasoning="Judge failed")

        try:
            data = json.loads(response.content)
            return JudgeResult(
                candidate=candidate,
                score=float(data.get("score", 0.5)),
                reasoning=data.get("reasoning", ""),
                coverage=float(data.get("coverage", 0.5)),
                specificity=float(data.get("specificity", 0.5)),
                novelty=float(data.get("novelty", 0.5)),
            )
        except (json.JSONDecodeError, ValueError):
            return JudgeResult(candidate=candidate, score=0.5, reasoning="Parse error")

    async def evaluate_candidates_batch(
        self,
        original_query: str,
        candidates: list[CandidateQuery],
    ) -> list[JudgeResult]:
        """Evaluate multiple candidate queries in parallel."""
        import asyncio

        tasks = [
            self.evaluate_query_candidate(original_query, c) for c in candidates
        ]
        return await asyncio.gather(*tasks)

    async def select_top_candidates(
        self,
        original_query: str,
        candidates: list[CandidateQuery],
        top_k: int = 5,
    ) -> list[JudgeResult]:
        """Evaluate candidates and return the top-k by judge score.

        This is the main entry point for the pipeline: it evaluates all
        candidates in parallel and returns the best ones.
        """
        if not candidates:
            return []

        results = await self.evaluate_candidates_batch(original_query, candidates)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]


class PaperJudge:
    """Strong model judge for evaluating individual paper relevance."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "paper_judge"

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            settings = get_settings()
            if settings.is_test or not settings.strong_model_api_key:
                self._provider = MockLLMProvider(model_name="paper_judge")
            else:
                self._provider = create_strong_judge_provider()
        return self._provider

    async def evaluate_paper(
        self,
        query: str,
        paper: Paper,
    ) -> PaperJudgeResult:
        """Evaluate a single paper's relevance to the query."""
        prompt = JUDGE_PAPER_TEMPLATE.format(
            query=query,
            title=paper.title,
            abstract=paper.abstract[:500] if paper.abstract else "N/A",
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )

        if not response.success:
            return PaperJudgeResult(
                paper_id=paper.paper_id,
                relevance_score=0.5,
                reasoning="Judge failed",
            )

        try:
            data = json.loads(response.content)
            return PaperJudgeResult(
                paper_id=paper.paper_id,
                relevance_score=float(data.get("relevance_score", 0.5)),
                authority_score=float(data.get("authority_score", 0.5)),
                reasoning=data.get("reasoning", ""),
                key_findings=data.get("key_findings", []),
            )
        except (json.JSONDecodeError, ValueError):
            return PaperJudgeResult(
                paper_id=paper.paper_id,
                relevance_score=0.5,
                reasoning="Parse error",
            )

    async def evaluate_papers_batch(
        self,
        query: str,
        papers: list[Paper],
    ) -> list[PaperJudgeResult]:
        """Evaluate multiple papers. Uses batch_size from settings."""
        import asyncio

        settings = get_settings()
        batch_size = settings.thompson_batch_size
        results: list[PaperJudgeResult] = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            tasks = [self.evaluate_paper(query, p) for p in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results
