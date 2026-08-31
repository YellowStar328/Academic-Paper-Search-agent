"""Strong model judge — evaluates candidate queries and papers.

The strong model MUST be different from the generation models (no self-evaluation).
"""

from __future__ import annotations

import json
from typing import Optional

from app.agents.base import BaseOpenAIProvider, LLMProvider, create_strong_judge_provider
from app.agents.mock import MockLLMProvider
from app.config import get_settings
from app.models.candidate import AgentPaperReport, AgentReport, CandidateQuery, JudgeResult, PaperJudgeResult
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

# Batch paper judge: evaluate multiple papers in a single LLM call.
JUDGE_PAPER_BATCH_SYSTEM_PROMPT = """You are a strong academic search evaluator. You evaluate papers' relevance to a search query.

For each paper, score:
- relevance_score: How relevant is the paper to the query? (0.0-1.0)
- authority_score: How authoritative is this paper (venue, citations, authors)? (0.0-1.0)
- reasoning: Brief explanation (one sentence)
- key_findings: List of key findings (1-3 items)

You MUST return JSON with an "evaluations" array, one entry per paper, in the SAME ORDER as provided.
Each entry: {"index": 0, "relevance_score": 0.0-1.0, "authority_score": 0.0-1.0, "reasoning": "...", "key_findings": [...]}"""

JUDGE_PAPER_BATCH_TEMPLATE = """Query: {query}

Evaluate the following {n} papers for relevance to the query. Return a JSON object with an "evaluations" array (one entry per paper, in order):

{papers_json}"""

JUDGE_PAPER_TEMPLATE = """Query: {query}
Paper title: {title}
Paper abstract: {abstract}

Evaluate this paper's relevance. Return JSON:
{{"relevance_score": 0.0-1.0, "authority_score": 0.0-1.0, "reasoning": "...", "key_findings": [...]}}"""


# ---------------------------------------------------------------------------
# Strong-model review of agent procurement reports (worker mode)
# ---------------------------------------------------------------------------

REVIEW_REPORTS_SYSTEM_PROMPT = """You are the strong model coordinating three procurement agents (Qwen, DeepSeek, GLM).

Each agent independently:
1. Generated its own search keywords
2. Dispatched retrieval providers (arXiv, Semantic Scholar, etc.)
3. Judged each paper's abstract for relevance

Now you receive each agent's report. Your job is to make the FINAL decision:

For each paper mentioned across all reports, give a final score (0.0-1.0) based on:
- The agents' assessments (weighted, not just averaged)
- Cross-agent agreement (papers found + judged relevant by multiple agents get a boost)
- Your own authority assessment (venue, citations, authors)

You MUST return JSON with a "final_papers" array. Each entry:
{
  "paper_id": "string",
  "final_relevance_score": 0.0-1.0,
  "final_authority_score": 0.0-1.0,
  "final_reasoning": "one sentence",
  "endorsed_by": ["qwen", "deepseek", "glm"]  // which agents endorsed this paper
}

Only include papers with final_relevance_score >= 0.5. Sort by final_relevance_score descending.
"""

REVIEW_REPORTS_TEMPLATE = """Original query: {query}

Below are procurement reports from {n} agents.

{reports_json}

Review these reports and make your FINAL selection. Return JSON with a "final_papers" array (sorted by final_relevance_score descending, only score >= 0.5)."""


class StrongJudge:
    """Strong model judge for evaluating candidate queries."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "strong_judge"
        self.last_token_usage: int = 0

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
        self.last_token_usage += response.token_usage

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

    async def review_agent_reports(
        self,
        query: SearchQuery,
        reports: list[AgentReport],
    ) -> list[tuple[Paper, float, str, list[str]]]:
        """Strong model reviews all agent procurement reports and makes final call.

        Args:
            query: The structured user query.
            reports: AgentReport list from Qwen/DeepSeek/GLM.

        Returns:
            List of (paper, final_score, reasoning, endorsed_by) for
            papers with final_score >= 0.5, sorted descending.
        """
        import json as _json

        # Build a compact representation of all reports for the strong model.
        reports_payload = []
        paper_index: dict[str, Paper] = {}

        for rep in reports:
            entry = {
                "agent": rep.agent_model,
                "search_queries": rep.search_queries,
                "rationale": rep.rationale,
                "papers": [],
            }
            for pr in rep.paper_reports:
                pid = pr.paper.identity_key() or pr.paper.title.lower()
                paper_index[pid] = pr.paper
                entry["papers"].append(
                    {
                        "paper_id": pid,
                        "title": pr.paper.title,
                        "abstract": (pr.paper.abstract[:300] if pr.paper.abstract else "N/A"),
                        "agent_relevance_score": pr.agent_relevance_score,
                        "agent_reasoning": pr.agent_reasoning,
                        "agent_key_findings": pr.agent_key_findings,
                        "source": pr.paper.source,
                        "year": pr.paper.year,
                        "venue": pr.paper.venue,
                        "citation_count": pr.paper.citation_count,
                    }
                )
            reports_payload.append(entry)

        prompt = REVIEW_REPORTS_TEMPLATE.format(
            query=query.semantic_core or query.original_query,
            n=len(reports),
            reports_json=_json.dumps(reports_payload, ensure_ascii=False, indent=2),
        )

        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.2,
            system_prompt=REVIEW_REPORTS_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage += response.token_usage

        if not response.success:
            # Fallback: weighted average of agent scores
            return self._fallback_review(reports, paper_index)

        try:
            data = _json.loads(response.content)
            final_papers = data.get("final_papers", [])
            results: list[tuple[Paper, float, str, list[str]]] = []
            for fp in final_papers:
                pid = fp.get("paper_id", "")
                paper = paper_index.get(pid)
                if paper is None:
                    continue
                score = float(fp.get("final_relevance_score", 0.0))
                if score < 0.5:
                    continue
                reasoning = fp.get("final_reasoning", "")
                endorsed = fp.get("endorsed_by", [])
                results.append((paper, score, reasoning, endorsed))

            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except (_json.JSONDecodeError, ValueError, TypeError):
            return self._fallback_review(reports, paper_index)

    @staticmethod
    def _fallback_review(
        reports: list[AgentReport],
        paper_index: dict[str, Paper],
    ) -> list[tuple[Paper, float, str, list[str]]]:
        """Fallback: weighted average of agent scores when LLM parsing fails."""
        # Collect scores per paper
        paper_scores: dict[str, list[float]] = {}
        paper_endorsed: dict[str, list[str]] = {}
        for rep in reports:
            for pr in rep.paper_reports:
                pid = pr.paper.identity_key() or pr.paper.title.lower()
                paper_scores.setdefault(pid, []).append(pr.agent_relevance_score)
                paper_endorsed.setdefault(pid, []).append(rep.agent_model)

        results = []
        for pid, scores in paper_scores.items():
            paper = paper_index.get(pid)
            if paper is None:
                continue
            avg = sum(scores) / len(scores)
            # Boost for cross-agent agreement
            boost = 0.05 * (len(scores) - 1)
            final_score = min(1.0, avg + boost)
            if final_score < 0.5:
                continue
            endorsed = paper_endorsed.get(pid, [])
            results.append((paper, final_score, "fallback weighted avg", endorsed))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


class PaperJudge:
    """Strong model judge for evaluating individual paper relevance."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "paper_judge"
        self.last_token_usage: int = 0

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
        """Evaluate a single paper's relevance to the query.

        Kept as a fallback for single-paper evaluation. The batch method
        `evaluate_papers_batch` is preferred as it collapses many papers
        into a single LLM call, dramatically reducing token usage.
        """
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
        self.last_token_usage += response.token_usage

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

    async def _evaluate_paper_batch_single_call(
        self,
        query: str,
        papers: list[Paper],
    ) -> list[PaperJudgeResult]:
        """Evaluate a batch of papers in a single LLM call.

        This collapses `len(papers)` individual calls into 1 call,
        reducing STRONG token usage by ~90%.
        """
        if not papers:
            return []

        papers_payload = []
        for idx, p in enumerate(papers):
            papers_payload.append(
                {
                    "index": idx,
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": (p.abstract[:500] if p.abstract else "N/A"),
                }
            )

        prompt = JUDGE_PAPER_BATCH_TEMPLATE.format(
            query=query,
            n=len(papers),
            papers_json=json.dumps(papers_payload, ensure_ascii=False, indent=2),
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=JUDGE_PAPER_BATCH_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage += response.token_usage

        fallback_results = [
            PaperJudgeResult(
                paper_id=p.paper_id,
                relevance_score=0.5,
                reasoning="Batch judge failed",
            )
            for p in papers
        ]

        if not response.success:
            return fallback_results

        try:
            data = json.loads(response.content)
            evals = data.get("evaluations", [])
            if not isinstance(evals, list):
                evals = []

            results: list[PaperJudgeResult] = list(fallback_results)
            for entry in evals:
                try:
                    idx = int(entry.get("index", -1))
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(papers):
                    results[idx] = PaperJudgeResult(
                        paper_id=papers[idx].paper_id,
                        relevance_score=float(entry.get("relevance_score", 0.5)),
                        authority_score=float(entry.get("authority_score", 0.5)),
                        reasoning=entry.get("reasoning", ""),
                        key_findings=entry.get("key_findings", []),
                    )
            return results
        except (json.JSONDecodeError, ValueError):
            return fallback_results

    async def evaluate_papers_batch(
        self,
        query: str,
        papers: list[Paper],
    ) -> list[PaperJudgeResult]:
        """Evaluate multiple papers using batch LLM calls.

        Instead of one LLM call per paper (50 papers = 50 STRONG calls),
        this groups papers into batches and evaluates each batch in a
        single call. With ``paper_batch_size=10`` and 50 papers, this
        results in only 5 STRONG calls (~90% reduction).
        """
        import asyncio

        if not papers:
            return []

        settings = get_settings()
        batch_size = getattr(settings, "paper_batch_size", 10) or 10
        results: list[PaperJudgeResult] = []

        batches = [
            papers[i : i + batch_size]
            for i in range(0, len(papers), batch_size)
        ]
        tasks = [
            self._evaluate_paper_batch_single_call(query, batch)
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)
        for br in batch_results:
            results.extend(br)

        return results
