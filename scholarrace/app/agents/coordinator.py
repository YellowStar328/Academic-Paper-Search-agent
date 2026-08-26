"""Multi-agent coordinator: orchestrates parallel query generation.

Manages the parallel execution of Qwen/DeepSeek/GLM agents,
collects their candidate queries, and handles failures gracefully
(single agent failure does not block others).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from app.agents.base import LLMProvider
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.qwen import QwenAgent
from app.models.agent import AgentRun
from app.models.candidate import CandidateQuery, JudgeResult
from app.models.query import SearchQuery

logger = logging.getLogger(__name__)

# Cross-evaluation prompt used by agent-based candidate judging
# (replaces STRONG StrongJudge calls).
CROSS_EVAL_SYSTEM_PROMPT = """You are an academic search query evaluator.

Evaluate the candidate search query against the original query.
Score on:
- coverage: How well does it cover the original intent? (0.0-1.0)
- specificity: Is it specific enough to find relevant papers? (0.0-1.0)
- novelty: Does it bring a unique perspective? (0.0-1.0)
- score: Overall quality (0.0-1.0, weighted average of the above)

Return JSON: {"score": 0.0-1.0, "reasoning": "...", "coverage": 0.0-1.0, "specificity": 0.0-1.0, "novelty": 0.0-1.0}"""

CROSS_EVAL_USER_TEMPLATE = """Original query: {original}
Candidate query: {candidate}
Proposer model: {model}

Evaluate this candidate query. Return JSON with score, reasoning, coverage, specificity, novelty."""


class MultiAgentCoordinator:
    """Coordinates parallel query generation across multiple agents."""

    def __init__(
        self,
        qwen_agent: Optional[QwenAgent] = None,
        deepseek_agent: Optional[DeepSeekAgent] = None,
        glm_agent: Optional[GLMAgent] = None,
    ):
        self.agents: list[QwenAgent | DeepSeekAgent | GLMAgent] = []
        if qwen_agent is not None:
            self.agents.append(qwen_agent)
        else:
            self.agents.append(QwenAgent())
        if deepseek_agent is not None:
            self.agents.append(deepseek_agent)
        else:
            self.agents.append(DeepSeekAgent())
        if glm_agent is not None:
            self.agents.append(glm_agent)
        else:
            self.agents.append(GLMAgent())

    @property
    def model_names(self) -> list[str]:
        """Return the model names of all agents."""
        return [a.model_name for a in self.agents]

    async def generate_candidates(
        self,
        query: SearchQuery,
        request_id: str = "",
    ) -> tuple[list[CandidateQuery], list[AgentRun]]:
        """Run all agents in parallel and collect candidate queries.

        Returns:
            Tuple of (all_candidates, agent_runs) where agent_runs
            contains execution metadata for observability.
        """
        start_time = time.time()

        async def _run_agent(agent):
            """Run a single agent and capture metadata."""
            t0 = time.time()
            try:
                candidates = await agent.generate_queries(query)
                latency_ms = (time.time() - t0) * 1000
                token_usage = getattr(agent, "last_token_usage", 0) or 0

                run = AgentRun(
                    request_id=request_id,
                    model_name=agent.model_name,
                    query_text=query.original_query,
                    generated_candidates=[c.query for c in candidates],
                    latency_ms=latency_ms,
                    token_usage=token_usage,
                    success=True,
                )
                return candidates, run
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                run = AgentRun(
                    request_id=request_id,
                    model_name=agent.model_name,
                    query_text=query.original_query,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
                return [], run

        # Run all agents in parallel
        results = await asyncio.gather(
            *[_run_agent(a) for a in self.agents], return_exceptions=False
        )

        all_candidates: list[CandidateQuery] = []
        agent_runs: list[AgentRun] = []

        for candidates, run in results:
            all_candidates.extend(candidates)
            agent_runs.append(run)

        return all_candidates, agent_runs

    async def generate_candidates_single(
        self,
        query: SearchQuery,
        model_name: str,
        request_id: str = "",
    ) -> tuple[list[CandidateQuery], list[AgentRun]]:
        """Run a single agent (for single_model strategy comparison).

        Args:
            query: The structured query.
            model_name: Which model to use ("qwen", "deepseek", or "glm").
            request_id: Request ID for tracking.
        """
        agent = None
        for a in self.agents:
            if a.model_name == model_name:
                agent = a
                break

        if agent is None:
            raise ValueError(f"Unknown model: {model_name}")

        start_time = time.time()
        candidates = await agent.generate_queries(query)
        latency_ms = (time.time() - start_time) * 1000
        token_usage = getattr(agent, "last_token_usage", 0) or 0

        run = AgentRun(
            request_id=request_id,
            model_name=agent.model_name,
            query_text=query.original_query,
            generated_candidates=[c.query for c in candidates],
            latency_ms=latency_ms,
            token_usage=token_usage,
            success=True,
        )

        return candidates, [run]

    async def generate_candidates_random(
        self,
        query: SearchQuery,
        request_id: str = "",
    ) -> tuple[list[CandidateQuery], list[AgentRun]]:
        """Run a single randomly-chosen agent (for random_multi_agent strategy).

        This is NOT Thompson Sampling — it picks uniformly at random.
        """
        import random

        agent = random.choice(self.agents)
        return await self.generate_candidates_single(
            query, agent.model_name, request_id
        )

    async def evaluate_candidates_cross(
        self,
        original_query: str,
        candidates: list[CandidateQuery],
    ) -> list[JudgeResult]:
        """Cross-evaluate candidate queries using the 3 agents.

        Each agent evaluates ALL candidates (including its own) — this
        replaces the STRONG StrongJudge LLM call. Each candidate's final
        score is the average of the 3 agent scores (0 STRONG calls).

        Returns ``JudgeResult`` objects (one per candidate) with averaged
        scores.
        """
        if not candidates:
            return []

        async def _eval_one(
            evaluator_agent,
            cand: CandidateQuery,
        ) -> tuple[str, float, str, float, float, float]:
            """Returns (candidate_query, score, reasoning, coverage, specificity, novelty)."""
            prompt = CROSS_EVAL_USER_TEMPLATE.format(
                original=original_query,
                candidate=cand.query,
                model=cand.proposer_model,
            )
            try:
                resp = await evaluator_agent.provider.generate(
                    prompt=prompt,
                    temperature=0.2,
                    system_prompt=CROSS_EVAL_SYSTEM_PROMPT,
                    response_schema={"type": "json_object"},
                )
            except Exception as e:
                logger.warning(
                    "Cross-eval by %s failed for candidate '%s': %s",
                    evaluator_agent.model_name,
                    cand.query[:50],
                    e,
                )
                return cand.query, 0.5, f"{evaluator_agent.model_name} eval failed", 0.5, 0.5, 0.5

            if not resp.success:
                return cand.query, 0.5, f"{evaluator_agent.model_name} eval failed", 0.5, 0.5, 0.5

            try:
                data = json.loads(resp.content)
                score = float(data.get("score", 0.5))
                reasoning = data.get("reasoning", "")
                coverage = float(data.get("coverage", 0.5))
                specificity = float(data.get("specificity", 0.5))
                novelty = float(data.get("novelty", 0.5))
                return cand.query, score, reasoning, coverage, specificity, novelty
            except (json.JSONDecodeError, ValueError):
                return cand.query, 0.5, f"{evaluator_agent.model_name} parse error", 0.5, 0.5, 0.5

        # Build (candidate, evaluator) pairs: each agent evaluates every candidate
        tasks = []
        for agent in self.agents:
            for cand in candidates:
                tasks.append(_eval_one(agent, cand))

        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Aggregate: average the scores per candidate
        score_map: dict[str, list[float]] = {}
        reasoning_map: dict[str, list[str]] = {}
        coverage_map: dict[str, list[float]] = {}
        specificity_map: dict[str, list[float]] = {}
        novelty_map: dict[str, list[float]] = {}
        for (
            cand_q,
            score,
            reasoning,
            cov,
            spec,
            nov,
        ) in results:
            score_map.setdefault(cand_q, []).append(score)
            reasoning_map.setdefault(cand_q, []).append(reasoning)
            coverage_map.setdefault(cand_q, []).append(cov)
            specificity_map.setdefault(cand_q, []).append(spec)
            novelty_map.setdefault(cand_q, []).append(nov)

        judged: list[JudgeResult] = []
        for cand in candidates:
            scores = score_map.get(cand.query, [0.5])
            avg_score = sum(scores) / len(scores) if scores else 0.5
            avg_cov = sum(coverage_map.get(cand.query, [0.5])) / len(
                coverage_map.get(cand.query, [0.5])
            )
            avg_spec = sum(specificity_map.get(cand.query, [0.5])) / len(
                specificity_map.get(cand.query, [0.5])
            )
            avg_nov = sum(novelty_map.get(cand.query, [0.5])) / len(
                novelty_map.get(cand.query, [0.5])
            )
            judged.append(
                JudgeResult(
                    candidate=cand,
                    score=avg_score,
                    reasoning=" | ".join(reasoning_map.get(cand.query, [])),
                    coverage=avg_cov,
                    specificity=avg_spec,
                    novelty=avg_nov,
                )
            )
        return judged
