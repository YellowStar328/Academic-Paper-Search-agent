"""Multi-agent coordinator: orchestrates parallel query generation.

Manages the parallel execution of Qwen/DeepSeek/GLM agents,
collects their candidate queries, and handles failures gracefully
(single agent failure does not block others).
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from app.agents.base import LLMProvider
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.qwen import QwenAgent
from app.models.agent import AgentRun
from app.models.candidate import CandidateQuery
from app.models.query import SearchQuery


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

                run = AgentRun(
                    request_id=request_id,
                    model_name=agent.model_name,
                    query_text=query.original_query,
                    generated_candidates=[c.query for c in candidates],
                    latency_ms=latency_ms,
                    token_usage=len(candidates) * 100,  # estimated
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

        run = AgentRun(
            request_id=request_id,
            model_name=agent.model_name,
            query_text=query.original_query,
            generated_candidates=[c.query for c in candidates],
            latency_ms=latency_ms,
            token_usage=len(candidates) * 100,
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
