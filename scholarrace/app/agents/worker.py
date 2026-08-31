"""SearchWorker — turns each LLM agent into a self-contained sub-agent.

A worker independently performs the full procurement loop:
1. Generate search keywords (reuses the agent's existing generate_queries)
2. Dispatch retrieval providers (arXiv / S2 / OpenAlex ...) in parallel
3. Judge each paper's abstract for relevance (using the agent's own LLM)
4. Write an AgentReport to the strong model

The strong model then reviews all three reports and makes the final call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional, Sequence

from app.agents.base import LLMProvider
from app.models.candidate import AgentPaperReport, AgentReport, CandidateQuery
from app.models.paper import Paper
from app.models.query import SearchQuery
from app.retrieval.arxiv import ArxivProvider
from app.retrieval.base import SearchProvider

logger = logging.getLogger(__name__)

# Worker judges paper abstracts with its OWN LLM (no STRONG call).
WORKER_JUDGE_SYSTEM_PROMPT = """You are an academic paper relevance evaluator.

Given a search query and a paper's title + abstract, judge how relevant the paper is.

Score:
- relevance_score: How relevant is the paper to the query? (0.0-1.0)
- reasoning: One-sentence explanation
- key_findings: 1-3 key findings (short phrases)

Return JSON: {"relevance_score": 0.0-1.0, "reasoning": "...", "key_findings": ["...", "..."]}"""

WORKER_JUDGE_TEMPLATE = """Query: {query}

Paper title: {title}
Paper abstract: {abstract}

Judge this paper's relevance to the query. Return JSON with relevance_score, reasoning, key_findings."""

# Batch version: judge multiple papers in one call (saves tokens).
WORKER_JUDGE_BATCH_SYSTEM_PROMPT = """You are an academic paper relevance evaluator.

For each paper, score:
- relevance_score: How relevant is the paper to the query? (0.0-1.0)
- reasoning: Brief explanation (one sentence)
- key_findings: 1-3 key findings (short phrases)

You MUST return JSON with an "evaluations" array, one entry per paper, in the SAME ORDER as provided.
Each entry: {"index": 0, "relevance_score": 0.0-1.0, "reasoning": "...", "key_findings": [...]}"""

WORKER_JUDGE_BATCH_TEMPLATE = """Query: {query}

Evaluate the following {n} papers for relevance to the query. Return a JSON object with an "evaluations" array (one entry per paper, in order):

{papers_json}"""


class SearchWorker:
    """Mixin that gives an LLM agent procurement + paper-judging ability.

    Any agent (Qwen/DeepSeek/GLM) can inherit this to gain the
    `search_and_judge` method. The agent provides:
    - `provider`: an LLMProvider for judging abstracts
    - `model_name`: str
    - `generate_queries(query) -> list[CandidateQuery]`: keyword generation
    """

    # Providers are injected at construction time so the worker can
    # dispatch retrieval without going through the pipeline.
    _providers: list[SearchProvider] = []
    _max_per_source: int = 10

    def set_providers(self, providers: Sequence[SearchProvider], max_per_source: int = 10) -> None:
        """Inject retrieval providers for this worker to dispatch."""
        self._providers = list(providers)
        self._max_per_source = max_per_source

    async def search_and_judge(
        self,
        query: SearchQuery,
        candidates: list[CandidateQuery],
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> AgentReport:
        """Full procurement loop: search → judge → report.

        Steps:
        1. Use the agent's candidate queries as search terms.
        2. Dispatch all injected providers × queries in parallel.
        3. Dedup papers.
        4. Judge each paper's abstract with the agent's own LLM (batch).
        5. Return an AgentReport to the strong model.
        """
        start = time.time()
        provider: LLMProvider = self.provider  # type: ignore[attr-defined]
        model_name: str = self.model_name  # type: ignore[attr-defined]
        token_usage = 0

        # 1. Collect search queries from this agent's candidates
        search_queries = [c.query for c in candidates if c.query]
        if not search_queries:
            search_queries = [query.original_query]

        # 2. Dispatch providers × queries in parallel
        all_papers: list[Paper] = []
        tasks = []
        for prov in self._providers:
            for q in search_queries[:3]:  # cap at 3 queries per agent
                tasks.append(self._search_one(prov, q, year_start, year_end))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("%s provider error: %s", model_name, r)
                continue
            all_papers.extend(r)

        # 3. Dedup by identity key
        seen: set[str] = set()
        unique: list[Paper] = []
        for p in all_papers:
            k = p.identity_key() or p.title.lower()
            if k and k not in seen:
                seen.add(k)
                unique.append(p)

        # 4. Judge papers with the agent's own LLM (batch)
        paper_reports = await self._judge_papers_batch(
            provider, query.semantic_core or query.original_query, unique
        )
        token_usage += getattr(self, "last_token_usage", 0)  # type: ignore[attr-defined]

        latency_ms = (time.time() - start) * 1000
        rationale = self._build_rationale(candidates)

        return AgentReport(
            agent_model=model_name,
            search_queries=search_queries[:3],
            rationale=rationale,
            paper_reports=paper_reports,
            token_usage=token_usage,
            latency_ms=latency_ms,
            success=True,
        )

    async def _search_one(
        self,
        prov: SearchProvider,
        search_query: str,
        year_start: Optional[int],
        year_end: Optional[int],
    ) -> list[Paper]:
        """Search a single provider (failure-isolated)."""
        try:
            is_arxiv = isinstance(prov, ArxivProvider)
            q = search_query
            # For arXiv, use the raw query (it may contain field syntax).
            # For others, strip arXiv field prefixes if present.
            if not is_arxiv and ('all:"' in q or 'ti:"' in q or 'abs:"' in q):
                # Extract keywords from arXiv syntax for generic providers
                import re

                keywords = re.findall(r'"([^"]+)"', q)
                q = " ".join(keywords) if keywords else q

            paper_list = await prov.search(
                q,
                max_results=self._max_per_source,
                year_start=year_start,
                year_end=year_end,
            )
            return paper_list.papers
        except Exception as e:
            logger.warning("%s search failed on %s: %s", self.model_name, getattr(prov, "source_name", "?"), e)  # type: ignore[attr-defined]
            return []

    async def _judge_papers_batch(
        self,
        provider: LLMProvider,
        query: str,
        papers: list[Paper],
    ) -> list[AgentPaperReport]:
        """Judge papers in batches using the agent's own LLM."""
        if not papers:
            return []

        batch_size = 10
        batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
        tasks = [self._judge_batch_single(provider, query, batch, papers) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        reports: list[AgentPaperReport] = []
        for br in batch_results:
            reports.extend(br)
        return reports

    async def _judge_batch_single(
        self,
        provider: LLMProvider,
        query: str,
        batch: list[Paper],
        all_papers: list[Paper],
    ) -> list[AgentPaperReport]:
        """Judge one batch in a single LLM call."""
        papers_payload = []
        for idx, p in enumerate(batch):
            papers_payload.append(
                {
                    "index": idx,
                    "title": p.title,
                    "abstract": (p.abstract[:500] if p.abstract else "N/A"),
                }
            )

        prompt = WORKER_JUDGE_BATCH_TEMPLATE.format(
            query=query,
            n=len(batch),
            papers_json=json.dumps(papers_payload, ensure_ascii=False, indent=2),
        )
        response = await provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=WORKER_JUDGE_BATCH_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        # Accumulate token usage on the agent
        self.last_token_usage = getattr(self, "last_token_usage", 0) + response.token_usage  # type: ignore[attr-defined]

        fallback = [
            AgentPaperReport(
                paper=p,
                agent_relevance_score=0.5,
                agent_reasoning="judge failed",
                source=p.source,
            )
            for p in batch
        ]

        if not response.success:
            return fallback

        try:
            data = json.loads(response.content)
            evals = data.get("evaluations", [])
            if not isinstance(evals, list):
                evals = []

            results = list(fallback)
            for entry in evals:
                try:
                    idx = int(entry.get("index", -1))
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(batch):
                    results[idx] = AgentPaperReport(
                        paper=batch[idx],
                        agent_relevance_score=float(entry.get("relevance_score", 0.5)),
                        agent_reasoning=entry.get("reasoning", ""),
                        agent_key_findings=entry.get("key_findings", []),
                        source=batch[idx].source,
                    )
            return results
        except (json.JSONDecodeError, ValueError):
            return fallback

    def _build_rationale(self, candidates: list[CandidateQuery]) -> str:
        """Build a short rationale string from the agent's candidate queries."""
        if not candidates:
            return ""
        parts = [f"[{c.query}] {c.rationale}" for c in candidates[:3] if c.rationale]
        return " | ".join(parts)
