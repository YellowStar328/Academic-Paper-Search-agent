"""SearchPipeline — 14-stage end-to-end orchestration.

The SearchPipeline is the single entry point for running a search.  It
orchestrates all modules in order:

1.  Query understanding       — parse user query into SearchQuery
2.  Multi-agent generation    — Qwen/DeepSeek/GLM generate sub-queries
3.  Strong judge              — evaluate and select best candidate queries
4.  Thompson budget alloc     — distribute budget across models
5.  Multi-source retrieval    — search all providers
6.  Citation expansion        — expand top papers' citations/references
7.  Deduplication             — PaperIdentityResolver
8.  Embedding coarse ranking  — FakeEncoder → Top-K
9.  LLM paper judging         — PaperJudge scores relevance
10. Authority scoring          — multi-signal authority
11. Final ranking              — weighted score + MMR
12. Research graph             — nodes/edges/clusters/timeline
13. Result assembly            — final SearchResult
14. Metrics recording          — observability

All external calls are wrapped in try/except for failure isolation.  The
pipeline degrades gracefully: if one stage fails, it logs a warning and
continues with the best available data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from app.agents.base import LLMProvider
from app.agents.coordinator import MultiAgentCoordinator
from app.agents.judge import PaperJudge, StrongJudge
from app.bandit.thompson import ThompsonSamplingManager
from app.citation.expansion import CitationExpander
from app.config import get_settings, Settings
from app.embedding.encoder import EmbeddingEncoder, FakeEncoder
from app.embedding.reranker import EmbeddingReranker
from app.graph.research_graph import ResearchGraphBuilder
from app.models.candidate import CandidateQuery, JudgeResult, PaperJudgeResult
from app.models.paper import Paper, PaperList
from app.models.query import SearchQuery, UserQuery
from app.models.result import (
    PaperWithScores,
    ResearchGraph,
    SearchResult,
    SearchSummary,
)
from app.query.parser import QueryParser
from app.retrieval.base import SearchProvider
from app.retrieval.resolver import PaperIdentityResolver
from app.ranking.authority import AuthorityScorer
from app.ranking.diversity import MMRSelector
from app.ranking.final_ranker import FinalRanker
from app.utils.observability import MetricsTracker

logger = logging.getLogger(__name__)


class SearchPipeline:
    """14-stage search pipeline orchestrator.

    Parameters
    ----------
    query_parser
        Parser for user query → SearchQuery.
    coordinator
        Multi-agent coordinator for parallel query generation.
    strong_judge
        Strong model judge for candidate query evaluation.
    providers
        List of search source providers.
    citation_expander
        Citation expansion module.
    embedding_encoder
        Embedding encoder for coarse ranking.
    paper_judge
        Paper-level judge for fine ranking.
    authority_scorer
        Authority scoring module.
    final_ranker
        Final ranking module.
    graph_builder
        Research graph builder.
    thompson_manager
        Thompson sampling manager for budget allocation.
    metrics
        Metrics tracker for observability.
    """

    def __init__(
        self,
        query_parser: QueryParser,
        coordinator: MultiAgentCoordinator,
        strong_judge: StrongJudge,
        providers: list[SearchProvider],
        citation_expander: CitationExpander,
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        paper_judge: Optional[PaperJudge] = None,
        authority_scorer: Optional[AuthorityScorer] = None,
        final_ranker: Optional[FinalRanker] = None,
        graph_builder: Optional[ResearchGraphBuilder] = None,
        thompson_manager: Optional[ThompsonSamplingManager] = None,
        metrics: Optional[MetricsTracker] = None,
        settings: Optional[Settings] = None,
    ):
        self._parser = query_parser
        self._coordinator = coordinator
        self._judge = strong_judge
        self._providers = providers
        self._citation_expander = citation_expander
        self._encoder = embedding_encoder or FakeEncoder()
        self._reranker = EmbeddingReranker(encoder=self._encoder)
        self._paper_judge = paper_judge
        self._authority = authority_scorer or AuthorityScorer()
        self._final_ranker = final_ranker or FinalRanker(
            authority_scorer=self._authority, encoder=self._encoder
        )
        self._graph_builder = graph_builder or ResearchGraphBuilder(
            encoder=self._encoder
        )
        self._thompson = thompson_manager
        self._metrics = metrics or MetricsTracker()
        self._settings = settings or get_settings()

    async def run(self, user_query: UserQuery) -> SearchResult:
        """Execute the full 14-stage pipeline and return SearchResult."""
        request_id = str(uuid4())
        start_time = time.time()

        # Initialize metrics tracker with query
        self._metrics = MetricsTracker(query=user_query.query)
        request_id = self._metrics.request_id

        logger.info(
            f"[{request_id}] Pipeline started: '{user_query.query[:100]}'"
        )

        # Stage 1: Query understanding
        search_query = await self._stage1_query_understanding(
            user_query, request_id
        )

        # Stage 2: Multi-agent generation
        candidates = await self._stage2_multi_agent_generation(
            search_query, request_id
        )

        # Stage 3: Strong judge
        judged = await self._stage3_strong_judge(
            candidates, search_query, request_id
        )

        # Stage 4: Thompson budget allocation
        budget = await self._stage4_thompson_budget(
            search_query, request_id
        )

        # Stage 5: Multi-source retrieval
        papers = await self._stage5_retrieval(
            judged, search_query, budget, request_id
        )

        # Stage 6: Citation expansion
        papers = await self._stage6_citation_expansion(
            papers, search_query, request_id
        )

        # Stage 7: Deduplication
        papers = await self._stage7_dedup(papers, request_id)

        # Stage 8: Embedding coarse ranking
        papers = await self._stage8_embedding_ranking(
            papers, search_query, request_id
        )

        # Stage 9: LLM paper judging
        judge_results = await self._stage9_paper_judging(
            papers, search_query, request_id
        )

        # Stage 10: Authority scoring (done inside FinalRanker)
        # Stage 11: Final ranking + MMR
        ranked = await self._stage10_11_final_ranking(
            papers, judge_results, search_query, request_id
        )

        # Stage 12: Research graph
        graph = await self._stage12_research_graph(
            ranked, search_query, request_id
        )

        # Stage 13: Result assembly
        result = self._stage13_assembly(
            ranked, graph, search_query, request_id, start_time
        )

        # Stage 14: Metrics recording
        self._stage14_metrics(result, request_id, start_time)

        logger.info(
            f"[{request_id}] Pipeline completed in {result.latency_ms:.0f}ms, "
            f"{len(result.papers)} papers returned"
        )

        return result

    # -----------------------------------------------------------------------
    # Stage implementations
    # -----------------------------------------------------------------------

    async def _stage1_query_understanding(
        self, user_query: UserQuery, request_id: str
    ) -> SearchQuery:
        """Parse user query into structured SearchQuery."""
        try:
            search_query = await self._parser.parse(
                user_query.query, user_query.options
            )
            self._metrics.record_llm_call(self._parser.last_token_usage)
            self._metrics.record_model_used(self._parser.model_name)
            logger.info(
                f"[{request_id}] Stage 1: domain={search_query.domain}, "
                f"intent={search_query.intent.value}"
            )
            return search_query
        except Exception as e:
            logger.error(f"[{request_id}] Stage 1 failed: {e}")
            # Fallback: simple query
            return SearchQuery(
                original_query=user_query.query,
                semantic_core=user_query.query,
                domain="general",
                options=user_query.options,
            )

    async def _stage2_multi_agent_generation(
        self, search_query: SearchQuery, request_id: str
    ) -> list[CandidateQuery]:
        """Generate candidate sub-queries via multi-agent."""
        try:
            candidates, agent_runs = await self._coordinator.generate_candidates(
                search_query
            )
            for run in agent_runs:
                self._metrics.record_model_used(run.model_name)
                self._metrics.record_llm_call(run.token_usage)
            logger.info(
                f"[{request_id}] Stage 2: {len(candidates)} candidates generated"
            )
            return candidates
        except Exception as e:
            logger.error(f"[{request_id}] Stage 2 failed: {e}")
            return []

    async def _stage3_strong_judge(
        self,
        candidates: list[CandidateQuery],
        search_query: SearchQuery,
        request_id: str,
    ) -> list[JudgeResult]:
        """Evaluate candidate queries via strong judge."""
        try:
            if not candidates:
                return []
            judged = await self._judge.evaluate_candidates_batch(
                search_query.original_query, candidates
            )
            self._metrics.record_llm_call(self._judge.last_token_usage)
            self._metrics.record_model_used(self._judge.model_name)
            logger.info(
                f"[{request_id}] Stage 3: {len(judged)} candidates judged"
            )
            return judged
        except Exception as e:
            logger.error(f"[{request_id}] Stage 3 failed: {e}")
            # Fallback: all candidates get default score
            return [
                JudgeResult(candidate=c, score=0.5) for c in candidates
            ]

    async def _stage4_thompson_budget(
        self, search_query: SearchQuery, request_id: str
    ) -> dict[str, int]:
        """Allocate budget via Thompson sampling."""
        try:
            if self._thompson is None:
                return {}
            model_names = self._coordinator.model_names
            budget = self._thompson.allocate_budget(
                model_names,
                search_query.domain,
                search_query.query_type,
                self._settings.thompson_total_budget,
            )
            self._metrics.record_thompson_allocation(budget)
            logger.info(
                f"[{request_id}] Stage 4: budget={budget}"
            )
            return budget
        except Exception as e:
            logger.error(f"[{request_id}] Stage 4 failed: {e}")
            return {}

    async def _stage5_retrieval(
        self,
        judged: list[JudgeResult],
        search_query: SearchQuery,
        budget: dict[str, int],
        request_id: str,
    ) -> list[Paper]:
        """Search all providers."""
        try:
            # Build search queries: always include original query + judged candidates
            queries = [search_query.original_query]
            if judged:
                # Add top judged candidates (sorted by score descending)
                candidate_queries = [
                    jr.candidate.query
                    for jr in sorted(judged, key=lambda j: j.score, reverse=True)
                ]
                for q in candidate_queries:
                    if q and q not in queries:
                        queries.append(q)

            # Search all providers in parallel
            max_per_source = self._settings.thompson_total_budget // max(
                len(queries), 1
            )

            all_papers: list[Paper] = []
            tasks = []
            for provider in self._providers:
                for q in queries[:5]:  # limit to top 5 queries
                    tasks.append(
                        self._search_provider(
                            provider, q, max_per_source, request_id
                        )
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"[{request_id}] Provider error: {r}")
                    continue
                all_papers.extend(r)

            self._metrics.record_papers_collected(len(all_papers))
            logger.info(
                f"[{request_id}] Stage 5: {len(all_papers)} papers retrieved"
            )
            return all_papers
        except Exception as e:
            logger.error(f"[{request_id}] Stage 5 failed: {e}")
            return []

    async def _search_provider(
        self,
        provider: SearchProvider,
        query: str,
        max_results: int,
        request_id: str,
    ) -> list[Paper]:
        """Search a single provider (failure-isolated)."""
        try:
            source_name = getattr(provider, "source_name", "unknown")
            paper_list = await provider.search(query, max_results=max_results)
            self._metrics.record_source_used(source_name)
            return paper_list.papers
        except Exception as e:
            logger.warning(
                f"[{request_id}] Provider "
                f"{getattr(provider, 'source_name', 'unknown')} failed: {e}"
            )
            return []

    async def _stage6_citation_expansion(
        self,
        papers: list[Paper],
        search_query: SearchQuery,
        request_id: str,
    ) -> list[Paper]:
        """Expand citations for top papers."""
        try:
            if not papers or not search_query.options.enable_citation_expansion:
                return papers

            expanded = await self._citation_expander.expand(papers)
            logger.info(
                f"[{request_id}] Stage 6: {len(papers)} → {len(expanded)} "
                f"after expansion"
            )
            return expanded
        except Exception as e:
            logger.error(f"[{request_id}] Stage 6 failed: {e}")
            return papers

    async def _stage7_dedup(
        self, papers: list[Paper], request_id: str
    ) -> list[Paper]:
        """Deduplicate papers."""
        try:
            resolver = PaperIdentityResolver()
            deduped = resolver.resolve(papers)
            self._metrics.record_papers_after_dedup(len(deduped))
            logger.info(
                f"[{request_id}] Stage 7: {len(papers)} → {len(deduped)} "
                f"after dedup"
            )
            return deduped
        except Exception as e:
            logger.error(f"[{request_id}] Stage 7 failed: {e}")
            return papers

    async def _stage8_embedding_ranking(
        self,
        papers: list[Paper],
        search_query: SearchQuery,
        request_id: str,
    ) -> list[Paper]:
        """Coarse ranking via embeddings."""
        try:
            if not papers or not search_query.options.enable_embedding_rerank:
                return papers[: self._settings.embedding_top_k]

            reranked = self._reranker.rerank_papers(
                search_query.semantic_core, papers
            )
            self._metrics.record_papers_after_rerank(len(reranked))
            logger.info(
                f"[{request_id}] Stage 8: {len(papers)} → {len(reranked)} "
                f"after embedding rank"
            )
            return reranked
        except Exception as e:
            logger.error(f"[{request_id}] Stage 8 failed: {e}")
            return papers[: self._settings.embedding_top_k]

    async def _stage9_paper_judging(
        self,
        papers: list[Paper],
        search_query: SearchQuery,
        request_id: str,
    ) -> list[PaperJudgeResult]:
        """LLM paper judging."""
        try:
            if self._paper_judge is None or not papers:
                return []

            results = await self._paper_judge.evaluate_papers_batch(
                search_query.semantic_core, papers
            )
            self._metrics.record_llm_call(self._paper_judge.last_token_usage)
            self._metrics.record_model_used(self._paper_judge.model_name)
            logger.info(
                f"[{request_id}] Stage 9: {len(results)} papers judged"
            )
            return results
        except Exception as e:
            logger.error(f"[{request_id}] Stage 9 failed: {e}")
            return []

    async def _stage10_11_final_ranking(
        self,
        papers: list[Paper],
        judge_results: list[PaperJudgeResult],
        search_query: SearchQuery,
        request_id: str,
    ) -> list[PaperWithScores]:
        """Authority + final ranking with MMR."""
        try:
            ranked = self._final_ranker.rank(
                papers,
                search_query.semantic_core,
                judge_results=judge_results if judge_results else None,
                top_k=search_query.options.top_k,
            )
            self._metrics.record_papers_final(len(ranked))
            logger.info(
                f"[{request_id}] Stage 10-11: {len(ranked)} papers ranked"
            )
            return ranked
        except Exception as e:
            logger.error(f"[{request_id}] Stage 10-11 failed: {e}")
            # Fallback: simple ranking by citation count
            sorted_papers = sorted(
                papers,
                key=lambda p: p.citation_count,
                reverse=True,
            )[: search_query.options.top_k]
            return [
                PaperWithScores(
                    paper=p,
                    relevance_score=0.5,
                    authority_score=0.5,
                    recency_score=0.5,
                    citation_score=0.5,
                    diversity_score=0.5,
                    redundancy_score=0.5,
                    final_score=0.5,
                )
                for p in sorted_papers
            ]

    async def _stage12_research_graph(
        self,
        ranked: list[PaperWithScores],
        search_query: SearchQuery,
        request_id: str,
    ) -> ResearchGraph:
        """Build research graph."""
        try:
            papers = [pws.paper for pws in ranked]
            graph = self._graph_builder.build(papers, ranked)
            logger.info(
                f"[{request_id}] Stage 12: graph with {len(graph.nodes)} "
                f"nodes, {len(graph.clusters)} clusters"
            )
            return graph
        except Exception as e:
            logger.error(f"[{request_id}] Stage 12 failed: {e}")
            return ResearchGraph()

    def _stage13_assembly(
        self,
        ranked: list[PaperWithScores],
        graph: ResearchGraph,
        search_query: SearchQuery,
        request_id: str,
        start_time: float,
    ) -> SearchResult:
        """Assemble final SearchResult."""
        latency_ms = (time.time() - start_time) * 1000

        summary = SearchSummary(
            total_papers=len(ranked),
            query=search_query.original_query,
            domain=search_query.domain,
            intent=search_query.intent.value,
            top_paper_title=ranked[0].paper.title if ranked else None,
            clusters_count=len(graph.clusters),
        )

        result = SearchResult(
            request_id=request_id,
            query=search_query.original_query,
            papers=ranked,
            graph=graph,
            summary=summary,
            latency_ms=latency_ms,
            metrics=self._metrics.get_metrics(),
        )

        return result

    def _stage14_metrics(
        self, result: SearchResult, request_id: str, start_time: float
    ) -> None:
        """Record final metrics."""
        m = self._metrics.get_metrics()
        logger.info(
            f"[{request_id}] Metrics: latency={m.total_latency_ms:.0f}ms, "
            f"papers={m.papers_final}, "
            f"llm_calls={m.llm_calls}, "
            f"tokens={m.token_usage}"
        )
