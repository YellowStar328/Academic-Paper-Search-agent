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
from app.models.candidate import (
    AgentPaperReport,
    AgentReport,
    CandidateQuery,
    JudgeResult,
    PaperJudgeResult,
)
from app.models.paper import Paper, PaperList
from app.models.query import SearchQuery, UserQuery
from app.models.result import (
    PaperWithScores,
    ResearchGraph,
    SearchResult,
    SearchSummary,
)
from app.query.parser import QueryParser
from app.query.refiner import QueryRefiner
from app.retrieval.arxiv import ArxivProvider
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
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        worker_mode: bool = False,
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
        self._query_refiner = QueryRefiner()
        # Optional publication year range for filtering search results.
        # When set, all provider search calls are restricted to this range.
        self._year_start = year_start
        self._year_end = year_end
        # Worker mode: each agent (Qwen/DeepSeek/GLM) independently
        # performs search + judge + report, then the strong model
        # reviews all reports and makes the final selection.
        self._worker_mode = worker_mode

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

        # Dynamically extract year range from parsed hard_filters so that
        # provider search calls and citation expansion respect the user's
        # year constraints (e.g. "2025-2026").
        hf = search_query.hard_filters
        if hf and (hf.year_start is not None or hf.year_end is not None):
            self._year_start = hf.year_start
            self._year_end = hf.year_end
            logger.info(
                f"[{request_id}] Year filter applied: "
                f"{self._year_start}-{self._year_end}"
            )

        # Stage 2: Multi-agent generation
        candidates = await self._stage2_multi_agent_generation(
            search_query, request_id
        )

        if self._worker_mode:
            # === Worker mode: agents do their own search + judge + report ===
            return await self._run_worker_mode(
                search_query, candidates, user_query, request_id, start_time
            )

        # === Iterative search loop ===
        # Stage 2-11 are repeated up to max_search_iterations.
        # After each round, if the number of highly-relevant papers
        # is below min_papers_threshold, the query is refined and
        # another search round is performed.
        max_iters = getattr(self._settings, "max_search_iterations", 3)
        min_papers = getattr(self._settings, "min_papers_threshold", 10)
        rel_floor = getattr(self._settings, "relevance_floor", 0.30)

        all_ranked: list[PaperWithScores] = []
        all_judge_results: list[PaperJudgeResult] = []
        best_round_ranked: list[PaperWithScores] = []
        best_round_judges: list[PaperJudgeResult] = []
        best_round_score: float = -1.0
        iterations_done = 0
        seen_paper_ids: set[str] = set()

        for iteration in range(max_iters):
            iter_id = f"{request_id}-iter{iteration}"
            logger.info(
                f"[{iter_id}] Search iteration {iteration + 1}/{max_iters}"
            )

            # Stage 3: Strong judge
            judged = await self._stage3_strong_judge(
                candidates, search_query, iter_id
            )

            # Stage 4: Thompson budget allocation
            budget = await self._stage4_thompson_budget(
                search_query, iter_id
            )

            # Stage 5: Multi-source retrieval
            papers = await self._stage5_retrieval(
                judged, search_query, budget, iter_id
            )

            # Stage 6: Citation expansion
            papers = await self._stage6_citation_expansion(
                papers, search_query, iter_id
            )

            # Stage 7: Deduplication
            papers = await self._stage7_dedup(papers, iter_id)

            # Stage 7b: arXiv ID enrichment via CrossRef DOI lookup
            papers = await self._stage7b_arxiv_enrichment(papers, iter_id)

            # Stage 8: Embedding coarse ranking
            papers = await self._stage8_embedding_ranking(
                papers, search_query, iter_id
            )

            # Stage 9: LLM paper judging
            judge_results = await self._stage9_paper_judging(
                papers, search_query, iter_id
            )

            # Stage 10-11: Final ranking + MMR
            ranked = await self._stage10_11_final_ranking(
                papers, judge_results, search_query, iter_id
            )

            iterations_done = iteration + 1

            # Accumulate non-duplicate papers across iterations
            for pws in ranked:
                pid = pws.paper.paper_id or pws.paper.title
                if pid not in seen_paper_ids:
                    seen_paper_ids.add(pid)
                    all_ranked.append(pws)
            all_judge_results.extend(judge_results)

            # Evaluate round quality
            relevant_count = sum(
                1 for p in ranked if p.relevance_score >= rel_floor
            )
            round_score = (
                relevant_count
                + sum(p.relevance_score for p in ranked) * 0.1
            )
            if round_score > best_round_score:
                best_round_score = round_score
                best_round_ranked = ranked
                best_round_judges = judge_results

            logger.info(
                f"[{iter_id}] Round {iteration + 1}: "
                f"{len(ranked)} ranked, {relevant_count} relevant "
                f"(score={round_score:.2f})"
            )

            # Convergence check
            if relevant_count >= min_papers:
                logger.info(
                    f"[{iter_id}] Sufficient relevant papers "
                    f"({relevant_count} >= {min_papers}), stopping"
                )
                break

            if iteration < max_iters - 1:
                # --- Dynamic query refinement ---
                # Use LLM to analyze the gap and suggest new keywords
                new_query = await self._refine_query_for_next_round(
                    search_query, ranked, iter_id
                )
                if new_query:
                    search_query = new_query
                    # Re-generate candidates for the refined query
                    candidates = await self._stage2_multi_agent_generation(
                        search_query, iter_id
                    )
                else:
                    logger.info(
                        f"[{iter_id}] No query refinement suggested, "
                        f"stopping"
                    )
                    break

        # Use the best round's results, supplemented with any unique
        # papers from other rounds
        final_ranked = best_round_ranked.copy()
        best_ids = {
            pws.paper.paper_id or pws.paper.title
            for pws in best_round_ranked
        }
        for pws in all_ranked:
            pid = pws.paper.paper_id or pws.paper.title
            if pid not in best_ids:
                final_ranked.append(pws)
                best_ids.add(pid)
        # Re-rank the merged set
        if len(final_ranked) > len(best_round_ranked):
            final_ranked = self._final_ranker.rank(
                [pws.paper for pws in final_ranked],
                search_query.semantic_core,
                judge_results=all_judge_results if all_judge_results else None,
                top_k=search_query.options.top_k,
            )

        ranked = final_ranked
        judge_results = best_round_judges

        # --- Thompson feedback: compute reward and update state ---
        await self._update_thompson_reward(
            search_query, ranked, request_id
        )

        # Stage 12: Research graph
        graph = await self._stage12_research_graph(
            ranked, search_query, request_id
        )

        # Stage 13: Result assembly
        result = await self._stage13_assembly(
            ranked, graph, search_query, request_id, start_time
        )
        result.summary.search_iterations = iterations_done

        # Stage 14: Metrics recording
        self._stage14_metrics(result, request_id, start_time)

        logger.info(
            f"[{request_id}] Pipeline completed in {result.latency_ms:.0f}ms, "
            f"{len(result.papers)} papers returned"
        )

        # Reset dynamic year filters so subsequent queries on the same
        # pipeline instance don't inherit stale constraints.
        self._year_start = None
        self._year_end = None

        return result

    # -----------------------------------------------------------------------
    # Stage implementations
    # -----------------------------------------------------------------------

    async def _stage1_query_understanding(
        self, user_query: UserQuery, request_id: str
    ) -> SearchQuery:
        """Parse user query into structured SearchQuery.

        Uses LLM-based parsing by default for richer semantic decomposition
        (intent detection, keyword extraction, sub-query generation).
        Falls back to rule-based parsing on LLM failure.
        """
        try:
            use_llm = not self._settings.is_test
            search_query = await self._parser.parse(
                user_query.query, user_query.options, use_llm=use_llm
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
        """Evaluate candidate queries.

        Token-saving strategy: by default uses agent cross-evaluation
        (0 STRONG calls) — each of the 3 agents (Qwen/DeepSeek/GLM)
        evaluates all candidates, and scores are averaged.

        Set ``settings.use_strong_judge=True`` to fall back to the
        original STRONG-based evaluation.
        """
        try:
            if not candidates:
                return []

            # Default: cross-evaluation via agents (0 STRONG calls)
            if not getattr(self._settings, "use_strong_judge", False):
                judged = await self._coordinator.evaluate_candidates_cross(
                    search_query.original_query, candidates
                )
                logger.info(
                    f"[{request_id}] Stage 3 (cross-eval): "
                    f"{len(judged)} candidates evaluated by agents"
                )
            else:
                judged = await self._judge.evaluate_candidates_batch(
                    search_query.original_query, candidates
                )
                self._metrics.record_llm_call(self._judge.last_token_usage)
                self._metrics.record_model_used(self._judge.model_name)
                logger.info(
                    f"[{request_id}] Stage 3 (STRONG): "
                    f"{len(judged)} candidates judged"
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
        """Search all providers using LLM-refined queries.

        Uses QueryRefiner (strong LLM) to convert natural-language queries into
        provider-specific search syntax:
        - arXiv: field-specific syntax (e.g. all:"data quality" AND all:"pretraining")
        - Semantic Scholar: concise keyword phrases
        - Other providers: concise keyword phrases
        """
        try:
            # Collect unique raw queries: original + judged candidates
            raw_queries = [search_query.original_query]
            sorted_judged = sorted(judged, key=lambda j: j.score, reverse=True)
            if sorted_judged:
                candidate_queries = [jr.candidate.query for jr in sorted_judged]
                for q in candidate_queries:
                    if q and q not in raw_queries:
                        raw_queries.append(q)

            queries_to_search = raw_queries[:5]  # limit to top 5 queries

            # Refine ALL queries via strong LLM into provider-specific syntax
            refined_results: list[dict[str, str | list[str]]] = []
            try:
                refined_results = await self._query_refiner.refine_candidates(
                    search_query,
                    [jr.candidate for jr in sorted_judged],
                )
                # Ensure we have one refined dict per raw query
                while len(refined_results) < len(queries_to_search):
                    refined_results.append({
                        "arxiv_query": queries_to_search[len(refined_results)],
                        "s2_query": queries_to_search[len(refined_results)],
                        "keywords": [],
                    })
                refined_results = refined_results[: len(queries_to_search)]
            except Exception as e:
                logger.warning(
                    f"[{request_id}] QueryRefiner failed, using raw queries: {e}"
                )
                refined_results = [
                    {"arxiv_query": q, "s2_query": q, "keywords": []}
                    for q in queries_to_search
                ]

            # Each provider × query call should retrieve up to max_per_source
            # papers. Budget is per-provider-per-query, NOT divided by the
            # number of queries (that would starve each call).
            #
            # thompson_total_budget is the *total* papers we want across all
            # sources. We divide by provider count so each provider fetches
            # its fair share. Dedup later handles overlaps.
            max_per_source = max(
                self._settings.thompson_total_budget // max(len(self._providers), 1),
                10,  # floor: at least 10 per call
            )

            all_papers: list[Paper] = []
            tasks = []
            for provider in self._providers:
                is_arxiv = isinstance(provider, ArxivProvider)
                # Pick the right refined query field for this provider
                query_field = "arxiv_query" if is_arxiv else "s2_query"
                provider_queries = [r[query_field] for r in refined_results]

                for q in provider_queries:
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
        """Search a single provider (failure-isolated).

        Passes the pipeline-level year_start/year_end (if any) to the
        provider's search method for publication-year filtering.
        """
        try:
            source_name = getattr(provider, "source_name", "unknown")
            kwargs: dict = {"max_results": max_results}
            if self._year_start is not None:
                kwargs["year_start"] = self._year_start
            if self._year_end is not None:
                kwargs["year_end"] = self._year_end
            paper_list = await provider.search(query, **kwargs)
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

            # Pass current year range to citation expander so expanded
            # papers also respect the user's year constraints.
            self._citation_expander.set_year_range(
                self._year_start, self._year_end
            )
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

    async def _stage7b_arxiv_enrichment(
        self, papers: list[Paper], request_id: str
    ) -> list[Paper]:
        """Enrich papers missing arXiv IDs by looking up their DOIs via CrossRef.

        Papers from OpenAlex/CrossRef/DBLP often lack an arXiv ID even when
        the underlying paper is on arXiv.  This stage batches DOI lookups to
        CrossRef, which returns the full record including links that may
        contain arXiv URLs.  Only papers with a DOI but no arXiv ID are
        looked up; existing arXiv IDs are never overwritten.
        """
        if not papers:
            return papers

        # Find the CrossRef provider (if any) for DOI→record lookup.
        crossref_provider = None
        for p in self._providers:
            if getattr(p, "source_name", "") == "crossref":
                crossref_provider = p
                break

        if crossref_provider is None:
            logger.debug(
                f"[{request_id}] Stage 7b: no CrossRef provider, "
                f"skipping arXiv enrichment"
            )
            return papers

        # Collect DOIs that need enrichment.
        to_enrich: list[tuple[int, str]] = []
        for i, paper in enumerate(papers):
            if paper.arxiv_id:
                continue
            # Also check identity.arxiv_id
            if paper.identity and paper.identity.arxiv_id:
                paper.arxiv_id = paper.identity.arxiv_id
                continue
            if paper.doi:
                to_enrich.append((i, paper.doi))

        if not to_enrich:
            logger.info(
                f"[{request_id}] Stage 7b: no papers need arXiv enrichment"
            )
            return papers

        enriched_count = 0
        sem = asyncio.Semaphore(5)  # limit concurrent CrossRef API calls

        async def _lookup(idx: int, doi: str) -> None:
            nonlocal enriched_count
            async with sem:
                try:
                    cr_paper = await crossref_provider.get_paper(doi)
                    if cr_paper and cr_paper.arxiv_id:
                        papers[idx].arxiv_id = cr_paper.arxiv_id
                        if papers[idx].identity:
                            papers[idx].identity.arxiv_id = cr_paper.arxiv_id
                        enriched_count += 1
                except Exception as e:
                    logger.debug(
                        f"[{request_id}] Stage 7b: CrossRef lookup "
                        f"failed for DOI {doi}: {e}"
                    )

        await asyncio.gather(
            *[_lookup(idx, doi) for idx, doi in to_enrich],
            return_exceptions=True,
        )

        logger.info(
            f"[{request_id}] Stage 7b: enriched {enriched_count}/"
            f"{len(to_enrich)} papers with arXiv IDs"
        )
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
            # LLM-based semantic topic labels for clusters
            if graph.clusters:
                await self._graph_builder.label_clusters_with_llm(
                    graph.clusters, papers
                )
            logger.info(
                f"[{request_id}] Stage 12: graph with {len(graph.nodes)} "
                f"nodes, {len(graph.clusters)} clusters"
            )
            return graph
        except Exception as e:
            logger.error(f"[{request_id}] Stage 12 failed: {e}")
            return ResearchGraph()

    async def _stage13_assembly(
        self,
        ranked: list[PaperWithScores],
        graph: ResearchGraph,
        search_query: SearchQuery,
        request_id: str,
        start_time: float,
    ) -> SearchResult:
        """Assemble final SearchResult."""
        latency_ms = (time.time() - start_time) * 1000

        # Generate LLM natural-language summary
        nl_summary = await self._generate_summary(
            ranked, graph, search_query, request_id
        )

        summary = SearchSummary(
            total_papers=len(ranked),
            query=search_query.original_query,
            domain=search_query.domain,
            intent=search_query.intent.value,
            top_paper_title=ranked[0].paper.title if ranked else None,
            clusters_count=len(graph.clusters),
            natural_language_summary=nl_summary,
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

    async def _generate_summary(
        self,
        ranked: list[PaperWithScores],
        graph: ResearchGraph,
        search_query: SearchQuery,
        request_id: str,
    ) -> str:
        """Generate a natural-language summary of search results via LLM.

        Falls back to a template-based summary on LLM failure.
        """
        if not ranked:
            return (
                f"No papers found for query '{search_query.original_query}'. "
                f"Consider refining the search terms."
            )

        # Build paper digest for LLM input
        top_papers = ranked[:10]
        paper_lines: list[str] = []
        tier_counts: dict[str, int] = {}
        for i, pws in enumerate(top_papers, 1):
            tier = pws.relevance_tier or "unrated"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            paper_lines.append(
                f"{i}. [{tier}] {pws.paper.title} "
                f"({pws.paper.year or 'N/A'}) "
                f"citations={pws.paper.citation_count or 0} "
                f"score={pws.final_score:.3f}"
            )
        papers_str = "\n".join(paper_lines)

        cluster_lines: list[str] = []
        for c in graph.clusters[:5]:
            cluster_lines.append(f"- {c.label} ({len(c.paper_ids)} papers)")
        clusters_str = "\n".join(cluster_lines) or "No clusters identified."

        prompt = (
            f"You are a research assistant. Summarize the following search results.\n\n"
            f"Query: {search_query.original_query}\n"
            f"Domain: {search_query.domain}\n"
            f"Intent: {search_query.intent.value}\n"
            f"Total papers found: {len(ranked)}\n"
            f"Clusters:\n{clusters_str}\n\n"
            f"Top papers:\n{papers_str}\n\n"
            f"Write a concise summary (3-5 sentences) covering:\n"
            f"1. What the search found overall\n"
            f"2. Key themes/clusters\n"
            f"3. Most relevant papers and why\n"
            f"Respond in the same language as the query."
        )

        try:
            from app.agents.qwen import QwenAgent

            provider = QwenAgent().provider
            resp = await provider.generate(
                prompt=prompt,
                system_prompt="You are a helpful research assistant.",
                temperature=0.3,
            )
            self._metrics.record_llm_call(getattr(provider, "last_token_usage", 0))
            self._metrics.record_model_used(provider.model_name)
            return resp.content.strip()
        except Exception as e:
            logger.warning(
                f"[{request_id}] Summary LLM failed: {e}, using template"
            )
            tier_str = ", ".join(
                f"{k}: {v}" for k, v in tier_counts.items()
            )
            return (
                f"Found {len(ranked)} papers for '{search_query.original_query}'. "
                f"Top result: {ranked[0].paper.title} "
                f"({ranked[0].paper.year or 'N/A'}). "
                f"Relevance distribution: {tier_str}. "
                f"Organized into {len(graph.clusters)} thematic clusters."
            )

    async def _refine_query_for_next_round(
        self,
        search_query: SearchQuery,
        ranked: list[PaperWithScores],
        request_id: str,
    ) -> Optional[SearchQuery]:
        """Dynamically refine the search query for the next iteration.

        Uses an LLM to analyze the current results and suggest:
        - Broader/different keywords if too few relevant papers
        - Alternative domain terminology
        - Relaxed constraints (e.g. broader year range)
        """
        if not ranked:
            # No results at all — try broadening the query
            new_core = search_query.semantic_core
            new_keywords = list(search_query.keywords) if search_query.keywords else []
            # Add a broader fallback: use original query verbatim
            return SearchQuery(
                original_query=search_query.original_query,
                semantic_core=new_core,
                domain=search_query.domain,
                intent=search_query.intent,
                keywords=new_keywords,
                hard_filters=search_query.hard_filters,
                options=search_query.options,
            )

        # Build a summary of what was found
        top_titles = [pws.paper.title for pws in ranked[:5]]
        avg_rel = sum(p.relevance_score for p in ranked) / len(ranked)

        prompt = (
            f"You are a search strategist. The current search yielded "
            f"{len(ranked)} papers with avg relevance {avg_rel:.2f}.\n\n"
            f"Current query: {search_query.semantic_core}\n"
            f"Domain: {search_query.domain}\n"
            f"Keywords: {search_query.keywords}\n\n"
            f"Top results found:\n"
            + "\n".join(f"- {t}" for t in top_titles)
            + "\n\nThe results are insufficient. Suggest a refined search "
            "strategy. Return a JSON object with:\n"
            '"semantic_core": "<refined core query>",\n'
            '"keywords": ["<kw1>", "<kw2>", ...],\n'
            '"domain": "<domain or unchanged>"\n'
            "Respond with ONLY the JSON, no explanation."
        )

        try:
            from app.agents.qwen import QwenAgent

            provider = QwenAgent().provider
            resp = await provider.generate(
                prompt=prompt,
                system_prompt="You are a search strategy assistant.",
                temperature=0.4,
            )
            import json

            # Parse JSON from response
            content = resp.content.strip()
            # Extract JSON from potential markdown code block
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            data = json.loads(content)

            new_core = data.get("semantic_core", search_query.semantic_core)
            new_keywords = data.get("keywords", search_query.keywords)
            new_domain = data.get("domain", search_query.domain)

            logger.info(
                f"[{request_id}] Query refined: "
                f"core='{new_core[:60]}', "
                f"keywords={new_keywords[:3]}"
            )

            return SearchQuery(
                original_query=search_query.original_query,
                semantic_core=new_core,
                domain=new_domain,
                intent=search_query.intent,
                keywords=new_keywords,
                hard_filters=search_query.hard_filters,
                options=search_query.options,
            )
        except Exception as e:
            logger.warning(
                f"[{request_id}] Query refinement failed: {e}"
            )
            return None

    async def _update_thompson_reward(
        self,
        search_query: SearchQuery,
        ranked: list[PaperWithScores],
        request_id: str,
    ) -> None:
        """Compute reward from search results and update Thompson state.

        Reward = weighted combination of:
        - Number of highly relevant papers (relevance_score >= threshold)
        - Average relevance score
        - Diversity of top-k papers

        This closes the Thompson sampling feedback loop.
        """
        if self._thompson is None or not ranked:
            return

        try:
            rel_floor = getattr(
                self._settings, "relevance_floor", 0.30
            )
            highly_relevant = sum(
                1 for p in ranked if p.relevance_score >= rel_floor
            )
            avg_relevance = sum(
                p.relevance_score for p in ranked
            ) / len(ranked)

            # Diversity: count unique first-authors in top-10
            top_authors = set()
            for pws in ranked[:10]:
                if pws.paper.authors:
                    top_authors.add(pws.paper.authors[0].lower())
            diversity = len(top_authors) / min(len(ranked), 10)

            reward = (
                min(highly_relevant / 10.0, 1.0) * 0.5
                + avg_relevance * 0.3
                + diversity * 0.2
            )

            # Update Thompson state for each model
            model_names = self._coordinator.model_names
            model_rewards = {
                name: reward for name in model_names
            }
            self._thompson.update_state_batch(
                model_rewards=model_rewards,
                domain=search_query.domain,
                query_type=search_query.query_type,
            )

            logger.info(
                f"[{request_id}] Thompson reward={reward:.3f} "
                f"(highly_relevant={highly_relevant}, "
                f"avg_rel={avg_relevance:.3f}, div={diversity:.3f})"
            )
        except Exception as e:
            logger.warning(
                f"[{request_id}] Thompson reward update failed: {e}"
            )

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

    # -----------------------------------------------------------------------
    # Worker mode: agents do their own procurement + judging, strong model
    # reviews all reports and makes the final selection.
    # -----------------------------------------------------------------------

    async def _run_worker_mode(
        self,
        search_query: SearchQuery,
        candidates: list[CandidateQuery],
        user_query: UserQuery,
        request_id: str,
        start_time: float,
    ) -> SearchResult:
        """Worker-mode flow: delegate procurement to agents, strong model reviews.

        Flow:
        1. Group candidates by agent model (from stage 2).
        2. Inject providers into agents.
        3. dispatch_and_collect → each agent searches + judges + reports.
        4. Strong model reviews all 3 reports → final selection.
        5. Dedup, citation expansion, graph, assembly (reuse existing stages).
        """
        # 2b. Inject providers so agents can dispatch retrieval themselves.
        max_per = self._settings.max_per_source if hasattr(self._settings, "max_per_source") else 10
        self._coordinator.inject_providers(self._providers, max_per_source=max_per)

        # Group candidates by proposer model for dispatch.
        candidates_per_agent: dict[str, list[CandidateQuery]] = {}
        for c in candidates:
            candidates_per_agent.setdefault(c.proposer_model, []).append(c)

        # Dispatch each agent independently.
        reports = await self._coordinator.dispatch_and_collect(
            query=search_query,
            candidates_per_agent=candidates_per_agent,
            year_start=self._year_start,
            year_end=self._year_end,
        )
        for rep in reports:
            self._metrics.record_llm_call(rep.token_usage)
            logger.info(
                f"[{request_id}] Worker {rep.agent_model}: "
                f"{len(rep.paper_reports)} papers, "
                f"{rep.token_usage} tokens, "
                f"{rep.latency_ms:.0f}ms"
            )

        # Strong model reviews all reports.
        final_selection = await self._judge.review_agent_reports(
            search_query, reports
        )
        self._metrics.record_llm_call(self._judge.last_token_usage)

        papers = [p for p, _, _, _ in final_selection]
        judge_results: list[PaperJudgeResult] = []
        for paper, score, reasoning, endorsed in final_selection:
            judge_results.append(
                PaperJudgeResult(
                    paper_id=paper.identity_key() or paper.title,
                    relevance_score=score,
                    reasoning=reasoning,
                    key_findings=[f"endorsed by: {', '.join(endorsed)}"],
                )
            )

        logger.info(
            f"[{request_id}] Worker mode: strong model selected "
            f"{len(papers)} papers from {sum(len(r.paper_reports) for r in reports)} "
            f"candidates reported by {len(reports)} agents"
        )

        # Dedup (some agents may have found the same paper).
        papers = await self._stage7_dedup(papers, request_id)

        # Citation expansion (optional, reuse existing stage).
        papers = await self._stage6_citation_expansion(
            papers, search_query, request_id
        )

        # Final ranking + MMR (reuse existing stage).
        ranked = await self._stage10_11_final_ranking(
            papers, judge_results, search_query, request_id
        )

        # Research graph.
        graph = await self._stage12_research_graph(
            ranked, search_query, request_id
        )

        # Assembly.
        result = await self._stage13_assembly(
            ranked, graph, search_query, request_id, start_time
        )

        # Metrics.
        self._stage14_metrics(result, request_id, start_time)

        logger.info(
            f"[{request_id}] Worker pipeline completed in "
            f"{result.latency_ms:.0f}ms, "
            f"{len(result.papers)} papers returned"
        )

        # Reset dynamic year filters.
        self._year_start = None
        self._year_end = None

        return result
