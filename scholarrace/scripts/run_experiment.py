"""run_experiment.py — run search with different strategies and output CSV.

Supports 6 strategies:
- single: single-model query generation (no multi-agent)
- multi: multi-agent without strong judge
- random: random budget allocation
- greedy: greedy budget allocation (always pick best historical)
- thompson: Thompson Sampling budget allocation
- thompson_full: Thompson + citation expansion + embedding + MMR (full pipeline)

Usage:
    python -m scripts.run_experiment --query "transformer survey" --strategy thompson
    python -m scripts.run_experiment --query "..." --all-strategies --output results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from typing import Optional

from app.agents.mock import MockLLMProvider
from app.agents.qwen import QwenAgent
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.coordinator import MultiAgentCoordinator
from app.agents.judge import PaperJudge, StrongJudge
from app.citation.expansion import CitationExpander
from app.config import Settings, get_settings
from app.embedding.encoder import FakeEncoder
from app.models.paper import Paper, PaperIdentity, PaperList
from app.models.query import UserQuery
from app.models.result import SearchResult
from app.pipeline.search_pipeline import SearchPipeline
from app.query.parser import QueryParser
from app.ranking.final_ranker import FinalRanker
from app.retrieval.base import BaseSearchProvider
from app.utils.observability import MetricsTracker

logger = logging.getLogger(__name__)

STRATEGIES = [
    "single",
    "multi",
    "random",
    "greedy",
    "thompson",
    "thompson_full",
]


class MockSearchProvider(BaseSearchProvider):
    """Mock provider for experiments."""

    def __init__(self, papers: list[Paper], source_name: str = "mock"):
        super().__init__(http_client=None)
        self._papers = papers
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    async def search(self, query: str, max_results: int = 50) -> PaperList:
        return PaperList(papers=self._papers[:max_results], source=self._source_name)


def make_experiment_papers() -> list[Paper]:
    """Create a set of test papers for experiments."""
    from uuid import uuid4

    papers_data = [
        ("Attention Is All You Need", 2017, 500, "10.1/attention"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", 2019, 300, "10.1/bert"),
        ("GPT-4 Technical Report", 2023, 200, "10.1/gpt4"),
        ("Deep Residual Learning for Image Recognition", 2016, 400, "10.1/resnet"),
        ("Generative Adversarial Networks", 2014, 600, "10.1/gan"),
        ("ImageNet Classification with Deep CNNs", 2012, 700, "10.1/alexnet"),
        ("Sequence to Sequence Learning with Neural Networks", 2014, 350, "10.1/seq2seq"),
        ("Language Models are Few-Shot Learners", 2020, 250, "10.1/gpt3"),
        ("Adam: A Method for Stochastic Optimization", 2015, 450, "10.1/adam"),
        ("Dropout: A Simple Way to Prevent Neural Networks from Overfitting", 2014, 300, "10.1/dropout"),
    ]

    papers = []
    for title, year, citations, doi in papers_data:
        papers.append(
            Paper(
                paper_id=str(uuid4()),
                identity=PaperIdentity(
                    doi=doi,
                    normalized_title=title.lower().replace(" ", ""),
                    year=year,
                ),
                title=title,
                abstract=f"Research paper about {title}",
                year=year,
                citation_count=citations,
                source="arxiv",
            )
        )
    return papers


def build_pipeline(strategy: str, settings: Settings | None = None) -> SearchPipeline:
    """Build a pipeline configured for the given strategy."""
    s = settings or Settings(app_env="test")
    llm = MockLLMProvider()

    parser = QueryParser(provider=llm)

    if strategy == "single":
        qwen = QwenAgent(llm)
        coordinator = MultiAgentCoordinator(qwen_agent=qwen)
    else:
        qwen = QwenAgent(llm)
        deepseek = DeepSeekAgent(llm)
        glm = GLMAgent(llm)
        coordinator = MultiAgentCoordinator(
            qwen_agent=qwen, deepseek_agent=deepseek, glm_agent=glm
        )

    strong_judge = StrongJudge(provider=llm)
    paper_judge = PaperJudge(provider=llm)

    papers = make_experiment_papers()
    providers = [
        MockSearchProvider(papers, "arxiv"),
        MockSearchProvider(papers[:5], "semantic_scholar"),
    ]

    if strategy == "thompson_full":
        citation_expander = CitationExpander(providers=providers)
    else:
        citation_expander = CitationExpander(providers=[])

    final_ranker = FinalRanker()

    thompson_manager = None
    if strategy in ("thompson", "thompson_full"):
        from app.bandit.thompson import ThompsonSamplingManager
        thompson_manager = ThompsonSamplingManager()

    return SearchPipeline(
        query_parser=parser,
        coordinator=coordinator,
        strong_judge=strong_judge,
        providers=providers,
        citation_expander=citation_expander,
        paper_judge=paper_judge,
        final_ranker=final_ranker,
        thompson_manager=thompson_manager,
        settings=s,
    )


async def run_single_experiment(
    query: str, strategy: str, settings: Settings | None = None
) -> dict:
    """Run a single experiment and return metrics."""
    pipeline = build_pipeline(strategy, settings)
    user_query = UserQuery(query=query)
    result = await pipeline.run(user_query)

    metrics = result.metrics
    return {
        "strategy": strategy,
        "query": query,
        "total_papers": len(result.papers),
        "latency_ms": round(result.latency_ms, 2),
        "top_score": round(result.papers[0].final_score, 4) if result.papers else 0,
        "papers_collected": metrics.papers_collected if metrics else 0,
        "papers_after_dedup": metrics.papers_after_dedup if metrics else 0,
        "papers_final": metrics.papers_final if metrics else 0,
        "llm_calls": metrics.llm_calls if metrics else 0,
        "token_usage": metrics.token_usage if metrics else 0,
        "clusters": len(result.graph.clusters),
        "timeline_entries": len(result.graph.timeline),
    }


async def run_all_strategies(
    query: str, settings: Settings | None = None
) -> list[dict]:
    """Run all strategies for the given query."""
    results = []
    for strategy in STRATEGIES:
        logger.info(f"Running strategy: {strategy}")
        try:
            r = await run_single_experiment(query, strategy, settings)
            results.append(r)
        except Exception as e:
            logger.error(f"Strategy {strategy} failed: {e}")
            results.append({
                "strategy": strategy,
                "query": query,
                "error": str(e),
            })
    return results


def write_csv(results: list[dict], output_path: str) -> None:
    """Write results to CSV."""
    if not results:
        return

    fields = [
        "strategy", "query", "total_papers", "latency_ms", "top_score",
        "papers_collected", "papers_after_dedup", "papers_final",
        "llm_calls", "token_usage", "clusters", "timeline_entries",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run search experiments")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="thompson_full",
        help="Strategy to use",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run all strategies",
    )
    parser.add_argument(
        "--output",
        default="experiment_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.all_strategies:
        results = asyncio.run(run_all_strategies(args.query))
    else:
        results = [asyncio.run(run_single_experiment(args.query, args.strategy))]

    write_csv(results, args.output)

    # Also print to stdout
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
