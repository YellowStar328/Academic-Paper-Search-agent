"""run_ablation.py — run ablation study (A-H) and output CSV."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging

from app.config import Settings
from app.models.query import SearchOptions, UserQuery
from scripts.run_experiment import build_pipeline

logger = logging.getLogger(__name__)

ABLATION_LEVELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

ABLATION_CONFIG = {
    "A": {"strategy": "single", "citation": False, "embedding": False},
    "B": {"strategy": "multi", "citation": False, "embedding": False},
    "C": {"strategy": "random", "citation": False, "embedding": False},
    "D": {"strategy": "greedy", "citation": False, "embedding": False},
    "E": {"strategy": "thompson", "citation": False, "embedding": False},
    "F": {"strategy": "thompson", "citation": True, "embedding": False},
    "G": {"strategy": "thompson", "citation": True, "embedding": True},
    "H": {"strategy": "thompson_full", "citation": True, "embedding": True},
}


def _get_level_description(level: str) -> str:
    descriptions = {
        "A": "Baseline: single agent, no enhancements",
        "B": "+ Multi-agent query generation",
        "C": "+ Random budget allocation",
        "D": "+ Greedy budget allocation",
        "E": "+ Thompson Sampling",
        "F": "+ Citation expansion",
        "G": "+ Embedding reranking",
        "H": "+ MMR diversity (full pipeline)",
    }
    return descriptions.get(level, "")


async def run_ablation_level(
    query: str, level: str, settings: Settings | None = None
) -> dict:
    config = ABLATION_CONFIG[level]
    s = settings or Settings(app_env="test")
    pipeline = build_pipeline(config["strategy"], s)
    user_query = UserQuery(
        query=query,
        options=SearchOptions(
            enable_citation_expansion=config["citation"],
            enable_embedding_rerank=config["embedding"],
        ),
    )
    result = await pipeline.run(user_query)
    metrics = result.metrics
    return {
        "level": level,
        "description": _get_level_description(level),
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


async def run_all_ablations(
    query: str, settings: Settings | None = None
) -> list[dict]:
    results = []
    for level in ABLATION_LEVELS:
        logger.info(f"Running ablation level {level}")
        try:
            r = await run_ablation_level(query, level, settings)
            results.append(r)
        except Exception as e:
            logger.error(f"Ablation {level} failed: {e}")
            results.append({"level": level, "query": query, "error": str(e)})
    return results


def write_csv(results: list[dict], output_path: str) -> None:
    if not results:
        return
    fields = [
        "level", "description", "query", "total_papers", "latency_ms",
        "top_score", "papers_collected", "papers_after_dedup",
        "papers_final", "llm_calls", "token_usage",
        "clusters", "timeline_entries",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info(f"Ablation results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study A-H")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--level", choices=ABLATION_LEVELS, help="Specific level")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.level:
        results = [asyncio.run(run_ablation_level(args.query, args.level))]
    else:
        results = asyncio.run(run_all_ablations(args.query))

    write_csv(results, args.output)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
