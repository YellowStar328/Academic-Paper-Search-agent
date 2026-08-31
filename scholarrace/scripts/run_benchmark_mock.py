"""run_benchmark_mock.py — run PaSa benchmark with MockLLMProvider.

Used to verify worker_mode pipeline correctness when real LLM API is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.mock import MockLLMProvider
from app.agents.qwen import QwenAgent
from app.agents.deepseek import DeepSeekAgent
from app.agents.glm import GLMAgent
from app.agents.coordinator import MultiAgentCoordinator
from app.agents.judge import StrongJudge
from app.citation.expansion import CitationExpander
from app.embedding.encoder import FakeEncoder
from app.pipeline.search_pipeline import SearchPipeline
from app.query.parser import QueryParser
from app.retrieval.arxiv import ArxivProvider
from app.retrieval.crossref import CrossrefProvider
from app.retrieval.dblp import DblpProvider
from app.retrieval.openalex import OpenAlexProvider
from app.retrieval.semantic_scholar import SemanticScholarProvider
from app.models.query import UserQuery, SearchOptions

from scripts.run_benchmark import load_pasa_dataset, _result_ids, compute_metrics, print_comparison

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark_mock")


def build_mock_pipeline(worker_mode: bool = False) -> SearchPipeline:
    """Build pipeline with MockLLM (no real API calls)."""
    mock = MockLLMProvider()
    parser = QueryParser(provider=mock)

    qwen = QwenAgent(provider=mock)
    deepseek = DeepSeekAgent(provider=mock)
    glm = GLMAgent(provider=mock)
    coordinator = MultiAgentCoordinator(
        qwen_agent=qwen, deepseek_agent=deepseek, glm_agent=glm
    )
    strong_judge = StrongJudge(provider=mock)

    providers = [
        SemanticScholarProvider(),
        ArxivProvider(),
        OpenAlexProvider(),
        CrossrefProvider(),
        DblpProvider(),
    ]
    citation_expander = CitationExpander(providers=providers)
    encoder = FakeEncoder()

    return SearchPipeline(
        query_parser=parser,
        coordinator=coordinator,
        strong_judge=strong_judge,
        providers=providers,
        citation_expander=citation_expander,
        embedding_encoder=encoder,
        worker_mode=worker_mode,
    )


async def run_one(pipeline: SearchPipeline, example: dict, top_k: int) -> dict:
    q = example["query"]
    gold = example["gold_ids"]
    options = SearchOptions(top_k=top_k)
    user_query = UserQuery(query=q, options=options)
    result = await pipeline.run(user_query)
    retrieved = _result_ids(result)
    metrics = compute_metrics(retrieved, gold, top_k)
    return {"query": q[:80], "gold": list(gold), "retrieved": retrieved, **metrics}


async def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    worker_mode = "--worker-mode" in sys.argv
    top_k = 50

    examples = load_pasa_dataset(skip_download=True)[:limit]
    logger.info("Loaded %d examples, worker_mode=%s", len(examples), worker_mode)

    pipeline = build_mock_pipeline(worker_mode=worker_mode)

    per_query = []
    for i, ex in enumerate(examples):
        logger.info("[%d/%d] %s", i + 1, len(examples), ex["query"][:60])
        r = await run_one(pipeline, ex, top_k)
        per_query.append(r)
        logger.info("  R@%d=%.3f P@%d=%.3f MRR=%.3f retrieved=%d gold=%d",
                    top_k, r[f"recall@{top_k}"], top_k, r[f"precision@{top_k}"],
                    r[f"mrr@{top_k}"], len(r["retrieved"]), len(r["gold"]))

    # aggregate
    agg = {}
    for m in [f"recall@{top_k}", f"precision@{top_k}", f"f1@{top_k}", f"mrr@{top_k}", f"hit@{top_k}"]:
        vals = [r[m] for r in per_query]
        agg[m] = sum(vals) / len(vals) if vals else 0
    agg["num_queries"] = len(per_query)
    agg["avg_retrieved"] = sum(len(r["retrieved"]) for r in per_query) / len(per_query) if per_query else 0
    agg["avg_gold"] = sum(len(r["gold"]) for r in per_query) / len(per_query) if per_query else 0

    mode = "worker_mode" if worker_mode else "normal_mode"
    print(f"\n===== PaSa Benchmark (mock LLM, {mode}, {len(per_query)} queries) =====")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print_comparison("pasa", agg, top_k)

    # per-query details
    print("\n--- Per-query ---")
    for r in per_query:
        print(f"  Q: {r['query']}")
        print(f"    gold={r['gold']}")
        print(f"    retrieved={r['retrieved']}")
        print(f"    R@{top_k}={r[f'recall@{top_k}']:.3f} MRR={r[f'mrr@{top_k}']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
