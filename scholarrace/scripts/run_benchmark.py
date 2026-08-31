"""run_benchmark.py — benchmark ScholarRace agent on PaSa & ASTA datasets.

Runs the real ScholarRace pipeline (real LLM agents + real Semantic Scholar /
arXiv providers) against two academic search benchmarks:

1. PaSa RealScholarQuery (ByteDance/北大)
   - Dataset: HuggingFace `CarlanLark/pasa-dataset`, split `RealScholarQuery/test`
   - Format: {"question": str, "answer": [title, ...], "answer_arxiv_id": [id, ...]}
   - Gold papers matched by arXiv ID (no S2 corpus_id in dataset)
   - Metrics: Recall@K, Precision@K, F1@K, MRR

2. ASTA PaperFindingBench (AllenAI)
   - Dataset: HuggingFace `allenai/asta-bench`, split `paper_finding/test`
   - Format: {"query": str, "gold_papers": [{"semantic_scholar_id": str, ...}, ...]}
   - Metrics: Recall@K, Precision@K, F1@K, MRR

Usage (run from scholarrace/ dir):
    python3 -m scripts.run_benchmark --benchmark pasa --limit 10 --output results/pasa.json
    python3 -m scripts.run_benchmark --benchmark asta --output results/asta.json
    python3 -m scripts.run_benchmark --benchmark all --output results/

Full options:
    --benchmark {pasa,asta,all}
    --limit N            only run first N queries (0 = all)
    --top-k K            K for Recall@K / Precision@K (default 50)
    --output PATH        write JSON report (default benchmark_report.json)
    --concurrency N      parallel query execution (default 1; increase with caution)
    --skip-download      reuse cached dataset in data/
    --dry-run           build pipeline + load data but skip LLM calls (sanity check)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Ensure the scholarrace package root is importable when run as a script.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from app.agents.coordinator import MultiAgentCoordinator  # noqa: E402
from app.agents.judge import PaperJudge, StrongJudge  # noqa: E402
from app.agents.qwen import QwenAgent  # noqa: E402
from app.agents.deepseek import DeepSeekAgent  # noqa: E402
from app.agents.glm import GLMAgent  # noqa: E402
from app.citation.expansion import CitationExpander  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embedding.encoder import ApiEncoder, FakeEncoder  # noqa: E402
from app.models.query import SearchOptions, UserQuery  # noqa: E402
from app.pipeline.search_pipeline import SearchPipeline  # noqa: E402
from app.query.parser import QueryParser  # noqa: E402
from app.ranking.final_ranker import FinalRanker  # noqa: E402
from app.retrieval.arxiv import ArxivProvider  # noqa: E402
from app.retrieval.crossref import CrossrefProvider  # noqa: E402
from app.retrieval.dblp import DblpProvider  # noqa: E402
from app.retrieval.openalex import OpenAlexProvider  # noqa: E402
from app.retrieval.semantic_scholar import SemanticScholarProvider  # noqa: E402
from app.utils.observability import MetricsTracker  # noqa: E402

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = _PKG_ROOT / "data" / "benchmark"


def _normalize_arxiv_id(aid: str) -> str:
    """Normalize arXiv ID: strip version suffix (e.g. 2309.04564v1 -> 2309.04564)."""
    import re

    aid = aid.strip()
    aid = re.sub(r"v\d+$", "", aid)
    return aid


_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{4}\.\d{1,5})$")


def _arxiv_id_to_year(aid: str) -> Optional[int]:
    """Extract publication year from an arXiv ID.

    Modern arXiv IDs use the format YYMM.NNNNN (e.g. 2309.04564 -> 2023).
    Legacy IDs like hep-th/9905110 use YYMM as the 4 digits after the slash.

    arXiv started in 1991, so:
        YY >= 91  -> 19YY
        YY <  91  -> 20YY
    """
    aid = aid.strip()
    # Strip version suffix
    aid = re.sub(r"v\d+$", "", aid)
    # Try modern format: YYMM.NNNNN
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", aid)
    if m:
        yy = int(m.group(1))
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return year
    # Try legacy format: category/YYMM.NNNN
    m = re.match(r"^[a-z\-]+/(\d{2})(\d{2})\.\d{1,5}$", aid)
    if m:
        yy = int(m.group(1))
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return year
    return None


def gold_ids_to_year_range(gold_ids: set[str]) -> tuple[Optional[int], Optional[int]]:
    """Compute the publication year range spanned by gold arXiv IDs.

    Returns (year_start, year_end) with a 1-year margin on each side to
    avoid edge effects.  Returns (None, None) if no years can be parsed.
    """
    years: list[int] = []
    for gid in gold_ids:
        y = _arxiv_id_to_year(gid)
        if y is not None:
            years.append(y)
    if not years:
        return None, None
    y_min = min(years)
    y_max = max(years)
    # Add 1-year margin to avoid missing gold papers at the boundary
    return y_min - 1, y_max + 1


def _hf_download(url: str, dest: Path) -> None:
    """Download a file from a URL to dest with a progress-free streaming save."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "scholarrace-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def load_pasa_dataset(skip_download: bool = False) -> list[dict]:
    """Load PaSa RealScholarQuery test split as a list of {query, gold_ids}.

    Supports loading from any local jsonl file in the benchmark data dir
    that matches the PaSa format ({"question": ..., "answer_arxiv_id": [...]}).
    Search order: pasa_real_scholar_query_test.jsonl, dev.jsonl, then download.
    """
    # Prefer the real scholar query test set (50 queries) over the dev
    # split (1000 synthetic queries) — the real set is what the user
    # benchmarked against and matches the PaSa paper's RealScholarQuery.
    candidates = [
        DATA_DIR / "pasa_real_scholar_query_test.jsonl",
        DATA_DIR / "dev.jsonl",
    ]
    dest = next((p for p in candidates if p.exists()), None)
    if dest is None:
        dest = candidates[-1]
        if skip_download:
            raise FileNotFoundError(
                f"PaSa dataset not found in {DATA_DIR} and --skip-download was set."
            )
        url = (
            "https://huggingface.co/datasets/CarlanLark/pasa-dataset/"
            "resolve/main/RealScholarQuery/test.jsonl"
        )
        try:
            _hf_download(url, dest)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to download PaSa dataset: %s", e)
            logger.error("If the URL is stale, manually place test.jsonl at %s", dest)
            raise
    else:
        logger.info("Using local dataset: %s", dest.name)
    examples: list[dict] = []
    with open(dest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = obj.get("question") or obj.get("query") or ""
            arxiv_ids = obj.get("answer_arxiv_id") or []
            gold_ids = {_normalize_arxiv_id(a) for a in arxiv_ids if a}
            if query and gold_ids:
                examples.append({
                    "query": query,
                    "gold_ids": gold_ids,
                    "raw": obj,
                })
    logger.info("Loaded %d PaSa examples from %s", len(examples), dest.name)
    return examples


def load_asta_dataset(skip_download: bool = False) -> list[dict]:
    """Load ASTA PaperFindingBench test split as a list of {query, gold_ids}."""
    # ASTA bench: allenai/asta-bench, paper_finding split.  The split is
    # shipped as a parquet; we try the jsonl mirror first, then fall back to
    # the HF `datasets` library if available.
    candidates = [
        (
            "https://huggingface.co/datasets/allenai/asta-bench/"
            "resolve/main/paper_finding/test.jsonl"
        ),
        DATA_DIR / "asta_paper_finding_test.jsonl",
    ]
    url, dest = candidates[0], candidates[1]
    if not skip_download:
        if not dest.exists():
            try:
                _hf_download(url, dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("Direct jsonl download failed (%s); trying datasets lib", e)
                dest = _load_asta_via_datasets(dest)
    examples: list[dict] = []
    if dest and dest.exists():
        with open(dest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = obj.get("query") or obj.get("question") or ""
                gold = obj.get("gold_papers") or obj.get("answer") or []
                ids = set()
                for g in gold:
                    gid = g.get("semantic_scholar_id") or g.get("corpus_id") or g.get("paper_id")
                    if gid:
                        ids.add(str(gid))
                if query and ids:
                    examples.append({"query": query, "gold_ids": ids, "raw": obj})
    logger.info("Loaded %d ASTA examples", len(examples))
    return examples


def _load_asta_via_datasets(dest: Path) -> Path:
    """Fallback: use the `datasets` library to load ASTA and dump to jsonl."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.error("`datasets` not installed; pip install datasets to use ASTA")
        return dest
    ds = load_dataset("allenai/asta-bench", "paper_finding", split="test")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest


# ---------------------------------------------------------------------------
# Real pipeline construction
# ---------------------------------------------------------------------------

def build_real_pipeline(
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    worker_mode: bool = False,
) -> SearchPipeline:
    """Build a SearchPipeline that uses REAL LLM + retrieval providers.

    Agents are constructed with provider=None so they auto-load the real
    DashScope / DeepSeek / GLM providers configured in .env when APP_ENV is
    not "test" and the API key is present.

    Parameters
    ----------
    year_start, year_end
        Optional publication year range for filtering search results.
        When set, all provider search calls are restricted to this range,
        preventing newer irrelevant papers from crowding out older gold
        answers in benchmark evaluation.
    """
    settings = get_settings()
    # sanity check: must not be test mode
    if settings.is_test:
        raise RuntimeError(
            "APP_ENV is 'test' — real pipeline requires APP_ENV=development or production. "
            "Check scholarrace/.env"
        )

    parser = QueryParser(provider=None)  # auto-loads strong judge provider

    qwen = QwenAgent(provider=None)
    deepseek = DeepSeekAgent(provider=None)
    glm = GLMAgent(provider=None)
    coordinator = MultiAgentCoordinator(
        qwen_agent=qwen, deepseek_agent=deepseek, glm_agent=glm
    )

    strong_judge = StrongJudge(provider=None)
    paper_judge = PaperJudge(provider=None)

    # Real retrieval providers — use all available sources for max coverage
    providers = [
        SemanticScholarProvider(),
        ArxivProvider(),
        OpenAlexProvider(),
        CrossrefProvider(),
        DblpProvider(),
    ]
    citation_expander = CitationExpander(providers=providers)
    final_ranker = FinalRanker()
    embedding_encoder = ApiEncoder()  # real embedding via DashScope API

    from app.bandit.thompson import ThompsonSamplingManager
    thompson_manager = ThompsonSamplingManager()

    return SearchPipeline(
        query_parser=parser,
        coordinator=coordinator,
        strong_judge=strong_judge,
        providers=providers,
        citation_expander=citation_expander,
        embedding_encoder=embedding_encoder,
        paper_judge=paper_judge,
        final_ranker=final_ranker,
        thompson_manager=thompson_manager,
        settings=settings,
        year_start=year_start,
        year_end=year_end,
        worker_mode=worker_mode,
    )


# ---------------------------------------------------------------------------
# Matching + metrics
# ---------------------------------------------------------------------------

def _result_ids(result: Any) -> list[str]:
    """Extract arxiv_ids from pipeline result papers (ordered by rank)."""
    ids: list[str] = []
    for pws in getattr(result, "papers", []) or []:
        paper = getattr(pws, "paper", None)
        if paper is None:
            continue
        aid = getattr(paper, "arxiv_id", None)
        if not aid:
            aid = getattr(paper.identity, "arxiv_id", None) if paper.identity else None
        if aid:
            normalized = _normalize_arxiv_id(str(aid))
            if _ARXIV_ID_RE.match(normalized):
                ids.append(normalized)
    return ids


async def run_one(
    pipeline: SearchPipeline,
    query: str,
    top_k: int,
    dry_run: bool = False,
) -> list[str]:
    """Run the pipeline for one query and return ordered paper arXiv IDs."""
    if dry_run:
        logger.info("[dry-run] skipping LLM call for: %s", query[:80])
        return []
    try:
        options = SearchOptions(top_k=top_k)
        user_query = UserQuery(query=query, options=options)
        result = await pipeline.run(user_query)
        return _result_ids(result)
    except Exception as e:  # noqa: BLE001
        logger.exception("Pipeline failed for query: %s — %s", query[:80], e)
        return []


def compute_metrics(retrieved: list[str], gold: set[str], k: int) -> dict:
    """Compute Recall@K, Precision@K, F1@K, MRR, Hit@K for a single query."""
    topk = retrieved[:k]
    hit_set = {x for x in topk if x in gold}
    num_gold = len(gold)
    num_ret = len(topk)
    tp = len(hit_set)
    recall = tp / num_gold if num_gold else 0.0
    precision = tp / num_ret if num_ret else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # MRR: reciprocal rank of first relevant hit
    mrr = 0.0
    for i, rid in enumerate(topk, 1):
        if rid in gold:
            mrr = 1.0 / i
            break
    hit = 1 if hit_set else 0
    return {
        f"recall@{k}": round(recall, 4),
        f"precision@{k}": round(precision, 4),
        f"f1@{k}": round(f1, 4),
        f"mrr@{k}": round(mrr, 4),
        f"hit@{k}": hit,
        "num_gold": num_gold,
        "num_retrieved": len(retrieved),
    }


# ---------------------------------------------------------------------------
# Reference metrics reported by the original papers (for comparison)
# ---------------------------------------------------------------------------

# PaSa (arXiv:2501.10120, ACL 2025) — RealScholarQuery (50 real-world queries).
# Table 4 in the paper: columns are Crawler Recall, Precision, Recall.
# Note: Google-based methods report only Recall@K (Table 5), not Precision/Recall.
# Source: https://arxiv.org/abs/2501.10120
PASA_REFERENCE: dict[str, dict] = {
    # Recall@K baselines (from Table 5 — AutoScholarQuery; RealScholarQuery
    # uses Precision/Recall columns below).
    "Google (Recall@20/50/100)": {
        "recall@20": 0.1568, "recall@50": 0.1891, "recall@100": 0.2015,
    },
    "Google Scholar (Recall@20/50/100)": {
        "recall@20": 0.0609, "recall@50": 0.0970, "recall@100": 0.1130,
    },
    "Google+GPT-4o (Recall@20/50/100)": {
        "recall@20": 0.1921, "recall@50": 0.2450, "recall@100": 0.2683,
    },
    # RealScholarQuery Precision/Recall (Table 4)
    "ChatGPT": {"precision": 0.2280, "recall": 0.2007},
    "GPT-o1": {"precision": 0.058, "recall": 0.0134},
    "PaSa-GPT-4o": {"crawler_recall": 0.5494, "precision": 0.4721, "recall": 0.3075},
    "PaSa-7b": {"crawler_recall": 0.7071, "precision": 0.5146, "recall": 0.6111},
    "PaSa-7b-ensemble": {"crawler_recall": 0.7503, "precision": 0.4938, "recall": 0.6488},
}

# ASTA (AstaBench, ICLR 2026, OpenReview M7TNf5J26u) — PaperFindingBench.
# The full paper is not publicly accessible yet; metrics to be filled from
# https://openreview.net/forum?id=M7TNf5J26u once available.
ASTA_REFERENCE: dict[str, dict] = {
    # Placeholder — replace with actual numbers from the ASTA paper/leaderboard.
    # "ASTA (GPT-4.1)": {"precision": ..., "recall": ..., "f1": ...},
    # "ReAct (GPT-4.1)": {"precision": ..., "recall": ..., "f1": ...},
    "_note": (
        "ASTA paper results not yet publicly available (ICLR 2026, OpenReview "
        "M7TNf5J26u). Fill from https://openreview.net/forum?id=M7TNf5J26u "
        "or the leaderboard once accessible."
    ),
}


def print_comparison(
    benchmark: str,
    our_agg: dict | None,
    top_k: int,
) -> None:
    """Print a comparison table of ScholarRace vs reference methods.

    PaSa RealScholarQuery reports: Crawler Recall, Precision, Recall.
    Our pipeline outputs: Recall@K, Precision@K, F1@K, MRR@K.
    We display both side-by-side, mapping our Recall@K → Recall, etc.
    """
    ref = PASA_REFERENCE if benchmark == "pasa" else ASTA_REFERENCE
    rkey = f"recall@{top_k}"
    pkey = f"precision@{top_k}"
    fkey = f"f1@{top_k}"

    print(f"\n{'='*80}")
    title = f"  {benchmark.upper()} Benchmark — ScholarRace vs Reference Methods (K={top_k})"
    print(title)
    print(f"{'='*80}")
    header = f"{'Method':<28}{'Recall':>10}{'Precision':>12}{'F1':>8}{'Crawler R':>12}"
    print(header)
    print("-" * 80)

    # ScholarRace row
    if our_agg is not None:
        our_r = our_agg.get(rkey, 0)
        our_p = our_agg.get(pkey, 0)
        our_f1 = our_agg.get(fkey, 0)
        print(f"{'ScholarRace (ours)':<28}{our_r:>10.4f}{our_p:>12.4f}{our_f1:>8.4f}{'—':>12}")
    else:
        print(f"{'ScholarRace (ours)':<28}{'N/A':>10}{'N/A':>12}{'N/A':>8}{'—':>12}")
    print("-" * 80)

    for method, vals in ref.items():
        if method.startswith("_"):
            continue
        rv = vals.get("recall", vals.get(rkey, "—"))
        pv = vals.get("precision", vals.get(pkey, "—"))
        cr = vals.get("crawler_recall", "—")
        fv = "—"
        if isinstance(rv, float) and isinstance(pv, float) and rv + pv > 0:
            fv = round(2 * rv * pv / (rv + pv), 4)
        rv_s = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
        pv_s = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
        cr_s = f"{cr:.4f}" if isinstance(cr, float) else str(cr)
        fv_s = f"{fv:.4f}" if isinstance(fv, float) else str(fv)
        print(f"{method:<28}{rv_s:>10}{pv_s:>12}{fv_s:>8}{cr_s:>12}")

    note = ref.get("_note")
    if note:
        print(f"\n  Note: {note}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------




async def run_benchmark(
    examples: list[dict],
    pipeline: SearchPipeline,
    top_k: int,
    concurrency: int,
    dry_run: bool,
) -> list[dict]:
    """Run pipeline over all benchmark examples with optional concurrency."""
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(idx: int, ex: dict) -> dict:
        async with sem:
            t0 = time.time()
            # Gold year-range filtering has been disabled: it was too
            # restrictive and dropped relevant results.  Search is now
            # unconstrained and relies on ranking signals.
            retrieved = await run_one(
                pipeline, ex["query"], top_k, dry_run,
            )
            elapsed = time.time() - t0
            m = compute_metrics(retrieved, ex["gold_ids"], top_k)
            m["query"] = ex["query"]
            m["elapsed_s"] = round(elapsed, 2)
            m["retrieved_ids"] = retrieved[:top_k]
            m["gold_ids"] = sorted(ex["gold_ids"])
            logger.info(
                "[%d/%d] R@%d=%.3f P@%d=%.3f F1=%.3f MRR=%.3f (%.1fs) %s",
                idx + 1,
                len(examples),
                top_k,
                m[f"recall@{top_k}"],
                top_k,
                m[f"precision@{top_k}"],
                m[f"f1@{top_k}"],
                m[f"mrr@{top_k}"],
                elapsed,
                ex["query"][:60],
            )
            return m

    tasks = [_worker(i, ex) for i, ex in enumerate(examples)]
    return await asyncio.gather(*tasks)


def aggregate(per_query: list[dict], top_k: int) -> dict:
    """Compute macro-averaged metrics across all queries."""
    n = len(per_query) or 1
    keys = [f"recall@{top_k}", f"precision@{top_k}", f"f1@{top_k}", f"mrr@{top_k}", f"hit@{top_k}"]
    agg: dict[str, float] = {}
    for k in keys:
        agg[k] = round(sum(q[k] for q in per_query) / n, 4)
    agg["num_queries"] = len(per_query)
    agg["avg_latency_s"] = round(sum(q["elapsed_s"] for q in per_query) / n, 2)
    agg["avg_retrieved"] = round(sum(q["num_retrieved"] for q in per_query) / n, 1)
    agg["avg_gold"] = round(sum(q["num_gold"] for q in per_query) / n, 1)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run ScholarRace benchmark evaluation")
    ap.add_argument("--benchmark", choices=["pasa", "asta", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="only run first N queries (0=all)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--output", default="benchmark_report.json")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker-mode", action="store_true",
                    help="Enable worker mode: agents do their own search+judge, strong model reviews")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    benchmarks: list[str] = ["pasa", "asta"] if args.benchmark == "all" else [args.benchmark]

    pipeline = None
    if not args.dry_run:
        logger.info("Building real pipeline (APP_ENV=%s, worker_mode=%s)...",
                    os.environ.get("APP_ENV", "development"), args.worker_mode)
        pipeline = build_real_pipeline(worker_mode=args.worker_mode)
    else:
        logger.warning("DRY RUN — pipeline will not execute; no metrics produced")

    report: dict = {
        "tool": "ScholarRace",
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "dry_run": args.dry_run,
        "worker_mode": args.worker_mode,
        "benchmarks": {},
    }

    for name in benchmarks:
        logger.info("=== Benchmark: %s ===", name)
        if name == "pasa":
            examples = load_pasa_dataset(args.skip_download)
        else:
            examples = load_asta_dataset(args.skip_download)

        if args.limit and args.limit > 0:
            examples = examples[: args.limit]
        if not examples:
            logger.error("No examples loaded for %s — skipping", name)
            continue

        per_query: list[dict] = []
        if pipeline is not None:
            per_query = asyncio.run(
                run_benchmark(examples, pipeline, args.top_k, args.concurrency, args.dry_run)
            )
        agg = aggregate(per_query, args.top_k)
        report["benchmarks"][name] = {
            "num_examples": len(examples),
            "aggregate": agg,
            "per_query": per_query,
        }
        # reference comparison table
        report["benchmarks"][name]["reference_metrics"] = (
            PASA_REFERENCE if name == "pasa" else ASTA_REFERENCE
        )
        print_comparison(name, agg if per_query else None, args.top_k)

        print(f"\n===== {name.upper()} Results =====")
        print(json.dumps(agg, indent=2, ensure_ascii=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written to %s", out_path)


if __name__ == "__main__":
    main()
