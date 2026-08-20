# ScholarRace

Academic search multi-agent system — intelligent literature discovery through multi-model query generation, Thompson Sampling budget allocation, and multi-source retrieval with citation expansion.

## Overview

ScholarRace outperforms single ReAct agents for complex academic queries by:
- **Multi-agent parallel query generation**: Qwen, DeepSeek, and GLM each generate sub-queries from different perspectives
- **Strong model judge**: An independent model evaluates candidate queries (no self-evaluation)
- **Thompson Sampling**: Beta-distribution-based dynamic budget allocation prevents model collapse
- **Multi-source retrieval**: arXiv (real) + Semantic Scholar/OpenAlex/PubMed/Crossref (mock interfaces)
- **Citation expansion**: Depth-1 reference/citation network traversal for high-value papers
- **Embedding coarse ranking**: Deterministic FakeEncoder truncates 1000+ candidates to Top-100
- **LLM fine ranking**: Paper-level relevance scoring by strong model
- **Final ranking**: Weighted scoring (relevance/authority/recency/citation/diversity/redundancy) + MMR
- **Research graph**: Nodes, edges, clusters, and timeline visualization data

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + Redis)

### Setup

```bash
cd scholarrace

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start PostgreSQL (with pgvector) and Redis
docker-compose up -d

# Run database migrations
alembic upgrade head
```

### Running Tests

```bash
# All tests (uses SQLite + fakeredis, no external dependencies)
APP_ENV=test python -m pytest

# Specific test file
APP_ENV=test python -m pytest tests/test_pipeline_integration.py -v
```

### Running the API

```bash
# Start the server
uvicorn app.main:app --reload

# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer architecture survey", "max_results": 10}'

# Feedback
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "...", "paper_id": "...", "rating": 5, "is_relevant": true}'
```

### Running Experiments

```bash
# Run a single strategy
python -m scripts.run_experiment --query "machine learning" --strategy thompson_full

# Run all 6 strategies
python -m scripts.run_experiment --query "machine learning" --all-strategies --output results.csv

# Run ablation study A-H
python -m scripts.run_ablation --query "machine learning" --output ablation.csv
```

## Architecture

```
User Query
    |
    v
1. Query Understanding (LLM -> SearchQuery)
    |
    v
2. Multi-Agent Generation (Qwen + DeepSeek + GLM -> CandidateQueries)
    |
    v
3. Strong Judge (independent model -> JudgeResults)
    |
    v
4. Thompson Budget Allocation (Beta sampling -> per-model budget)
    |
    v
5. Multi-Source Retrieval (arXiv + Mocks -> Papers)
    |
    v
6. Citation Expansion (depth=1 -> expanded Papers)
    |
    v
7. Deduplication (DOI > arXiv > S2 > title+year)
    |
    v
8. Embedding Coarse Ranking (FakeEncoder -> Top-K)
    |
    v
9. LLM Paper Judging (strong model -> PaperJudgeResults)
    |
    v
10-11. Final Ranking (weighted + MMR -> PaperWithScores)
    |
    v
12. Research Graph (nodes/edges/clusters/timeline)
    |
    v
13-14. Result Assembly + Metrics
    |
    v
SearchResult
```

## Project Structure

```
scholarrace/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Pydantic Settings
│   ├── api/                 # API routes (search, feedback, health)
│   ├── models/              # Pydantic models (Paper, Query, Candidate, etc.)
│   ├── agents/              # LLM providers + agents (Qwen/DeepSeek/GLM/Judge)
│   ├── query/               # Query parser + decomposer
│   ├── bandit/              # Thompson Sampling
│   ├── retrieval/           # Search providers (arXiv real, others mock)
│   ├── citation/            # Citation expansion
│   ├── embedding/           # FakeEncoder + reranker
│   ├── ranking/             # Authority, MMR, FinalRanker
│   ├── graph/               # Research graph builder
│   ├── storage/             # SQLAlchemy + Redis
│   ├── pipeline/            # SearchPipeline orchestrator
│   └── utils/               # HTTP client, observability
├── tests/                   # 338 tests
├── scripts/                 # Experiment + ablation scripts
├── migrations/              # Alembic migrations
├── docker-compose.yml       # PostgreSQL + Redis
├── requirements.txt
└── .env.example
```

## Configuration

All configuration is in `app/config.py` (Pydantic BaseSettings), read from `.env`:

| Key | Default | Description |
|-----|---------|-------------|
| `APP_ENV` | development | test/dev/production mode |
| `DATABASE_URL` | postgresql+asyncpg://... | PostgreSQL connection |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `EMBEDDING_DIM` | 256 | FakeEncoder vector dimension |
| `EMBEDDING_TOP_K` | 100 | Coarse ranking cutoff |
| `FINAL_TOP_K` | 10 | Final paper count |
| `MMR_LAMBDA` | 0.7 | MMR relevance/diversity trade-off |
| `THOMPSON_TOTAL_BUDGET` | 50 | Total search budget |
| `THOMPSON_EXPLORATION_FLOOR` | 0.10 | Min budget per model |
| `THOMPSON_BATCH_SIZE` | 16 | Batch size for judge |
| `CITATION_EXPANSION_TOP_N` | 5 | Papers to expand citations for |
| `W_RELEVANCE` | 0.35 | Final ranking weight |
| `W_AUTHORITY` | 0.20 | Final ranking weight |
| `W_RECENCY` | 0.15 | Final ranking weight |
| `W_CITATION` | 0.10 | Final ranking weight |
| `W_DIVERSITY` | 0.10 | Final ranking weight |
| `W_REDUNDANCY` | 0.10 | Final ranking weight |

## Strategies & Ablation

### 6 Strategies

| Strategy | Description |
|----------|-------------|
| `single` | Single-model query generation (no multi-agent) |
| `multi` | Multi-agent generation without Thompson |
| `random` | Random budget allocation |
| `greedy` | Greedy budget allocation |
| `thompson` | Thompson Sampling budget allocation |
| `thompson_full` | Full pipeline (Thompson + citation + embedding + MMR) |

### 8-Level Ablation (A-H)

| Level | Description |
|-------|-------------|
| A | Baseline: single agent, no enhancements |
| B | + Multi-agent query generation |
| C | + Random budget allocation |
| D | + Greedy budget allocation |
| E | + Thompson Sampling |
| F | + Citation expansion |
| G | + Embedding reranking |
| H | + MMR diversity (full pipeline) |

## Key Design Decisions

1. **LLM Adapter Pattern**: All LLM providers use `openai` SDK with `AsyncOpenAI(base_url=...)`. MockLLMProvider returns deterministic JSON for testing.
2. **Thompson Sampling**: Real `np.random.beta()` sampling, fractional update (`alpha += reward, beta += 1 - reward`), exploration floor >= 10%.
3. **Judge Isolation**: Strong model never evaluates its own candidates.
4. **Batch Processing**: Paper judge processes in batches of 16.
5. **Deduplication Chain**: DOI -> arXiv ID -> S2 ID -> normalized(title+year).
6. **Failure Isolation**: Single provider failure doesn't block the pipeline.
7. **Test-First**: 338 tests, all run offline with SQLite + fakeredis.

## License

MIT
