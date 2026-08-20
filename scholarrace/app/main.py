"""FastAPI application entry point.

Registers all API routes and provides a pipeline factory function.
The pipeline is automatically initialized on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.api import feedback, search
from app.config import get_settings

logger = logging.getLogger(__name__)


def create_pipeline():
    """Create a SearchPipeline with real or mock providers based on config.

    - LLM: uses real API if key is set, otherwise MockLLMProvider
    - Search: uses real arXiv provider (free, no key needed)
    - Thompson: enabled by default for budget allocation
    """
    from app.agents.qwen import QwenAgent
    from app.agents.deepseek import DeepSeekAgent
    from app.agents.glm import GLMAgent
    from app.agents.coordinator import MultiAgentCoordinator
    from app.agents.judge import PaperJudge, StrongJudge
    from app.bandit.thompson import ThompsonSamplingManager
    from app.citation.expansion import CitationExpander
    from app.embedding.encoder import FakeEncoder
    from app.pipeline.search_pipeline import SearchPipeline
    from app.query.parser import QueryParser
    from app.ranking.final_ranker import FinalRanker
    from app.retrieval.arxiv import ArxivProvider
    from app.retrieval.crossref import CrossrefProvider
    from app.retrieval.dblp import DblpProvider
    from app.retrieval.openalex import OpenAlexProvider
    from app.retrieval.semantic_scholar import SemanticScholarProvider

    settings = get_settings()

    # LLM provider — agents auto-fallback to Mock if no API key
    qwen = QwenAgent()
    deepseek = DeepSeekAgent()
    glm = GLMAgent()

    # QueryParser/Judge auto-resolve their own LLMProvider (None = lazy)
    parser = QueryParser(provider=None)
    coordinator = MultiAgentCoordinator(
        qwen_agent=qwen, deepseek_agent=deepseek, glm_agent=glm
    )
    strong_judge = StrongJudge(provider=None)
    paper_judge = PaperJudge(provider=None)

    # Real search providers (all free, no API key required)
    providers = [
        ArxivProvider(),
        SemanticScholarProvider(),
        OpenAlexProvider(),
        CrossrefProvider(),
        DblpProvider(),
    ]

    # Citation expander using the same providers
    citation_expander = CitationExpander(providers=providers)

    # Thompson Sampling
    thompson_manager = ThompsonSamplingManager()

    # Final ranker
    final_ranker = FinalRanker()

    return SearchPipeline(
        query_parser=parser,
        coordinator=coordinator,
        strong_judge=strong_judge,
        providers=providers,
        citation_expander=citation_expander,
        paper_judge=paper_judge,
        final_ranker=final_ranker,
        thompson_manager=thompson_manager,
        settings=settings,
    )


# Global pipeline instance
_pipeline = None


def get_pipeline():
    """Get the global pipeline instance (or None if not initialized)."""
    return _pipeline


def set_pipeline(pipeline) -> None:
    """Set the global pipeline instance (for testing or startup)."""
    global _pipeline
    _pipeline = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the search pipeline on application startup."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing search pipeline...")
        try:
            _pipeline = create_pipeline()
            logger.info("Search pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
    yield


app = FastAPI(
    title="ScholarRace",
    description="Academic search multi-agent system",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(search.router, tags=["search"])
app.include_router(feedback.router, tags=["feedback"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
