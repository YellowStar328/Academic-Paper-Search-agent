"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Database
    database_url: str = (
        "postgresql+asyncpg://scholar:scholar@localhost:5432/scholarrace"
    )
    database_url_test: str = "sqlite+aiosqlite:///:memory:"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM Providers
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4-plus"

    strong_model_api_key: str = ""
    strong_model_base_url: str = "https://api.deepseek.com/v1"
    strong_model_name: str = "deepseek-reasoner"

    # Search Sources
    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_max_results: int = 50
    arxiv_timeout: int = 30

    # Semantic Scholar
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1/paper/search"
    semantic_scholar_api_key: str = ""
    semantic_scholar_max_results: int = 50
    semantic_scholar_timeout: int = 30

    # OpenAlex
    openalex_base_url: str = "https://api.openalex.org/works"
    openalex_email: str = ""
    openalex_max_results: int = 50
    openalex_timeout: int = 30

    # CrossRef
    crossref_base_url: str = "https://api.crossref.org/works"
    crossref_email: str = ""
    crossref_max_results: int = 50
    crossref_timeout: int = 30

    # DBLP
    dblp_base_url: str = "https://dblp.org/search/publ/api"
    dblp_max_results: int = 50
    dblp_timeout: int = 30

    # Embedding
    embedding_top_k: int = 100
    embedding_dim: int = 256
    embedding_backend: Literal["fake", "api"] = "fake"

    # Ranking Weights
    w_relevance: float = 0.35
    w_authority: float = 0.15
    w_recency: float = 0.15
    w_citation: float = 0.15
    w_diversity: float = 0.15
    w_redundancy: float = 0.05
    mmr_lambda: float = 0.7
    final_top_k: int = 20

    # Thompson Sampling
    thompson_exploration_floor: float = 0.10
    thompson_batch_size: int = 16
    thompson_initial_alpha: float = 1.0
    thompson_initial_beta: float = 1.0
    thompson_total_budget: int = 50

    # Citation Expansion
    citation_expansion_depth: int = 1
    citation_expansion_top_n: int = 10

    # Retrieval
    retrieval_max_results_per_source: int = 50

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def effective_database_url(self) -> str:
        if self.is_test:
            return self.database_url_test
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
