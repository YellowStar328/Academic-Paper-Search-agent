"""GLM agent provider — focuses on applications, datasets, and synonyms."""

from __future__ import annotations

import json
from typing import Optional

from app.agents.base import BaseOpenAIProvider, LLMProvider, create_glm_provider
from app.agents.mock import MockLLMProvider
from app.config import get_settings
from app.models.query import SearchQuery
from app.models.candidate import CandidateQuery

GLM_SYSTEM_PROMPT = """You are an academic search strategist specializing in applications, datasets, and synonyms.

Your task: Given a user's research topic, generate 2-3 search queries that focus on:
- Application domains and use cases
- Datasets and benchmarks
- Synonyms and alternative terminology
- Interdisciplinary connections

Return JSON: {"candidates": [{"query": "...", "rationale": "...", "keywords": [...], "logic": "OR|AND"}]}"""

GLM_USER_TEMPLATE = """Research topic: {topic}
Semantic core: {semantic_core}
Domain: {domain}

Generate search queries focusing on applications, datasets, and synonyms."""


class GLMAgent:
    """Query generation agent using GLM model, focused on applications."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "glm"
        self.last_token_usage: int = 0

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            settings = get_settings()
            if settings.is_test or not settings.glm_api_key:
                self._provider = MockLLMProvider(model_name="glm")
            else:
                self._provider = create_glm_provider()
        return self._provider

    async def generate_queries(self, query: SearchQuery) -> list[CandidateQuery]:
        prompt = GLM_USER_TEMPLATE.format(
            topic=query.original_query,
            semantic_core=query.semantic_core,
            domain=query.domain,
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.8,
            system_prompt=GLM_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage = response.token_usage

        if not response.success:
            return []

        try:
            data = json.loads(response.content)
            candidates_data = data.get("candidates", [])
            return [
                CandidateQuery(
                    query=c.get("query", ""),
                    proposer_model=self.model_name,
                    rationale=c.get("rationale", ""),
                    keywords=c.get("keywords", []),
                    logic=c.get("logic", "OR"),
                )
                for c in candidates_data
            ]
        except (json.JSONDecodeError, KeyError):
            return []
