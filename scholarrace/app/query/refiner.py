"""Query refiner — uses a strong LLM to convert natural-language queries into
provider-specific search syntax.

- For **arXiv**: generates field-specific syntax like ``all:"large language model" AND ti:pretraining``
- For **Semantic Scholar**: generates concise keyword phrases like ``"data quality pretraining large language model"``
- Returns keywords list for general use

This ensures every provider receives properly distilled search terms instead of
raw natural-language sentences.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from app.agents.base import LLMProvider, create_strong_judge_provider
from app.agents.mock import MockLLMProvider
from app.config import get_settings
from app.models.query import SearchQuery
from app.models.candidate import CandidateQuery

logger = logging.getLogger(__name__)

REFINER_SYSTEM_PROMPT = """You are an expert at constructing academic search queries from natural-language questions.

Given a natural-language academic search question, your job is to:
1. Identify the core technical concepts, methods, and keywords.
2. Generate three outputs:
   a) **arxiv_query**: arXiv field-specific search syntax
   b) **s2_query**: a concise keyword phrase for Semantic Scholar (space-separated keywords, no boolean operators)
   c) **keywords**: the list of extracted keywords

### arXiv search syntax reference:
- all:"phrase"     — search in title, abstract, and comments
- ti:"phrase"      — search in title only
- abs:"phrase"     — search in abstract only
- au:"name"        — search in author name
- cat:cs.CL        — restrict to category (e.g. cs.CL, cs.AI, stat.ML)
- AND, OR, NOT     — boolean operators (must be uppercase)

### Semantic Scholar query format:
- A simple string of space-separated keywords, e.g. "data quality pretraining language model"
- No boolean operators, no field prefixes, no quotes
- 3-6 keywords, focused on technical terms

### Rules:
- Use 3-5 well-chosen keywords/phrases.
- Do NOT include conversational words ("give me", "find papers about", "show that").
- Focus on technical terms that would appear in paper titles and abstracts.

### CRITICAL — arXiv query must be BROAD, not restrictive:
- Use **at most ONE AND condition**. Prefer OR for synonyms.
- Good: `all:"ranking" OR all:"search ranking" OR all:"language model ranking"` (broad, high recall)
- Good: `all:"ranking" AND all:"search"` (one AND is OK)
- Bad: `all:"ranking" AND all:"search" AND all:"language model" AND all:"retrieval"` (too strict, returns 0-5 results)
- Use `all:""` with short individual keywords, not long phrases.
- Do NOT add category restrictions unless the query is very domain-specific.
- Prefer grouping synonyms with OR rather than narrowing with AND.
- Example pattern: `all:"concept A" OR all:"concept B" OR all:"concept C"`
- Count the ANDs: if you have more than 1 AND, replace extras with OR.

Return JSON: {"arxiv_query": "...", "s2_query": "...", "keywords": ["...", "..."], "reasoning": "..."}"""

REFINER_USER_TEMPLATE = """User question: {question}

Semantic core: {semantic_core}
Keywords already identified: {keywords}

Generate the three search query outputs."""


class QueryRefiner:
    """Uses an LLM to convert natural-language queries into provider-specific search syntax.

    Produces:
    - arXiv field-specific syntax (e.g. all:"data quality" AND all:"pretraining")
    - Semantic Scholar keyword phrase (e.g. "data quality pretraining language model")
    - Keywords list

    Token-saving strategy: by default uses a lightweight agent provider
    (Qwen) instead of STRONG, saving ~600 tokens per query × 5 queries
    = ~3000 STRONG tokens. Set ``use_strong=True`` to fall back to STRONG.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "query_refiner"
        self.last_token_usage: int = 0

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            settings = get_settings()
            if settings.is_test or not settings.strong_model_api_key:
                self._provider = MockLLMProvider(model_name="query_refiner")
            elif getattr(settings, "use_strong_refiner", False):
                # Explicit fallback to STRONG
                self._provider = create_strong_judge_provider()
            else:
                # Default: use a lightweight agent (Qwen) instead of STRONG
                from app.agents.qwen import QwenAgent

                self._provider = QwenAgent().provider
        return self._provider

    async def refine(self, search_query: SearchQuery) -> dict[str, str | list[str]]:
        """Refine a SearchQuery into provider-specific search syntax.

        Returns a dict with keys: "arxiv_query", "s2_query", "keywords".
        Falls back to the original query if LLM fails.
        """
        prompt = REFINER_USER_TEMPLATE.format(
            question=search_query.original_query,
            semantic_core=search_query.semantic_core,
            keywords=", ".join(search_query.keywords) if search_query.keywords else "none",
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=REFINER_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage += response.token_usage

        if not response.success:
            logger.warning("QueryRefiner LLM failed: %s", response.error)
            return {
                "arxiv_query": search_query.original_query,
                "s2_query": search_query.semantic_core or search_query.original_query,
                "keywords": search_query.keywords or [],
            }

        try:
            data = json.loads(response.content)
            arxiv_q = data.get("arxiv_query", "").strip()
            s2_q = data.get("s2_query", "").strip()
            kws = data.get("keywords", [])
            if not isinstance(kws, list):
                kws = [kws] if kws else []

            result = {
                "arxiv_query": arxiv_q or search_query.original_query,
                "s2_query": s2_q or (search_query.semantic_core or search_query.original_query),
                "keywords": kws,
            }
            logger.info(
                "QueryRefiner: '%s' -> arxiv='%s' s2='%s'",
                search_query.original_query[:60],
                result["arxiv_query"][:100],
                result["s2_query"][:100],
            )
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("QueryRefiner JSON parse failed: %s", e)
            return {
                "arxiv_query": search_query.original_query,
                "s2_query": search_query.semantic_core or search_query.original_query,
                "keywords": search_query.keywords or [],
            }

    async def refine_candidates(
        self,
        search_query: SearchQuery,
        candidates: list[CandidateQuery],
    ) -> list[dict[str, str | list[str]]]:
        """Refine the original query + candidate queries into provider-specific syntax.

        Returns a list of refined dicts (one per query), each with keys:
        "arxiv_query", "s2_query", "keywords".
        """
        # Collect unique queries to refine
        queries_to_refine = [(search_query.original_query, search_query.semantic_core, search_query.keywords)]
        for c in candidates:
            if c.query and c.query not in [q[0] for q in queries_to_refine]:
                queries_to_refine.append((c.query, c.query, []))

        # Refine all in parallel
        tasks = [self._refine_single(q, core, kws) for q, core, kws in queries_to_refine]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        refined: list[dict[str, str | list[str]]] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("QueryRefiner candidate %d failed: %s", i, r)
                q, core, kws = queries_to_refine[i]
                refined.append({
                    "arxiv_query": q,
                    "s2_query": core or q,
                    "keywords": kws,
                })
            else:
                refined.append(r)
        return refined

    async def _refine_single(
        self,
        query: str,
        semantic_core: str = "",
        keywords: list[str] | None = None,
    ) -> dict[str, str | list[str]]:
        """Refine a single query string into provider-specific syntax."""
        prompt = REFINER_USER_TEMPLATE.format(
            question=query,
            semantic_core=semantic_core or query,
            keywords=", ".join(keywords) if keywords else "none",
        )
        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=REFINER_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage += response.token_usage

        if not response.success:
            return {
                "arxiv_query": query,
                "s2_query": semantic_core or query,
                "keywords": keywords or [],
            }

        try:
            data = json.loads(response.content)
            arxiv_q = data.get("arxiv_query", "").strip()
            s2_q = data.get("s2_query", "").strip()
            kws = data.get("keywords", [])
            if not isinstance(kws, list):
                kws = [kws] if kws else []
            return {
                "arxiv_query": arxiv_q or query,
                "s2_query": s2_q or (semantic_core or query),
                "keywords": kws,
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "arxiv_query": query,
                "s2_query": semantic_core or query,
                "keywords": keywords or [],
            }
