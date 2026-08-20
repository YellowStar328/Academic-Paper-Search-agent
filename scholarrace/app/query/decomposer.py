"""Query decomposer: helper for sub-query decomposition and domain refinement.

Splits complex queries into simpler sub-queries that can be dispatched
to different search sources.
"""

from __future__ import annotations

import re
from typing import Optional

from app.models.query import SearchQuery, LogicOperator


class QueryDecomposer:
    """Decompose a SearchQuery into search-source-specific sub-queries."""

    # Mapping from domain to preferred search sources
    DOMAIN_SOURCES = {
        "cs": ["arxiv", "semantic_scholar", "openalex"],
        "physics": ["arxiv", "semantic_scholar"],
        "biology": ["pubmed", "semantic_scholar", "openalex"],
        "medicine": ["pubmed", "semantic_scholar"],
        "chemistry": ["semantic_scholar", "crossref"],
        "math": ["arxiv", "semantic_scholar"],
        "general": ["semantic_scholar", "arxiv", "openalex"],
    }

    def get_sources_for_domain(self, domain: str) -> list[str]:
        """Get recommended search sources for a given domain."""
        return self.DOMAIN_SOURCES.get(domain, self.DOMAIN_SOURCES["general"])

    def decompose(self, query: SearchQuery) -> list[str]:
        """Decompose the query into source-specific search strings.

        This does NOT call LLM — it uses rules to transform the semantic core
        and sub-queries into source-optimized queries.
        """
        queries = list(query.sub_queries)
        if query.semantic_core and query.semantic_core not in queries:
            queries.insert(0, query.semantic_core)
        if not queries:
            queries = [query.original_query]

        # Expand keywords into additional queries using logic operators
        if query.keywords and len(queries) < 5:
            keyword_query = " ".join(query.keywords[:3])
            if keyword_query and keyword_query not in queries:
                queries.append(keyword_query)

        return queries

    def build_source_query(
        self,
        query: str,
        source: str,
        hard_filters_text: str = "",
    ) -> str:
        """Build a source-specific query string.

        Different sources have different query syntax:
        - arXiv: simple text search
        - Semantic Scholar: text + field filters
        - PubMed: MeSH terms + text
        """
        if source == "arxiv":
            # arXiv: all fields search, no complex syntax
            return query
        elif source == "semantic_scholar":
            # S2: text query
            return query
        elif source == "pubmed":
            # PubMed: add [Title/Abstract] field tag
            return f"{query}[Title/Abstract]"
        elif source == "openalex":
            # OpenAlex: text search
            return query
        elif source == "crossref":
            # Crossref: bibliographic search
            return query
        return query

    def extract_logic_string(self, query: SearchQuery) -> str:
        """Extract a logic operator string for complex queries."""
        if not query.logic:
            return "OR"
        return query.logic[0].value

    def split_conjunction(self, text: str) -> list[str]:
        """Split a query on conjunctions (and, or, +)."""
        parts = re.split(r"\s+(?:and|or|\+)\s+", text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]
