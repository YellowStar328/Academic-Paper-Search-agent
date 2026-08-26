"""Query understanding: parse natural language into structured SearchQuery.

This module uses an LLM to transform a user's natural language research topic
into a structured SearchQuery object, separating hard constraints (year, venue,
open access) from the semantic core (the distilled intent).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.agents.base import LLMProvider
from app.agents.mock import MockLLMProvider
from app.config import get_settings
from app.models.query import (
    HardFilter,
    LogicOperator,
    QueryIntent,
    SearchOptions,
    SearchQuery,
)

QUERY_PARSE_SYSTEM_PROMPT = """You are a query understanding module for an academic search system.

Your task: Parse the user's research topic into a structured query object.

Extract:
- semantic_core: The distilled semantic intent (what are they really looking for?)
- domain: Academic domain (cs, physics, biology, chemistry, medicine, math, general)
- intent: One of survey, comparison, method, application, definition, recent, reproduction
- sub_queries: 2-4 decomposed sub-queries
- keywords: Key terms for search
- hard_filters: Hard constraints (year_start, year_end, venue, open_access_only, min_citations, has_code, language)
- logic: Logic operators (AND, OR, NOT) as strings

Return JSON: {
  "semantic_core": "...",
  "domain": "cs",
  "intent": "survey",
  "sub_queries": ["...", "..."],
  "keywords": ["...", "..."],
  "hard_filters": {"year_start": 2020, "open_access_only": false},
  "logic": ["AND", "OR"]
}"""

QUERY_PARSE_USER_TEMPLATE = """Parse the following research topic:

Topic: {topic}

If options are provided, incorporate them:
Options: {options_json}"""


class QueryParser:
    """Parse natural language queries into structured SearchQuery objects.

    Token-saving strategy: rule-based parsing is the default path (0 STRONG
    calls). The LLM is only invoked when ``use_llm=True`` is explicitly
    requested by the caller (e.g., for complex multi-constraint queries).
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.model_name = "query_parser"
        self.last_token_usage: int = 0

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            settings = get_settings()
            if settings.is_test or not settings.strong_model_api_key:
                self._provider = MockLLMProvider(model_name="query_parser")
            else:
                from app.agents.base import create_strong_judge_provider

                self._provider = create_strong_judge_provider()
        return self._provider

    async def parse(
        self,
        topic: str,
        options: Optional[SearchOptions] = None,
        use_llm: bool = False,
    ) -> SearchQuery:
        """Parse a natural language topic into a structured SearchQuery.

        By default uses rule-based parsing (0 STRONG tokens). Set
        ``use_llm=True`` to use the LLM, which costs ~800 tokens but
        may produce richer semantic decomposition for complex queries.
        """
        if options is None:
            options = SearchOptions()

        # Default: rule-based parsing (0 STRONG calls)
        if not use_llm:
            return self._rule_based_parse(topic, options)

        # Optional: LLM-based parsing
        prompt = QUERY_PARSE_USER_TEMPLATE.format(
            topic=topic,
            options_json=options.model_dump_json(),
        )

        response = await self.provider.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt=QUERY_PARSE_SYSTEM_PROMPT,
            response_schema={"type": "json_object"},
        )
        self.last_token_usage = response.token_usage

        if response.success:
            try:
                data = json.loads(response.content)
                return self._build_search_query(topic, data, options)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback: rule-based parsing
        return self._rule_based_parse(topic, options)

    def _build_search_query(
        self,
        topic: str,
        data: dict,
        options: SearchOptions,
    ) -> SearchQuery:
        """Build a SearchQuery from LLM-parsed data."""
        # Parse hard filters
        hf_data = data.get("hard_filters", {})
        hard_filters = HardFilter(
            year_start=hf_data.get("year_start"),
            year_end=hf_data.get("year_end"),
            venue=hf_data.get("venue"),
            open_access_only=hf_data.get("open_access_only", False),
            min_citations=hf_data.get("min_citations"),
            fields_of_study=hf_data.get("fields_of_study", []),
            has_code=hf_data.get("has_code", False),
            language=hf_data.get("language"),
        )

        # Parse intent
        intent_str = data.get("intent", "survey")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.SURVEY

        # Parse logic operators
        logic_strs = data.get("logic", [])
        logic = []
        for ls in logic_strs:
            try:
                logic.append(LogicOperator(ls))
            except ValueError:
                logic.append(LogicOperator.AND)

        return SearchQuery(
            original_query=topic,
            semantic_core=data.get("semantic_core", topic),
            domain=data.get("domain", "general"),
            intent=intent,
            sub_queries=data.get("sub_queries", []),
            hard_filters=hard_filters,
            logic=logic,
            keywords=data.get("keywords", []),
            options=options,
        )

    def _rule_based_parse(self, topic: str, options: SearchOptions) -> SearchQuery:
        """Fallback rule-based parsing when LLM is unavailable."""
        # Extract year ranges (e.g., "after 2020", "2020-2024", "since 2019")
        year_start = None
        year_end = None

        # "after YYYY" or "since YYYY"
        match = re.search(r"(?:after|since|from)\s+(\d{4})", topic, re.IGNORECASE)
        if match:
            year_start = int(match.group(1))

        # "before YYYY" or "until YYYY"
        match = re.search(r"(?:before|until|to)\s+(\d{4})", topic, re.IGNORECASE)
        if match:
            year_end = int(match.group(1))

        # "YYYY-YYYY" range
        match = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", topic)
        if match:
            year_start = int(match.group(1))
            year_end = int(match.group(2))

        # Detect domain
        domain = self._detect_domain(topic)

        # Detect intent
        intent = self._detect_intent(topic)

        # Extract keywords
        keywords = self._extract_keywords(topic)

        # Generate sub-queries
        sub_queries = self._generate_sub_queries(topic, intent)

        return SearchQuery(
            original_query=topic,
            semantic_core=topic,
            domain=domain,
            intent=intent,
            sub_queries=sub_queries,
            hard_filters=HardFilter(year_start=year_start, year_end=year_end),
            keywords=keywords,
            options=options,
        )

    def _detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        domains = {
            "cs": ["computer", "software", "algorithm", "machine learning", "deep learning",
                   "neural", "programming", "compiler", "database", "nlp", "computer vision"],
            "physics": ["physics", "quantum", "particle", "relativity", "thermodynamic"],
            "biology": ["biology", "genetic", "protein", "cell", "organism", "genome", "evolution"],
            "chemistry": ["chemistry", "molecular", "reaction", "catalyst", "synthesis"],
            "medicine": ["medical", "clinical", "disease", "patient", "treatment", "diagnosis"],
            "math": ["mathematics", "theorem", "proof", "algebra", "topology", "geometry"],
        }
        for domain, keywords in domains.items():
            if any(k in text_lower for k in keywords):
                return domain
        return "general"

    def _detect_intent(self, text: str) -> QueryIntent:
        text_lower = text.lower()
        if any(w in text_lower for w in ["compare", "comparison", "versus", "vs", "difference between"]):
            return QueryIntent.COMPARISON
        if any(w in text_lower for w in ["method", "algorithm", "how to", "implement", "approach"]):
            return QueryIntent.METHOD
        if any(w in text_lower for w in ["application", "apply", "use case", "deploy", "real-world"]):
            return QueryIntent.APPLICATION
        if any(w in text_lower for w in ["what is", "definition", "explain", "introduction"]):
            return QueryIntent.DEFINITION
        if any(w in text_lower for w in ["recent", "latest", "new", "2024", "2025", "state of the art"]):
            return QueryIntent.RECENT
        if any(w in text_lower for w in ["reproduce", "replicate", "reimplementation"]):
            return QueryIntent.REPRODUCTION
        return QueryIntent.SURVEY

    def _extract_keywords(self, text: str) -> list[str]:
        stop_words = {
            "the", "this", "that", "with", "from", "your", "have", "been",
            "about", "what", "how", "for", "are", "was", "were", "will",
            "can", "could", "should", "would", "their", "these", "those",
            "after", "before", "since", "until", "between", "among",
        }
        words = re.findall(r"[a-z]{3,}", text.lower())
        keywords = []
        for w in words:
            if w not in stop_words and w not in keywords:
                keywords.append(w)
        return keywords[:8]

    def _generate_sub_queries(self, topic: str, intent: QueryIntent) -> list[str]:
        """Generate basic sub-queries from the topic."""
        sub_queries = [topic]

        if intent == QueryIntent.SURVEY:
            sub_queries.append(f"{topic} survey")
            sub_queries.append(f"{topic} review")
        elif intent == QueryIntent.COMPARISON:
            sub_queries.append(f"{topic} comparison")
            sub_queries.append(f"{topic} evaluation")
        elif intent == QueryIntent.METHOD:
            sub_queries.append(f"{topic} method")
            sub_queries.append(f"{topic} algorithm")
        elif intent == QueryIntent.RECENT:
            sub_queries.append(f"{topic} recent advances")
            sub_queries.append(f"{topic} state of the art")

        return sub_queries[:4]
