"""Tests for QueryParser and QueryDecomposer."""

import pytest

from app.agents.mock import MockLLMProvider
from app.query.parser import QueryParser
from app.query.decomposer import QueryDecomposer
from app.models.query import (
    HardFilter,
    QueryIntent,
    SearchOptions,
    SearchQuery,
    LogicOperator,
)


class TestQueryParser:
    @pytest.mark.asyncio
    async def test_parse_with_mock_llm(self):
        parser = QueryParser(provider=MockLLMProvider())
        query = await parser.parse("transformer architectures for NLP")
        assert isinstance(query, SearchQuery)
        assert query.original_query == "transformer architectures for NLP"
        assert query.semantic_core  # non-empty
        assert query.domain  # non-empty
        assert len(query.sub_queries) > 0

    @pytest.mark.asyncio
    async def test_parse_includes_options(self):
        parser = QueryParser(provider=MockLLMProvider())
        options = SearchOptions(top_k=50, mode="human_review")
        query = await parser.parse("machine learning survey", options=options)
        assert query.options.top_k == 50
        assert query.options.mode == "human_review"

    @pytest.mark.asyncio
    async def test_rule_based_fallback_year_range(self):
        """Test that rule-based parsing extracts year ranges."""
        parser = QueryParser(provider=MockLLMProvider())
        query = await parser.parse("deep learning papers from 2020-2024")
        # Should have some year extraction
        assert query.original_query == "deep learning papers from 2020-2024"

    @pytest.mark.asyncio
    async def test_rule_based_fallback_after_year(self):
        parser = QueryParser(provider=MockLLMProvider())
        result = parser._rule_based_parse("papers after 2019", SearchOptions())
        assert result.hard_filters.year_start == 2019

    @pytest.mark.asyncio
    async def test_rule_based_fallback_year_range_dash(self):
        parser = QueryParser(provider=MockLLMProvider())
        result = parser._rule_based_parse("papers from 2018-2023", SearchOptions())
        assert result.hard_filters.year_start == 2018
        assert result.hard_filters.year_end == 2023

    @pytest.mark.asyncio
    async def test_rule_based_fallback_before_year(self):
        parser = QueryParser(provider=MockLLMProvider())
        result = parser._rule_based_parse("papers before 2020", SearchOptions())
        assert result.hard_filters.year_end == 2020

    def test_detect_domain_cs(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_domain("machine learning and neural networks") == "cs"
        assert parser._detect_domain("deep learning algorithms") == "cs"

    def test_detect_domain_physics(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_domain("quantum physics research") == "physics"

    def test_detect_domain_biology(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_domain("protein folding and genetics") == "biology"

    def test_detect_domain_medicine(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_domain("clinical trial for disease treatment") == "medicine"

    def test_detect_domain_general(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_domain("some random topic") == "general"

    def test_detect_intent_survey(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_intent("overview of transformers") == QueryIntent.SURVEY

    def test_detect_intent_comparison(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_intent("compare CNN vs RNN") == QueryIntent.COMPARISON

    def test_detect_intent_method(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_intent("how to implement attention mechanism") == QueryIntent.METHOD

    def test_detect_intent_recent(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_intent("recent advances in LLM 2024") == QueryIntent.RECENT

    def test_detect_intent_definition(self):
        parser = QueryParser(provider=MockLLMProvider())
        assert parser._detect_intent("what is reinforcement learning") == QueryIntent.DEFINITION

    def test_extract_keywords(self):
        parser = QueryParser(provider=MockLLMProvider())
        keywords = parser._extract_keywords("graph neural networks for node classification")
        assert len(keywords) > 0
        assert "graph" in keywords
        assert "neural" in keywords
        assert "networks" in keywords

    def test_extract_keywords_stops_words(self):
        parser = QueryParser(provider=MockLLMProvider())
        keywords = parser._extract_keywords("the transformer architecture was introduced")
        assert "the" not in keywords
        assert "was" not in keywords

    def test_generate_sub_queries_survey(self):
        parser = QueryParser(provider=MockLLMProvider())
        subs = parser._generate_sub_queries("transformer survey", QueryIntent.SURVEY)
        assert len(subs) <= 4
        assert any("survey" in s.lower() or "review" in s.lower() for s in subs)

    def test_generate_sub_queries_comparison(self):
        parser = QueryParser(provider=MockLLMProvider())
        subs = parser._generate_sub_queries("CNN vs RNN", QueryIntent.COMPARISON)
        assert any("comparison" in s.lower() or "evaluation" in s.lower() for s in subs)

    @pytest.mark.asyncio
    async def test_failed_llm_falls_back_to_rules(self):
        """When LLM fails, should fall back to rule-based parsing."""
        from app.agents.base import LLMResponse

        class FailingProvider:
            model_name = "fail"

            async def generate(self, prompt, temperature=0.7, response_schema=None, system_prompt=None):
                return LLMResponse(content="", model="fail", success=False, error="error")

        parser = QueryParser(provider=FailingProvider())
        query = await parser.parse("machine learning survey after 2020")
        assert query.original_query == "machine learning survey after 2020"
        assert query.hard_filters.year_start == 2020
        assert query.domain == "cs"

    @pytest.mark.asyncio
    async def test_build_search_query_from_llm_data(self):
        parser = QueryParser(provider=MockLLMProvider())
        data = {
            "semantic_core": "UCB value methods in RL",
            "domain": "cs",
            "intent": "survey",
            "sub_queries": ["UCB value methods", "non-stationary RL"],
            "keywords": ["UCB", "RL", "value"],
            "hard_filters": {"year_start": 2020, "open_access_only": True},
            "logic": ["AND", "OR"],
        }
        sq = parser._build_search_query("original", data, SearchOptions())
        assert sq.semantic_core == "UCB value methods in RL"
        assert sq.domain == "cs"
        assert sq.intent == QueryIntent.SURVEY
        assert sq.hard_filters.year_start == 2020
        assert sq.hard_filters.open_access_only is True
        assert len(sq.logic) == 2
        assert sq.logic[0] == LogicOperator.AND


class TestQueryDecomposer:
    def test_get_sources_for_cs(self):
        d = QueryDecomposer()
        sources = d.get_sources_for_domain("cs")
        assert "arxiv" in sources
        assert "semantic_scholar" in sources

    def test_get_sources_for_medicine(self):
        d = QueryDecomposer()
        sources = d.get_sources_for_domain("medicine")
        assert "pubmed" in sources

    def test_get_sources_for_general(self):
        d = QueryDecomposer()
        sources = d.get_sources_for_domain("general")
        assert len(sources) >= 2

    def test_get_sources_unknown_domain(self):
        d = QueryDecomposer()
        sources = d.get_sources_for_domain("unknown_domain")
        assert len(sources) >= 2  # falls back to general

    def test_decompose_with_sub_queries(self):
        d = QueryDecomposer()
        query = SearchQuery(
            original_query="test",
            semantic_core="semantic test",
            sub_queries=["sub1", "sub2"],
        )
        queries = d.decompose(query)
        assert len(queries) >= 2
        assert "sub1" in queries
        assert "sub2" in queries

    def test_decompose_includes_semantic_core(self):
        d = QueryDecomposer()
        query = SearchQuery(
            original_query="test",
            semantic_core="semantic core",
            sub_queries=["sub1"],
        )
        queries = d.decompose(query)
        assert "semantic core" in queries

    def test_decompose_falls_back_to_original(self):
        d = QueryDecomposer()
        query = SearchQuery(
            original_query="original query",
            semantic_core="",
            sub_queries=[],
        )
        queries = d.decompose(query)
        assert "original query" in queries

    def test_decompose_expands_keywords(self):
        d = QueryDecomposer()
        query = SearchQuery(
            original_query="test",
            semantic_core="test",
            sub_queries=["sub1"],
            keywords=["keyword1", "keyword2", "keyword3"],
        )
        queries = d.decompose(query)
        # Should include keyword-based query
        assert any("keyword1" in q for q in queries)

    def test_build_source_query_arxiv(self):
        d = QueryDecomposer()
        result = d.build_source_query("machine learning", "arxiv")
        assert result == "machine learning"

    def test_build_source_query_pubmed(self):
        d = QueryDecomposer()
        result = d.build_source_query("cancer treatment", "pubmed")
        assert "[Title/Abstract]" in result

    def test_build_source_query_semantic_scholar(self):
        d = QueryDecomposer()
        result = d.build_source_query("neural networks", "semantic_scholar")
        assert result == "neural networks"

    def test_build_source_query_default(self):
        d = QueryDecomposer()
        result = d.build_source_query("test query", "unknown_source")
        assert result == "test query"

    def test_extract_logic_string(self):
        d = QueryDecomposer()
        query = SearchQuery(
            original_query="test",
            semantic_core="test",
            logic=[LogicOperator.AND],
        )
        assert d.extract_logic_string(query) == "AND"

    def test_extract_logic_string_empty(self):
        d = QueryDecomposer()
        query = SearchQuery(original_query="test", semantic_core="test")
        assert d.extract_logic_string(query) == "OR"

    def test_split_conjunction(self):
        d = QueryDecomposer()
        parts = d.split_conjunction("deep learning and neural networks")
        assert len(parts) == 2
        assert "deep learning" in parts
        assert "neural networks" in parts

    def test_split_conjunction_or(self):
        d = QueryDecomposer()
        parts = d.split_conjunction("CNN or RNN")
        assert len(parts) == 2

    def test_split_conjunction_plus(self):
        d = QueryDecomposer()
        parts = d.split_conjunction("transformer + attention")
        assert len(parts) == 2
