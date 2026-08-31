"""Mock LLM Provider for testing.

Returns deterministic responses based on the prompt content,
allowing full pipeline testing without real API keys.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

from app.agents.base import LLMProvider, LLMResponse


class MockLLMProvider:
    """Deterministic mock LLM provider for testing.

    Returns JSON responses that look like real LLM output.
    The response is seeded by the prompt content, making it
    deterministic for the same prompt.
    """

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        response_schema: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        start = time.time()
        # Simulate small latency
        latency = 10.0 + (hash(prompt) % 50)

        # Determine what type of response to generate based on prompt content
        # Combine system_prompt + prompt for matching
        combined = ""
        if system_prompt:
            combined += system_prompt + "\n"
        combined += prompt
        content = self._generate_response(combined)
        latency = (time.time() - start) * 1000 + latency

        return LLMResponse(
            content=content,
            model=self.model_name,
            latency_ms=latency,
            token_usage=len(combined) // 4 + 50,
            success=True,
        )

    def _generate_response(self, prompt: str) -> str:
        """Generate a deterministic mock response based on prompt content."""
        prompt_lower = prompt.lower()

        # Strong model reviewing agent reports — must return "final_papers"
        if "final_papers" in prompt_lower or "review these reports" in prompt_lower:
            # Extract paper_id values from the reports JSON in the prompt
            paper_ids = re.findall(r'"paper_id"\s*:\s*"([^"]+)"', prompt)
            final_papers = [
                {
                    "paper_id": pid,
                    "final_relevance_score": 0.8,
                    "final_authority_score": 0.7,
                    "final_reasoning": "Relevant and authoritative.",
                    "endorsed_by": ["qwen", "deepseek", "glm"],
                }
                for pid in paper_ids
            ]
            return json.dumps({"final_papers": final_papers})

        # Paper relevance evaluation (most specific — must check before query judge)
        if "paper" in prompt_lower and ("relevance" in prompt_lower or "abstract" in prompt_lower or "paper title" in prompt_lower):
            return json.dumps({
                "relevance_score": 0.75,
                "authority_score": 0.8,
                "reasoning": "The paper is relevant to the query.",
                "key_findings": ["finding 1", "finding 2"],
            })

        # Query understanding / parsing
        if "semantic_core" in prompt_lower or "query understanding" in prompt_lower:
            return json.dumps({
                "semantic_core": self._extract_topic(prompt),
                "domain": self._detect_domain(prompt),
                "intent": "survey",
                "sub_queries": [
                    f"{self._extract_topic(prompt)} recent advances",
                    f"{self._extract_topic(prompt)} methods comparison",
                ],
                "keywords": self._extract_keywords(prompt),
                "hard_filters": {"year_start": 2020},
            })

        # Candidate query generation — system prompts mention "candidates" and "strategist"
        if "candidates" in prompt_lower or "sub-quer" in prompt_lower or "generate search quer" in prompt_lower or "search strategist" in prompt_lower:
            topic = self._extract_topic(prompt)
            # Extract real keywords (not stopwords) from the topic for better arXiv matching
            kws = self._extract_keywords(topic)
            # Build 2 candidate queries: keyword-focused + broader
            kw_query = " ".join(kws[:5]) if kws else topic
            return json.dumps({
                "candidates": [
                    {
                        "query": kw_query,
                        "rationale": "Keyword-focused search for precise matching",
                        "keywords": kws[:5],
                        "logic": "AND",
                    },
                    {
                        "query": f"{kws[0]} {kws[1]} survey" if len(kws) >= 2 else f"{topic} survey",
                        "rationale": "Broader survey perspective",
                        "keywords": [kws[0], kws[1]] if len(kws) >= 2 else [topic],
                        "logic": "OR",
                    },
                ]
            })

        # Query judge evaluation — mentions "evaluate" + "candidate" + "coverage/specificity"
        if "evaluate" in prompt_lower and ("coverage" in prompt_lower or "specificity" in prompt_lower or "proposer model" in prompt_lower or "score" in prompt_lower):
            return json.dumps({
                "score": 0.85,
                "reasoning": "The candidate query covers the main intent well.",
                "coverage": 0.8,
                "specificity": 0.7,
                "novelty": 0.6,
            })

        # Default: return a generic JSON
        return json.dumps({
            "response": "mock response",
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
        })

    def _extract_topic(self, prompt: str) -> str:
        """Try to extract the main topic from the prompt."""
        # Look for "Topic:" pattern (most reliable)
        match = re.search(r"(?:topic|query|question)\s*:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Look for quoted text
        match = re.search(r'"([^"]+)"', prompt)
        if match:
            return match.group(1)
        # Look for "query:" pattern
        match = re.search(r"query[:\s]+(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Look for topic after "for" or "about"
        match = re.search(r"(?:for|about|on)\s+(.+?)(?:\n|\.|,|$)", prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "machine learning"

    def _detect_domain(self, prompt: str) -> str:
        """Detect academic domain from prompt."""
        prompt_lower = prompt.lower()
        domains = {
            "cs": ["computer", "software", "algorithm", "machine learning", "deep learning", "neural"],
            "physics": ["physics", "quantum", "particle", "relativity"],
            "biology": ["biology", "genetic", "protein", "cell", "organism"],
            "chemistry": ["chemistry", "molecular", "reaction", "catalyst"],
            "medicine": ["medical", "clinical", "disease", "patient", "treatment"],
            "math": ["mathematics", "theorem", "proof", "algebra", "topology"],
        }
        for domain, keywords in domains.items():
            if any(k in prompt_lower for k in keywords):
                return domain
        return "general"

    def _extract_keywords(self, prompt: str) -> list[str]:
        """Extract keywords from prompt."""
        # Extended stopwords
        stop = {
            "the", "this", "that", "with", "from", "your", "have", "been",
            "could", "list", "provide", "mention", "works", "work", "paper",
            "papers", "some", "related", "about", "which", "what", "studies",
            "discuss", "effects", "mechanisms", "research", "study", "article",
            "articles", "find", "describe", "explain", "introduce", "novel",
        }
        words = re.findall(r"[a-z]{4,}", prompt.lower())
        seen = []
        for w in words:
            if w not in seen and w not in stop:
                seen.append(w)
        return seen[:8]
