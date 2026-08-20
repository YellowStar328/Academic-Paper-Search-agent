"""LLM Provider Protocol and base infrastructure.

All LLM providers use the OpenAI-compatible interface via `openai` SDK.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from app.config import get_settings


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    content: str
    model: str
    latency_ms: float = 0.0
    token_usage: int = 0
    success: bool = True
    error: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers (Qwen, DeepSeek, GLM, Strong Judge)."""

    model_name: str

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        response_schema: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            temperature: Sampling temperature (0.0-1.0).
            response_schema: Optional JSON schema for structured output.
            system_prompt: Optional system message.
        """
        ...


class BaseOpenAIProvider:
    """Base class for OpenAI-compatible LLM providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazily create the AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key or "dummy-key",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        response_schema: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text via OpenAI-compatible API."""
        start = time.time()
        client = self._get_client()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }

            if response_schema:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            latency = (time.time() - start) * 1000

            content = response.choices[0].message.content or ""
            token_usage = 0
            if response.usage:
                token_usage = response.usage.total_tokens

            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency,
                token_usage=token_usage,
                success=True,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency,
                success=False,
                error=str(e),
            )


def create_qwen_provider() -> BaseOpenAIProvider:
    """Create a Qwen (DashScope) LLM provider."""
    s = get_settings()
    return BaseOpenAIProvider(
        api_key=s.qwen_api_key,
        base_url=s.qwen_base_url,
        model=s.qwen_model,
    )


def create_deepseek_provider() -> BaseOpenAIProvider:
    """Create a DeepSeek LLM provider."""
    s = get_settings()
    return BaseOpenAIProvider(
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        model=s.deepseek_model,
    )


def create_glm_provider() -> BaseOpenAIProvider:
    """Create a GLM (Zhipu) LLM provider."""
    s = get_settings()
    return BaseOpenAIProvider(
        api_key=s.glm_api_key,
        base_url=s.glm_base_url,
        model=s.glm_model,
    )


def create_strong_judge_provider() -> BaseOpenAIProvider:
    """Create the strong model judge provider."""
    s = get_settings()
    return BaseOpenAIProvider(
        api_key=s.strong_model_api_key,
        base_url=s.strong_model_base_url,
        model=s.strong_model_name,
    )
