"""HTTP client utilities with retry, backoff, and circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreaker:
    """Simple circuit breaker for external API calls.

    States: CLOSED (normal), OPEN (failing, reject fast), HALF_OPEN (test).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half_open

    @property
    def state(self) -> str:
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    def can_proceed(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._last_failure_time is None:
                return True
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half_open"
                return True
            return False
        # half_open
        return True


async def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (httpx.HTTPError, asyncio.TimeoutError),
    **kwargs,
) -> Any:
    """Retry a coroutine with exponential backoff.

    Args:
        func: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries (seconds).
        max_delay: Maximum delay cap (seconds).
        exceptions: Tuple of exception types to catch and retry.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise
            delay = min(initial_delay * (2 ** attempt), max_delay)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore[misc]


class HttpClient:
    """HTTP client with retry, backoff, and circuit breaker."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def get(self, url: str, params: Optional[dict] = None, **kwargs) -> httpx.Response:
        """GET request with retry and circuit breaker."""
        if not self.circuit_breaker.can_proceed():
            raise CircuitBreakerOpen(f"Circuit breaker is open for {url}")

        async def _do_get():
            client = await self.get_client()
            return await client.get(url, params=params, **kwargs)

        try:
            response = await retry_with_backoff(
                _do_get, max_retries=self.max_retries
            )
            if response.status_code < 400:
                self.circuit_breaker.record_success()
            elif response.status_code >= 500:
                self.circuit_breaker.record_failure()
            # 4xx (including 429) don't trip circuit breaker
            return response
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    async def post(self, url: str, json: Optional[dict] = None, **kwargs) -> httpx.Response:
        """POST request with retry and circuit breaker."""
        if not self.circuit_breaker.can_proceed():
            raise CircuitBreakerOpen(f"Circuit breaker is open for {url}")

        async def _do_post():
            client = await self.get_client()
            return await client.post(url, json=json, **kwargs)

        try:
            response = await retry_with_backoff(
                _do_post, max_retries=self.max_retries
            )
            if response.status_code < 500:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()
            return response
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
