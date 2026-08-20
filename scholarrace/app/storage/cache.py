"""Redis cache wrapper with fallback to fakeredis for testing."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]

try:
    import fakeredis
except ImportError:
    fakeredis = None  # type: ignore[assignment]

from app.config import get_settings


class Cache:
    """Async Redis cache wrapper. Falls back to fakeredis in test mode."""

    def __init__(self, redis_url: Optional[str] = None, is_test: bool = False):
        self._is_test = is_test
        self._redis = None
        self._redis_url = redis_url
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        if self._is_test:
            if fakeredis is not None:
                self._redis = fakeredis.FakeAsyncRedis()
            self._connected = True
            return
        if aioredis is not None and self._redis_url:
            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
            self._connected = True

    async def disconnect(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self._connected = False

    async def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            await self.connect()
        if self._redis is None:
            return None
        data = await self._redis.get(key)
        if data is None:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if not self._connected:
            await self.connect()
        if self._redis is None:
            return
        serialized = json.dumps(value, default=str)
        await self._redis.set(key, serialized, ex=ttl)

    async def delete(self, key: str) -> None:
        if not self._connected:
            await self.connect()
        if self._redis is None:
            return
        await self._redis.delete(key)

    async def exists(self, key: str) -> bool:
        if not self._connected:
            await self.connect()
        if self._redis is None:
            return False
        return bool(await self._redis.exists(key))

    async def flush(self) -> None:
        if not self._connected:
            await self.connect()
        if self._redis is not None:
            await self._redis.flushdb()

    def make_key(self, *parts: str) -> str:
        """Build a cache key from components."""
        return ":".join(str(p) for p in parts)


# Global cache instance
_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        settings = get_settings()
        _cache = Cache(redis_url=settings.redis_url, is_test=settings.is_test)
    return _cache


async def reset_cache() -> None:
    """Reset cache — useful for tests."""
    global _cache
    if _cache is not None:
        await _cache.disconnect()
    _cache = None
