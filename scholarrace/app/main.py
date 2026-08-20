"""FastAPI application entry point.

Registers all API routes and provides a pipeline factory function.
The pipeline is lazily initialized on first request.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI

from app.api import feedback, search

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ScholarRace",
    description="Academic search multi-agent system",
    version="0.1.0",
)

# Register routers
app.include_router(search.router, tags=["search"])
app.include_router(feedback.router, tags=["feedback"])

# Global pipeline instance (lazily initialized)
_pipeline = None


def get_pipeline():
    """Get the global pipeline instance (or None if not initialized)."""
    return _pipeline


def set_pipeline(pipeline) -> None:
    """Set the global pipeline instance (for testing or startup)."""
    global _pipeline
    _pipeline = pipeline


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
