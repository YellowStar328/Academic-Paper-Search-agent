"""Minimal FastAPI application entry point.

Routes will be registered progressively in later steps.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="ScholarRace",
    description="Academic search multi-agent system",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
