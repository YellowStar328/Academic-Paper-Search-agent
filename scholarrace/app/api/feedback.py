"""POST /api/feedback — user feedback endpoint.

Records user feedback (relevance ratings, suggestions) for pipeline
improvement and Thompson Sampling reward updates.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    """Request body for POST /api/feedback."""

    request_id: str = Field(..., description="Original search request ID")
    paper_id: str = Field(..., description="Paper being rated")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    is_relevant: bool = Field(..., description="Is this paper relevant to the query?")


class FeedbackResponse(BaseModel):
    """Response body for POST /api/feedback."""

    status: str
    request_id: str
    paper_id: str
    recorded: bool


@router.post("/api/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback for a search result."""
    logger.info(
        f"Feedback received: request_id={request.request_id}, "
        f"paper_id={request.paper_id}, rating={request.rating}, "
        f"relevant={request.is_relevant}"
    )

    # Convert rating to reward signal (0-1 scale)
    reward = (request.rating - 1) / 4.0  # 1->0.0, 5->1.0

    # TODO: persist feedback and update bandit rewards

    return FeedbackResponse(
        status="ok",
        request_id=request.request_id,
        paper_id=request.paper_id,
        recorded=True,
    )
