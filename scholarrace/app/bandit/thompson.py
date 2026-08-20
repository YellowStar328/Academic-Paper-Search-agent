"""Thompson Sampling manager for model budget allocation.

This module wraps the BanditState with:
- Persistence (load/save from ModelConfidenceRepository)
- Reward computation (from judge scores and paper relevance)
- Integration with the pipeline (allocate budget, update rewards)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.models.bandit import BanditState, BetaArm
from app.models.candidate import JudgeResult, PaperJudgeResult

logger = logging.getLogger(__name__)


class ThompsonSamplingManager:
    """Manages Thompson Sampling for multi-agent budget allocation.

    Flow:
    1. load_state(domain, query_type, models) — fetch from DB or init defaults
    2. allocate_budget(models, domain, query_type, total_budget) — sample Beta
    3. After pipeline runs, compute_rewards and update_state
    """

    def __init__(
        self,
        initial_alpha: Optional[float] = None,
        initial_beta: Optional[float] = None,
        exploration_floor: Optional[float] = None,
    ):
        settings = get_settings()
        self.initial_alpha = initial_alpha or settings.thompson_initial_alpha
        self.initial_beta = initial_beta or settings.thompson_initial_beta
        self.exploration_floor = exploration_floor or settings.thompson_exploration_floor
        self.state = BanditState(
            initial_alpha=self.initial_alpha,
            initial_beta=self.initial_beta,
            exploration_floor=self.exploration_floor,
        )

    def allocate_budget(
        self,
        models: list[str],
        domain: str,
        query_type: str,
        total_budget: int = 10,
    ) -> dict[str, int]:
        """Allocate search budget across models using Thompson Sampling.

        Each model gets at least exploration_floor fraction of total budget
        to prevent Model Collapse.
        """
        return self.state.allocate_budget(models, domain, query_type, total_budget)

    def sample_thetas(
        self, models: list[str], domain: str, query_type: str
    ) -> dict[str, float]:
        """Sample theta values for each model (for debugging/visualization)."""
        return self.state.sample_all(models, domain, query_type)

    def compute_reward_from_judge(
        self,
        judge_results: list[JudgeResult],
    ) -> dict[str, float]:
        """Compute average reward per model from judge results.

        Reward = average judge score for candidates proposed by that model.
        """
        if not judge_results:
            return {}

        model_rewards: dict[str, list[float]] = {}
        for jr in judge_results:
            model = jr.candidate.proposer_model
            if model not in model_rewards:
                model_rewards[model] = []
            model_rewards[model].append(jr.score)

        avg_rewards = {}
        for model, scores in model_rewards.items():
            avg_rewards[model] = sum(scores) / len(scores)

        return avg_rewards

    def compute_reward_from_papers(
        self,
        paper_results: list[PaperJudgeResult],
        model_paper_mapping: dict[str, list[str]],
    ) -> dict[str, float]:
        """Compute reward from paper judge results.

        Args:
            paper_results: List of paper judge results.
            model_paper_mapping: Maps model name to list of paper_ids
                that were retrieved using that model's queries.

        Returns:
            Dict of {model: avg_relevance_score}
        """
        if not paper_results or not model_paper_mapping:
            return {}

        # Create paper_id -> relevance_score map
        paper_scores = {r.paper_id: r.relevance_score for r in paper_results}

        model_rewards = {}
        for model, paper_ids in model_paper_mapping.items():
            scores = [
                paper_scores[pid] for pid in paper_ids if pid in paper_scores
            ]
            if scores:
                model_rewards[model] = sum(scores) / len(scores)
            else:
                model_rewards[model] = 0.5  # default

        return model_rewards

    def update_state(
        self,
        model: str,
        domain: str,
        query_type: str,
        reward: float,
    ) -> None:
        """Update the Beta arm for a model with fractional reward."""
        self.state.update(model, domain, query_type, reward)
        logger.debug(
            f"Thompson update: {model}/{domain}/{query_type} "
            f"reward={reward:.3f} alpha={self.state.get_arm(model, domain, query_type).alpha:.2f}"
        )

    def update_state_batch(
        self,
        model_rewards: dict[str, float],
        domain: str,
        query_type: str,
    ) -> None:
        """Update state for multiple models at once."""
        for model, reward in model_rewards.items():
            self.update_state(model, domain, query_type, reward)

    def get_confidence(
        self, model: str, domain: str, query_type: str
    ) -> float:
        """Get the posterior mean (confidence) for a model."""
        return self.state.get_arm(model, domain, query_type).mean()

    def get_all_confidences(
        self, models: list[str], domain: str, query_type: str
    ) -> dict[str, float]:
        """Get posterior means for all models."""
        return {m: self.get_confidence(m, domain, query_type) for m in models}

    def reset(self) -> None:
        """Reset all arms to initial values."""
        self.state.reset()

    async def load_from_db(
        self,
        models: list[str],
        domain: str,
        query_type: str,
        session=None,
    ) -> None:
        """Load confidence state from database.

        If no saved state exists, uses default alpha/beta.
        """
        if session is None:
            return

        from app.storage.repositories import ModelConfidenceRepository

        repo = ModelConfidenceRepository(session)
        for model in models:
            mc = await repo.get(model, domain, query_type)
            if mc is not None:
                arm = self.state.get_arm(model, domain, query_type)
                arm.alpha = mc.alpha
                arm.beta = mc.beta

    async def save_to_db(
        self,
        models: list[str],
        domain: str,
        query_type: str,
        session=None,
    ) -> None:
        """Save confidence state to database."""
        if session is None:
            return

        from app.models.agent import ModelConfidence
        from app.storage.repositories import ModelConfidenceRepository

        repo = ModelConfidenceRepository(session)
        for model in models:
            arm = self.state.get_arm(model, domain, query_type)
            total = arm.alpha + arm.beta
            avg_reward = arm.alpha / total if total > 0 else 0.5

            mc = ModelConfidence(
                model_name=model,
                domain=domain,
                query_type=query_type,
                alpha=arm.alpha,
                beta=arm.beta,
                total_runs=int(total - 2),  # subtract initial alpha+beta
                avg_reward=avg_reward,
            )
            await repo.upsert(mc)
