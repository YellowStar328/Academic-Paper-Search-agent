"""Thompson Sampling bandit state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BetaArm:
    """A single Beta(alpha, beta) arm for Thompson Sampling.

    Uses fractional reward updates: alpha += reward, beta += (1 - reward).
    """

    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        """Sample theta from Beta(alpha, beta)."""
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, reward: float) -> None:
        """Update arm with fractional reward (0.0 to 1.0)."""
        reward = max(0.0, min(1.0, reward))
        self.alpha += reward
        self.beta += 1.0 - reward

    def mean(self) -> float:
        """Posterior mean."""
        total = self.alpha + self.beta
        if total == 0:
            return 0.5
        return self.alpha / total

    def reset(self) -> None:
        self.alpha = 1.0
        self.beta = 1.0


@dataclass
class BanditState:
    """Thompson Sampling state for multiple models.

    Keyed by (model, domain, query_type).
    """

    arms: dict[tuple[str, str, str], BetaArm] = field(default_factory=dict)
    initial_alpha: float = 1.0
    initial_beta: float = 1.0
    exploration_floor: float = 0.10

    def get_arm(self, model: str, domain: str, query_type: str) -> BetaArm:
        """Get or create a BetaArm for the given key."""
        key = (model, domain, query_type)
        if key not in self.arms:
            self.arms[key] = BetaArm(
                alpha=self.initial_alpha, beta=self.initial_beta
            )
        return self.arms[key]

    def sample_all(self, models: list[str], domain: str, query_type: str) -> dict[str, float]:
        """Sample theta for each model.

        Returns {model: theta}.
        """
        return {m: self.get_arm(m, domain, query_type).sample() for m in models}

    def allocate_budget(
        self,
        models: list[str],
        domain: str,
        query_type: str,
        total_budget: int = 10,
    ) -> dict[str, int]:
        """Allocate integer budget across models using Thompson Sampling.

        Applies exploration floor: every model gets at least
        exploration_floor fraction of the total budget.
        """
        if not models or total_budget <= 0:
            return {m: 0 for m in models}

        n = len(models)
        thetas = self.sample_all(models, domain, query_type)
        total_theta = sum(thetas.values())
        if total_theta == 0:
            # Uniform fallback
            raw_alloc = {m: 1.0 / n for m in models}
        else:
            raw_alloc = {m: t / total_theta for m, t in thetas.items()}

        # Apply exploration floor
        floor_frac = self.exploration_floor
        if floor_frac * n > 1.0:
            # If floor * n > 1, reduce floor so it sums to 1
            floor_frac = 1.0 / n

        # Linear blend: floor_uniform + (1 - floor*n) * raw
        # Each model gets at least floor_frac, rest allocated proportionally
        floor_alloc = {m: floor_frac for m in models}
        remaining_frac = max(0.0, 1.0 - floor_frac * n)
        blended = {
            m: floor_alloc[m] + remaining_frac * raw_alloc.get(m, 0.0)
            for m in models
        }

        # Normalize and convert to integers (largest remainder method)
        raw_int = {m: blended[m] * total_budget for m in models}
        allocated = {m: int(raw_int[m]) for m in models}
        remainder = total_budget - sum(allocated.values())

        # Distribute remainder to models with largest fractional parts
        frac_parts = {
            m: raw_int[m] - allocated[m] for m in models
        }
        for m in sorted(frac_parts, key=frac_parts.get, reverse=True)[:remainder]:
            allocated[m] += 1

        return allocated

    def update(
        self,
        model: str,
        domain: str,
        query_type: str,
        reward: float,
    ) -> None:
        """Update the arm for the given key with a fractional reward."""
        arm = self.get_arm(model, domain, query_type)
        arm.update(reward)

    def reset(self) -> None:
        for arm in self.arms.values():
            arm.reset()


@dataclass
class SourceBanditState:
    """Bandit state for search sources (source x domain success rate)."""

    arms: dict[tuple[str, str], BetaArm] = field(default_factory=dict)

    def get_arm(self, source: str, domain: str) -> BetaArm:
        key = (source, domain)
        if key not in self.arms:
            self.arms[key] = BetaArm()
        return self.arms[key]

    def sample(self, source: str, domain: str) -> float:
        return self.get_arm(source, domain).sample()

    def update(self, source: str, domain: str, reward: float) -> None:
        self.get_arm(source, domain).update(reward)
