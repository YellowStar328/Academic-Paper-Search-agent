"""Source-level Bandit for search source selection.

Tracks success rate per (source, domain) and helps prioritize
which search sources to query first.
"""

from __future__ import annotations

from app.models.bandit import SourceBanditState


class SourceBanditManager:
    """Manages search source prioritization via Thompson Sampling."""

    def __init__(self):
        self.state = SourceBanditState()

    def get_source_priorities(
        self,
        sources: list[str],
        domain: str,
    ) -> list[str]:
        """Return sources sorted by sampled theta (descending).

        Sources with higher sampled success rates are queried first.
        """
        scored = [(s, self.state.sample(s, domain)) for s in sources]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    def update_source(
        self,
        source: str,
        domain: str,
        reward: float,
    ) -> None:
        """Update source success rate.

        Reward can be binary (1.0 for success, 0.0 for failure) or
        fractional (e.g., fraction of useful papers returned).
        """
        self.state.update(source, domain, reward)

    def get_source_confidence(
        self,
        source: str,
        domain: str,
    ) -> float:
        """Get the posterior mean success rate for a source."""
        return self.state.get_arm(source, domain).mean()

    def reset(self) -> None:
        """Reset all source arms."""
        for arm in self.state.arms.values():
            arm.reset()
