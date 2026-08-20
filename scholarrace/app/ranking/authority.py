"""AuthorityScorer — multi-signal authority scoring.

Computes an authority score for a paper based on multiple signals:
- Citation count (normalized via log scale)
- Venue prestige (if available)
- Author h-index (if available)
- Source reliability

The score is a weighted combination in [0, 1].
"""

from __future__ import annotations

import math
from typing import Optional

from app.models.paper import Paper


class AuthorityScorer:
    """Multi-signal authority scorer.

    Parameters
    ----------
    max_citation_threshold
        Citation count at which the citation signal saturates (log-scale).
    venue_weight / citation_weight / author_weight / source_weight
        Weights for each signal.  Must sum to 1.0.
    """

    def __init__(
        self,
        max_citation_threshold: int = 1000,
        venue_weight: float = 0.3,
        citation_weight: float = 0.4,
        author_weight: float = 0.15,
        source_weight: float = 0.15,
    ):
        self._max_citations = max_citation_threshold
        self._venue_w = venue_weight
        self._citation_w = citation_weight
        self._author_w = author_weight
        self._source_w = source_weight

    def score_citation(self, paper: Paper) -> float:
        """Log-normalized citation score in [0, 1]."""
        if paper.citation_count <= 0:
            return 0.0
        # Log-scale: log(1 + citations) / log(1 + max)
        return min(
            math.log(1 + paper.citation_count)
            / math.log(1 + self._max_citations),
            1.0,
        )

    def score_venue(self, paper: Paper) -> float:
        """Venue prestige score in [0, 1].

        If the paper has no venue info, return a neutral 0.5.
        """
        venue = (paper.venue or "").lower().strip()
        if not venue:
            return 0.5

        # Tier list (simplified)
        top_tier = {
            "nature", "science", "cell", "neurips", "icml", "iclr",
            "cvpr", "acl", "emnlp", "aaai", "ijcai", "sigcomm",
            "sosp", "osdi", "sigmod", "vldb", "kdd", "www",
        }
        second_tier = {
            "workshop", "arxiv", "preprint", "tech report",
        }

        if any(t in venue for t in top_tier):
            return 0.9
        if any(t in venue for t in second_tier):
            return 0.3
        return 0.6  # generic conference/journal

    def score_authors(self, paper: Paper) -> float:
        """Author authority score in [0, 1].

        If no author info, return neutral 0.5.
        """
        if not paper.authors:
            return 0.5
        # Heuristic: papers with more authors tend to be larger collaborations
        # (slight signal), but we cap at a reasonable number.
        n = len(paper.authors)
        return min(0.5 + 0.05 * min(n, 10), 1.0)

    def score_source(self, source: str = "") -> float:
        """Source reliability score in [0, 1]."""
        source_lower = source.lower()
        if "arxiv" in source_lower:
            return 0.6
        if "semantic_scholar" in source_lower or "s2" in source_lower:
            return 0.8
        if "openalex" in source_lower:
            return 0.7
        if "pubmed" in source_lower:
            return 0.8
        if "crossref" in source_lower:
            return 0.7
        return 0.5

    def score(self, paper: Paper, source: str = "") -> float:
        """Combined authority score in [0, 1]."""
        citation = self.score_citation(paper)
        venue = self.score_venue(paper)
        authors = self.score_authors(paper)
        source_s = self.score_source(source or paper.source or "")

        combined = (
            self._citation_w * citation
            + self._venue_w * venue
            + self._author_w * authors
            + self._source_w * source_s
        )
        return min(combined, 1.0)
