"""CitationExpander — depth=1 citation/reference expansion.

For high-value papers (determined by a selection criterion), the expander
fetches their citing papers and referenced papers via the SearchProvider's
``get_citations`` / ``get_references`` methods.  The expanded papers are then
merged back into the candidate pool and deduplicated by PaperIdentityResolver.

Depth=1 means we only expand one level: we do not recursively expand the
citations of citations.  This keeps the API call count bounded while still
surfacing important related work.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.models.paper import Paper, PaperList
from app.retrieval.base import SearchProvider, filter_papers_by_year
from app.retrieval.resolver import PaperIdentityResolver

logger = logging.getLogger(__name__)


class CitationExpander:
    """Expand citations and references for high-value papers (depth=1).

    Parameters
    ----------
    providers
        List of search providers to use for citation/reference lookup.
    max_papers_to_expand
        Maximum number of top papers to expand (ranked by citation_count).
    max_citations_per_paper
        Maximum citations to fetch per paper.
    max_references_per_paper
        Maximum references to fetch per paper.
    resolver
        Optional PaperIdentityResolver for deduplication.  If ``None``, a
        new one is created.
    """

    def __init__(
        self,
        providers: list[SearchProvider],
        max_papers_to_expand: Optional[int] = None,
        max_citations_per_paper: Optional[int] = None,
        max_references_per_paper: Optional[int] = None,
        resolver: Optional[PaperIdentityResolver] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ):
        settings = get_settings()
        self._providers = providers
        self._max_expand = max_papers_to_expand or settings.citation_expansion_top_n
        self._max_citations = max_citations_per_paper or 20
        self._max_references = max_references_per_paper or 20
        self._resolver = resolver or PaperIdentityResolver()
        self._year_start = year_start
        self._year_end = year_end

    def set_year_range(
        self, year_start: Optional[int], year_end: Optional[int]
    ) -> None:
        """Update the publication year range for filtering expanded papers."""
        self._year_start = year_start
        self._year_end = year_end

    async def expand(self, papers: list[Paper]) -> list[Paper]:
        """Expand citations and references for top papers (depth=1).

        Returns the combined list of original + expanded papers, deduplicated.
        """
        if not papers:
            return []

        # Select top papers by citation_count
        top_papers = sorted(
            papers, key=lambda p: p.citation_count, reverse=True
        )[: self._max_expand]

        expanded: list[Paper] = []

        for paper in top_papers:
            for provider in self._providers:
                try:
                    # Pass the Paper object so providers can access real IDs
                    citations = await provider.get_citations(
                        paper, max_results=self._max_citations
                    )
                    expanded.extend(citations.papers)
                except Exception as e:
                    logger.warning(
                        f"Citation expansion failed for paper {paper.paper_id} "
                        f"via {getattr(provider, 'source_name', 'unknown')}: {e}"
                    )

                try:
                    references = await provider.get_references(
                        paper, max_results=self._max_references
                    )
                    expanded.extend(references.papers)
                except Exception as e:
                    logger.warning(
                        f"Reference expansion failed for paper {paper.paper_id} "
                        f"via {getattr(provider, 'source_name', 'unknown')}: {e}"
                    )

        # Merge original + expanded, then deduplicate
        all_papers = list(papers) + expanded
        deduped = self._resolver.resolve(all_papers)

        # Filter expanded papers by year range (if set) so that citation
        # expansion doesn't introduce out-of-range papers.
        deduped = filter_papers_by_year(
            deduped, self._year_start, self._year_end
        )

        logger.info(
            f"CitationExpander: expanded {len(top_papers)} papers, "
            f"collected {len(expanded)} new papers, "
            f"after dedup+year filter: {len(deduped)} total "
            f"(was {len(papers)})"
        )

        return deduped

    async def expand_paper_list(self, paper_list: PaperList) -> PaperList:
        """Expand and deduplicate a PaperList."""
        expanded = await self.expand(paper_list.papers)
        return PaperList(papers=expanded, source=paper_list.source)
