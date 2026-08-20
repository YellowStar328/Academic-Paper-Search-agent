"""PaperIdentityResolver — deduplication via canonical identity keys.

Deduplication priority chain: DOI > arXiv ID > S2 ID > OpenAlex ID > PubMed ID > title+year.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.paper import Paper, PaperIdentity, PaperList, normalize_title

logger = logging.getLogger(__name__)


class PaperIdentityResolver:
    """Resolves duplicate papers by canonical identity keys.

    When two papers share the same identity_key, they are merged:
    - Metadata from both is combined (non-empty fields take priority)
    - Citation/reference lists are unioned
    """

    def resolve(self, papers: list[Paper]) -> list[Paper]:
        """Deduplicate a list of papers by identity key.

        Returns a new list with duplicates merged.
        """
        if not papers:
            return []

        seen: dict[str, Paper] = {}  # identity_key -> merged Paper

        for paper in papers:
            key = paper.identity_key()
            if not key:
                # No identity — generate a unique key from paper_id
                key = f"paper_id:{paper.paper_id}"

            if key in seen:
                seen[key] = self._merge_papers(seen[key], paper)
            else:
                seen[key] = paper

        result = list(seen.values())
        logger.debug(
            f"PaperIdentityResolver: {len(papers)} -> {len(result)} papers "
            f"(removed {len(papers) - len(result)} duplicates)"
        )
        return result

    def resolve_paper_list(self, paper_list: PaperList) -> PaperList:
        """Deduplicate a PaperList."""
        deduped = self.resolve(paper_list.papers)
        return PaperList(papers=deduped, source=paper_list.source)

    def _merge_papers(self, primary: Paper, secondary: Paper) -> Paper:
        """Merge two papers, preferring non-empty fields from either."""
        # Prefer the paper with more complete metadata
        # Title: prefer longer title (more descriptive)
        if len(secondary.title) > len(primary.title):
            title = secondary.title
        else:
            title = primary.title

        # Abstract: prefer longer
        if len(secondary.abstract) > len(primary.abstract):
            abstract = secondary.abstract
        else:
            abstract = primary.abstract

        # Authors: union
        authors = list(dict.fromkeys(primary.authors + secondary.authors))

        # Year: prefer the one that exists
        year = primary.year or secondary.year

        # Venue: prefer non-empty
        venue = primary.venue or secondary.venue

        # DOIs, IDs: prefer non-empty (check both top-level and identity)
        doi = primary.doi or secondary.doi or primary.identity.doi or secondary.identity.doi
        arxiv_id = primary.arxiv_id or secondary.arxiv_id or primary.identity.arxiv_id or secondary.identity.arxiv_id
        s2_id = primary.semantic_scholar_id or secondary.semantic_scholar_id or primary.identity.semantic_scholar_id or secondary.identity.semantic_scholar_id
        openalex_id = primary.openalex_id or secondary.openalex_id or primary.identity.openalex_id or secondary.identity.openalex_id
        pubmed_id = primary.pubmed_id or secondary.pubmed_id or primary.identity.pubmed_id or secondary.identity.pubmed_id

        # Citation count: take max
        citation_count = max(primary.citation_count, secondary.citation_count)
        reference_count = max(primary.reference_count, secondary.reference_count)

        # Fields of study: union
        fields = list(dict.fromkeys(primary.fields_of_study + secondary.fields_of_study))

        # Keywords: union
        keywords = list(dict.fromkeys(primary.keywords + secondary.keywords))

        # References/citations: union
        references = list(dict.fromkeys(primary.references + secondary.references))
        citations = list(dict.fromkeys(primary.citations + secondary.citations))

        # URLs: prefer non-empty
        url = primary.url or secondary.url
        pdf_url = primary.pdf_url or secondary.pdf_url

        # Source: prefer the one with more data (or combine)
        source = primary.source

        # Publication date: prefer the one that exists
        pub_date = primary.publication_date or secondary.publication_date

        merged_identity = PaperIdentity(
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=s2_id,
            openalex_id=openalex_id,
            pubmed_id=pubmed_id,
            normalized_title=normalize_title(title),
            year=year,
        )

        return Paper(
            paper_id=primary.paper_id,  # keep primary ID
            identity=merged_identity,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=venue,
            publication_date=pub_date,
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=s2_id,
            openalex_id=openalex_id,
            pubmed_id=pubmed_id,
            url=url,
            pdf_url=pdf_url,
            citation_count=citation_count,
            reference_count=reference_count,
            fields_of_study=fields,
            keywords=keywords,
            source=source,
            references=references,
            citations=citations,
        )
