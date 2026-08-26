"""SearchProvider Protocol and base infrastructure."""

from __future__ import annotations

import abc
from typing import Optional, Protocol, runtime_checkable

from app.models.paper import Paper, PaperList
from app.models.query import SearchQuery
from app.utils.http_client import HttpClient


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for academic search providers."""

    source_name: str

    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search for papers matching the query.

        If year_start/year_end are provided, filter results to the given
        publication year range (inclusive).
        """
        ...

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by ID."""
        ...

    async def get_citations(self, paper_id: str, max_results: int = 50) -> PaperList:
        """Get papers that cite this paper."""
        ...

    async def get_references(self, paper_id: str, max_results: int = 50) -> PaperList:
        """Get papers referenced by this paper."""
        ...


class BaseSearchProvider(abc.ABC):
    """Base class for search providers."""

    def __init__(self, http_client: Optional[HttpClient] = None, timeout: float = 30.0):
        self._http_client = http_client or HttpClient(timeout=timeout)

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Return the source name identifier."""
        ...

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> PaperList:
        """Search for papers.

        If year_start/year_end are provided, filter results to the given
        publication year range (inclusive).
        """
        ...

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper. Override in subclasses if supported."""
        return None

    async def get_citations(self, paper, max_results: int = 50) -> PaperList:
        """Get papers that cite this paper. Accepts Paper or paper_id str."""
        return PaperList(source=self.source_name)

    async def get_references(self, paper, max_results: int = 50) -> PaperList:
        """Get papers referenced by this paper. Accepts Paper or paper_id str."""
        return PaperList(source=self.source_name)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http_client.close()


def filter_papers_by_year(
    papers: list[Paper],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> list[Paper]:
    """Filter a list of papers by publication year range (inclusive).

    Papers with ``year is None`` are dropped when a year filter is active,
    so that results strictly respect the user's year constraints.
    """
    if year_start is None and year_end is None:
        return papers
    result = []
    for p in papers:
        y = p.year
        if y is None:
            continue  # strict: drop papers with unknown year
        if year_start is not None and y < year_start:
            continue
        if year_end is not None and y > year_end:
            continue
        result.append(p)
    return result
