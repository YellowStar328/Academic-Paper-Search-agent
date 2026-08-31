"""ResearchGraph builder — constructs structured graph from paper data.

Builds a ResearchGraph from a list of papers (and optionally their scores)
WITHOUT calling any LLM.  The graph consists of:

- **Nodes**: one per paper (paper_id, title, year, cluster_id)
- **Edges**: citation/reference relationships between papers
- **Clusters**: grouping of papers by embedding similarity (k-means style)
- **Timeline**: papers grouped by publication year

Clustering uses the FakeEncoder for simplicity.  The number of clusters
is determined by the square root of the number of papers (capped at 10).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from app.config import get_settings
from app.embedding.encoder import EmbeddingEncoder, FakeEncoder, cosine_similarity
from app.models.paper import Paper
from app.models.result import (
    GraphCluster,
    GraphEdge,
    GraphNode,
    PaperWithScores,
    ResearchGraph,
    TimelineEntry,
)

logger = logging.getLogger(__name__)


class ResearchGraphBuilder:
    """Builds a ResearchGraph from papers and their relationships.

    Parameters
    ----------
    encoder
        Embedding encoder for clustering (default: FakeEncoder).
    max_clusters
        Maximum number of clusters (default: 10).
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        max_clusters: int = 10,
    ):
        self._encoder = encoder or FakeEncoder()
        self._max_clusters = max_clusters

    def build(
        self,
        papers: list[Paper],
        scored_papers: Optional[list[PaperWithScores]] = None,
    ) -> ResearchGraph:
        """Build a ResearchGraph from the given papers.

        Parameters
        ----------
        papers
            The final set of papers to include in the graph.
        scored_papers
            Optional scored papers (for richer metadata).  If provided,
            edges and clusters will be built on the paper set.
        """
        if not papers:
            return ResearchGraph()

        # Build nodes
        nodes = self._build_nodes(papers)

        # Build edges from citation/reference lists
        edges = self._build_edges(papers)

        # Build clusters via embedding similarity (k-means style)
        clusters = self._build_clusters(papers)

        # Build timeline from publication years
        timeline = self._build_timeline(papers)

        graph = ResearchGraph(
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            timeline=timeline,
        )

        logger.info(
            f"ResearchGraph: {len(nodes)} nodes, {len(edges)} edges, "
            f"{len(clusters)} clusters, {len(timeline)} timeline entries"
        )

        return graph

    def _build_nodes(self, papers: list[Paper]) -> list[GraphNode]:
        """Create a graph node for each paper."""
        nodes = []
        for paper in papers:
            nodes.append(
                GraphNode(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                )
            )
        return nodes

    def _build_edges(self, papers: list[Paper]) -> list[GraphEdge]:
        """Build citation/reference edges between papers.

        Each paper has ``references`` and ``citations`` lists containing
        paper_ids.  We create edges for references (paper → referenced paper)
        and citations (citing paper → paper).

        Only edges where both endpoints are in the paper set are included
        (internal edges).
        """
        paper_ids = {p.paper_id for p in papers}
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for paper in papers:
            # References: this paper references others
            for ref_id in paper.references:
                if ref_id in paper_ids and ref_id != paper.paper_id:
                    key = (paper.paper_id, ref_id, "reference")
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(
                            GraphEdge(
                                source_id=paper.paper_id,
                                target_id=ref_id,
                                edge_type="reference",
                                weight=1.0,
                            )
                        )

            # Citations: other papers cite this one
            for cite_id in paper.citations:
                if cite_id in paper_ids and cite_id != paper.paper_id:
                    key = (cite_id, paper.paper_id, "citation")
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(
                            GraphEdge(
                                source_id=cite_id,
                                target_id=paper.paper_id,
                                edge_type="citation",
                                weight=1.0,
                            )
                        )

        return edges

    def _build_clusters(self, papers: list[Paper]) -> list[GraphCluster]:
        """Cluster papers by embedding similarity using k-means.

        The number of clusters is min(sqrt(n), max_clusters).
        Each cluster is labeled by its centroid paper's title.
        """
        n = len(papers)
        if n <= 1:
            if n == 1:
                return [
                    GraphCluster(
                        cluster_id=0,
                        label=papers[0].title[:50],
                        paper_ids=[papers[0].paper_id],
                        centroid_title=papers[0].title,
                    )
                ]
            return []

        # Determine number of clusters
        k = min(max(int(math.sqrt(n)), 2), self._max_clusters, n)

        # Encode all papers
        texts = [f"{p.title} {p.abstract or ''}" for p in papers]
        vecs = np.array([self._encoder.encode(t) for t in texts])

        # Simple k-means clustering
        cluster_assignments = self._kmeans(vecs, k)

        # Build clusters
        clusters: list[GraphCluster] = []
        for cluster_id in range(k):
            member_indices = [
                i for i, c in enumerate(cluster_assignments) if c == cluster_id
            ]
            if not member_indices:
                continue

            # Find centroid (paper closest to cluster mean)
            member_vecs = vecs[member_indices]
            centroid_vec = member_vecs.mean(axis=0)
            centroid_idx = member_indices[
                int(np.argmax([
                    cosine_similarity(centroid_vec, member_vecs[j])
                    for j in range(len(member_indices))
                ]))
            ]

            centroid_paper = papers[centroid_idx]
            member_paper_ids = [papers[i].paper_id for i in member_indices]

            clusters.append(
                GraphCluster(
                    cluster_id=cluster_id,
                    label=centroid_paper.title[:50],
                    paper_ids=member_paper_ids,
                    centroid_title=centroid_paper.title,
                )
            )

        return clusters

    async def label_clusters_with_llm(
        self,
        clusters: list[GraphCluster],
        papers: list[Paper],
    ) -> None:
        """Generate semantic topic labels for clusters via LLM.

        Replaces the centroid-title heuristic with an LLM-generated
        concise topic label.  Falls back to centroid title on failure.
        Should be called after ``build()`` from an async context.
        """
        settings = get_settings()
        if settings.is_test or not settings.qwen_api_key:
            return

        try:
            from app.agents.qwen import QwenAgent

            provider = QwenAgent().provider
        except Exception as e:
            logger.warning(f"Cluster LLM labeling skipped: {e}")
            return

        paper_map = {p.paper_id: p for p in papers}

        for cluster in clusters:
            member_papers = [
                paper_map[pid]
                for pid in cluster.paper_ids
                if pid in paper_map
            ]
            if not member_papers:
                continue

            titles_str = "\n".join(
                f"- {p.title}" for p in member_papers[:10]
            )
            prompt = (
                f"You are a research taxonomy expert.\n"
                f"Below are {len(member_papers)} paper titles from a cluster:\n"
                f"{titles_str}\n\n"
                f"Generate a concise topic label (3-8 words) that captures "
                f"the common theme. Respond with ONLY the label, no explanation."
            )

            try:
                resp = await provider.generate(
                    prompt=prompt,
                    system_prompt="You are a research taxonomy expert.",
                    temperature=0.2,
                )
                label = resp.content.strip().strip('"').strip("'")
                if label and len(label) <= 80:
                    cluster.label = label
            except Exception as e:
                logger.debug(f"Cluster {cluster.cluster_id} label failed: {e}")
                continue

    def _kmeans(self, vecs: np.ndarray, k: int, max_iter: int = 20) -> list[int]:
        """Simple k-means clustering.

        Returns a list of cluster assignments (one per paper).
        """
        n = len(vecs)
        dim = vecs.shape[1]

        # Initialize centroids: pick k spread-out points
        # Use first k papers as initial centroids (deterministic)
        if k >= n:
            return list(range(n))  # each paper is its own cluster

        # Pick initial centroids using k-means++ style (deterministic)
        centroids = [vecs[0].copy()]
        for _ in range(1, k):
            # Pick the point farthest from existing centroids
            max_dist = -1.0
            best_idx = 0
            for i in range(n):
                min_sim = max(
                    cosine_similarity(vecs[i], c) for c in centroids
                )
                dist = 1.0 - min_sim
                if dist > max_dist:
                    max_dist = dist
                    best_idx = i
            centroids.append(vecs[best_idx].copy())

        centroids = np.array(centroids)

        # Iterate
        assignments = [0] * n
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                # Assign to nearest centroid
                best_cluster = 0
                best_sim = -1.0
                for j in range(k):
                    sim = cosine_similarity(vecs[i], centroids[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = j
                if assignments[i] != best_cluster:
                    assignments[i] = best_cluster
                    changed = True

            if not changed:
                break

            # Update centroids
            for j in range(k):
                members = vecs[np.array(assignments) == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)

        return assignments

    def _build_timeline(self, papers: list[Paper]) -> list[TimelineEntry]:
        """Build a timeline of papers grouped by year."""
        year_map: dict[int, list[str]] = {}
        for paper in papers:
            if paper.year is not None:
                year_map.setdefault(paper.year, []).append(paper.paper_id)

        timeline = [
            TimelineEntry(
                year=year,
                paper_ids=sorted(pids),
                count=len(pids),
            )
            for year, pids in sorted(year_map.items())
        ]

        return timeline
