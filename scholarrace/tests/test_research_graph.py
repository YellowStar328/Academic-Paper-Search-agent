"""Tests for ResearchGraphBuilder."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.embedding.encoder import FakeEncoder
from app.graph.research_graph import ResearchGraphBuilder
from app.models.paper import Paper, PaperIdentity
from app.models.result import ResearchGraph


def make_paper(
    title: str = "Test Paper",
    year: int = 2024,
    references: list[str] | None = None,
    citations: list[str] | None = None,
    abstract: str = "Test abstract",
    paper_id: str | None = None,
) -> Paper:
    return Paper(
        paper_id=paper_id or str(uuid4()),
        identity=PaperIdentity(normalized_title=title.lower().replace(" ", "")),
        title=title,
        abstract=abstract,
        year=year,
        references=references or [],
        citations=citations or [],
        source="test",
    )


class TestResearchGraphBuilder:
    """Tests for ResearchGraphBuilder."""

    def test_empty_input(self):
        builder = ResearchGraphBuilder()
        graph = builder.build([])
        assert isinstance(graph, ResearchGraph)
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.clusters == []
        assert graph.timeline == []

    def test_single_paper(self):
        builder = ResearchGraphBuilder()
        paper = make_paper("Single Paper", year=2024)
        graph = builder.build([paper])
        assert len(graph.nodes) == 1
        assert graph.nodes[0].paper_id == paper.paper_id
        assert graph.nodes[0].title == "Single Paper"
        assert len(graph.clusters) == 1
        assert len(graph.timeline) == 1

    def test_nodes_built_correctly(self):
        builder = ResearchGraphBuilder()
        p1 = make_paper("Paper One", year=2023)
        p2 = make_paper("Paper Two", year=2024)
        graph = builder.build([p1, p2])
        assert len(graph.nodes) == 2
        titles = {n.title for n in graph.nodes}
        assert "Paper One" in titles
        assert "Paper Two" in titles

    def test_nodes_include_year(self):
        builder = ResearchGraphBuilder()
        paper = make_paper("Year Paper", year=2021)
        graph = builder.build([paper])
        assert graph.nodes[0].year == 2021

    def test_edges_from_references(self):
        p1 = make_paper("Source Paper")
        p2 = make_paper("Target Paper")
        p1.references = [p2.paper_id]
        builder = ResearchGraphBuilder()
        graph = builder.build([p1, p2])
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source_id == p1.paper_id
        assert edge.target_id == p2.paper_id
        assert edge.edge_type == "reference"

    def test_edges_from_citations(self):
        p1 = make_paper("Cited Paper")
        p2 = make_paper("Citing Paper")
        p1.citations = [p2.paper_id]
        builder = ResearchGraphBuilder()
        graph = builder.build([p1, p2])
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.source_id == p2.paper_id
        assert edge.target_id == p1.paper_id
        assert edge.edge_type == "citation"

    def test_edges_only_internal(self):
        p1 = make_paper("Paper One")
        p1.references = ["external-paper-id"]
        p2 = make_paper("Paper Two")
        builder = ResearchGraphBuilder()
        graph = builder.build([p1, p2])
        assert len(graph.edges) == 0

    def test_no_self_edges(self):
        p1 = make_paper("Self-ref")
        p1.references = [p1.paper_id]
        builder = ResearchGraphBuilder()
        graph = builder.build([p1])
        assert len(graph.edges) == 0

    def test_no_duplicate_edges(self):
        p1 = make_paper("Source")
        p2 = make_paper("Target")
        p1.references = [p2.paper_id, p2.paper_id]
        builder = ResearchGraphBuilder()
        graph = builder.build([p1, p2])
        assert len(graph.edges) == 1

    def test_clusters_group_similar_papers(self):
        builder = ResearchGraphBuilder(encoder=FakeEncoder(dimension=64))
        papers_a = [
            make_paper(f"Transformer Attention Model {i}", abstract="attention")
            for i in range(5)
        ]
        papers_b = [
            make_paper(f"Graph Neural Network {i}", abstract="gnn graph")
            for i in range(5)
        ]
        all_papers = papers_a + papers_b
        graph = builder.build(all_papers)
        assert len(graph.clusters) >= 2

    def test_clusters_have_paper_ids(self):
        builder = ResearchGraphBuilder()
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        graph = builder.build(papers)
        total = sum(len(c.paper_ids) for c in graph.clusters)
        assert total == len(papers)

    def test_clusters_have_labels(self):
        builder = ResearchGraphBuilder()
        papers = [make_paper("Important Research Paper")]
        graph = builder.build(papers)
        assert graph.clusters[0].label != ""

    def test_clusters_have_centroid_title(self):
        builder = ResearchGraphBuilder()
        papers = [make_paper("Centroid Paper Title")]
        graph = builder.build(papers)
        assert graph.clusters[0].centroid_title == "Centroid Paper Title"

    def test_timeline_grouped_by_year(self):
        builder = ResearchGraphBuilder()
        papers = [
            make_paper("P2022", year=2022),
            make_paper("P2023", year=2023),
            make_paper("P2023b", year=2023),
            make_paper("P2024", year=2024),
        ]
        graph = builder.build(papers)
        assert len(graph.timeline) == 3
        years = [t.year for t in graph.timeline]
        assert years == sorted(years)

    def test_timeline_count(self):
        builder = ResearchGraphBuilder()
        papers = [
            make_paper("A", year=2023),
            make_paper("B", year=2023),
            make_paper("C", year=2023),
        ]
        graph = builder.build(papers)
        assert len(graph.timeline) == 1
        assert graph.timeline[0].count == 3

    def test_timeline_sorted_by_year(self):
        builder = ResearchGraphBuilder()
        papers = [
            make_paper("New", year=2024),
            make_paper("Old", year=2010),
            make_paper("Mid", year=2017),
        ]
        graph = builder.build(papers)
        years = [t.year for t in graph.timeline]
        assert years == [2010, 2017, 2024]

    def test_paper_without_year_excluded_from_timeline(self):
        builder = ResearchGraphBuilder()
        p1 = make_paper("WithYear", year=2024)
        p2 = make_paper("NoYear", year=None)
        graph = builder.build([p1, p2])
        assert len(graph.timeline) == 1
        assert graph.timeline[0].year == 2024

    def test_max_clusters_limit(self):
        builder = ResearchGraphBuilder(max_clusters=3)
        papers = [make_paper(f"Paper {i}") for i in range(20)]
        graph = builder.build(papers)
        assert len(graph.clusters) <= 3

    def test_graph_completeness(self):
        """All papers should appear in exactly one cluster and timeline."""
        builder = ResearchGraphBuilder()
        papers = [make_paper(f"Paper {i}", year=2020 + i) for i in range(5)]
        graph = builder.build(papers)

        cluster_paper_ids = set()
        for c in graph.clusters:
            cluster_paper_ids.update(c.paper_ids)
        assert len(cluster_paper_ids) == 5

        timeline_paper_ids = set()
        for t in graph.timeline:
            timeline_paper_ids.update(t.paper_ids)
        assert len(timeline_paper_ids) == 5

    def test_deterministic_output(self):
        """Same input should produce same graph."""
        builder = ResearchGraphBuilder(encoder=FakeEncoder(dimension=64))
        papers = [make_paper(f"Paper {i}", year=2020 + i) for i in range(5)]
        g1 = builder.build(papers)
        g2 = builder.build(papers)
        assert len(g1.nodes) == len(g2.nodes)
        assert len(g1.clusters) == len(g2.clusters)
        assert len(g1.edges) == len(g2.edges)
        # Cluster assignments should match
        c1_ids = {c.cluster_id for c in g1.clusters}
        c2_ids = {c.cluster_id for c in g2.clusters}
        assert c1_ids == c2_ids

    def test_edges_with_both_references_and_citations(self):
        p1 = make_paper("Paper A")
        p2 = make_paper("Paper B")
        p3 = make_paper("Paper C")
        p1.references = [p2.paper_id]
        p2.citations = [p3.paper_id]
        p3.references = [p1.paper_id]
        builder = ResearchGraphBuilder()
        graph = builder.build([p1, p2, p3])
        assert len(graph.edges) == 3
        edge_types = {e.edge_type for e in graph.edges}
        assert "reference" in edge_types
        assert "citation" in edge_types
