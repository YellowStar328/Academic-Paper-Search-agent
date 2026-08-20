"""Tests for FakeEncoder and EmbeddingReranker."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from app.config import Settings
from app.embedding.encoder import FakeEncoder, cosine_similarity
from app.embedding.reranker import EmbeddingReranker
from app.models.paper import Paper, PaperIdentity


def make_paper(title: str, abstract: str = "", citation_count: int = 0) -> Paper:
    return Paper(
        paper_id=str(uuid4()),
        identity=PaperIdentity(normalized_title=title.lower().replace(" ", "")),
        title=title,
        abstract=abstract,
        year=2024,
        citation_count=citation_count,
        source="test",
    )


class TestFakeEncoder:
    """Tests for FakeEncoder deterministic hashing."""

    def test_dimension(self):
        enc = FakeEncoder(dimension=128)
        assert enc.dimension == 128

    def test_encode_returns_correct_dimension(self):
        enc = FakeEncoder(dimension=64)
        vec = enc.encode("hello world")
        assert vec.shape == (64,)

    def test_deterministic_same_input(self):
        enc = FakeEncoder(dimension=128)
        v1 = enc.encode("machine learning")
        v2 = enc.encode("machine learning")
        np.testing.assert_array_equal(v1, v2)

    def test_different_input_different_output(self):
        enc = FakeEncoder(dimension=128)
        v1 = enc.encode("machine learning")
        v2 = enc.encode("deep learning models")
        assert not np.array_equal(v1, v2)

    def test_l2_normalized(self):
        enc = FakeEncoder(dimension=64)
        vec = enc.encode("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_string(self):
        enc = FakeEncoder(dimension=32)
        vec = enc.encode("")
        assert vec.shape == (32,)
        assert not np.all(vec == 0)

    def test_encode_batch(self):
        enc = FakeEncoder(dimension=32)
        vecs = enc.encode_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(v.shape == (32,) for v in vecs)

    def test_values_in_range(self):
        enc = FakeEncoder(dimension=64)
        vec = enc.encode("bounded values test")
        assert np.all(np.abs(vec) <= 1.0)


class TestCosineSimilarity:
    """Tests for cosine_similarity helper."""

    def test_identical_vectors(self):
        enc = FakeEncoder(dimension=64)
        v = enc.encode("test")
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-5

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert abs(sim + 1.0) < 1e-5


class TestEmbeddingReranker:
    """Tests for EmbeddingReranker."""

    def test_empty_input(self):
        reranker = EmbeddingReranker(top_k=10)
        result = reranker.rerank("query", [])
        assert result == []

    def test_returns_top_k(self):
        reranker = EmbeddingReranker(top_k=3)
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        result = reranker.rerank("query", papers)
        assert len(result) == 3

    def test_returns_all_if_fewer_than_top_k(self):
        reranker = EmbeddingReranker(top_k=10)
        papers = [make_paper(f"Paper {i}") for i in range(3)]
        result = reranker.rerank("query", papers)
        assert len(result) == 3

    def test_sorted_by_similarity_descending(self):
        reranker = EmbeddingReranker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(10)]
        result = reranker.rerank("machine learning", papers)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_papers_returns_papers_only(self):
        reranker = EmbeddingReranker(top_k=3)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        result = reranker.rerank_papers("query", papers)
        assert len(result) == 3
        assert all(isinstance(p, Paper) for p in result)

    def test_similarity_scores_in_range(self):
        reranker = EmbeddingReranker(top_k=5)
        papers = [make_paper(f"Paper {i}") for i in range(5)]
        result = reranker.rerank("query", papers)
        for _, score in result:
            assert -1.0 <= score <= 1.0
