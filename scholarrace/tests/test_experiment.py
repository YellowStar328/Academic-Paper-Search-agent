"""Tests for experiment and ablation scripts."""

from __future__ import annotations

import asyncio
import csv
import os
import tempfile

import pytest

from app.config import Settings
from scripts.run_experiment import (
    STRATEGIES,
    build_pipeline,
    make_experiment_papers,
    run_single_experiment,
    run_all_strategies,
    write_csv,
)
from scripts.run_ablation import (
    ABLATION_LEVELS,
    run_ablation_level,
    run_all_ablations,
    write_csv as write_ablation_csv,
)


class TestExperimentScripts:
    """Tests for run_experiment.py."""

    @pytest.mark.asyncio
    async def test_run_single_experiment_single(self):
        result = await run_single_experiment(
            "machine learning", "single", Settings(app_env="test")
        )
        assert result["strategy"] == "single"
        assert result["total_papers"] > 0
        assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_single_experiment_multi(self):
        result = await run_single_experiment(
            "deep learning", "multi", Settings(app_env="test")
        )
        assert result["strategy"] == "multi"
        assert result["total_papers"] > 0

    @pytest.mark.asyncio
    async def test_run_single_experiment_thompson(self):
        result = await run_single_experiment(
            "transformers", "thompson", Settings(app_env="test")
        )
        assert result["strategy"] == "thompson"
        assert result["total_papers"] > 0

    @pytest.mark.asyncio
    async def test_run_single_experiment_thompson_full(self):
        result = await run_single_experiment(
            "neural networks", "thompson_full", Settings(app_env="test")
        )
        assert result["strategy"] == "thompson_full"
        assert result["total_papers"] > 0

    @pytest.mark.asyncio
    async def test_run_all_strategies(self):
        results = await run_all_strategies(
            "AI research", Settings(app_env="test")
        )
        assert len(results) == len(STRATEGIES)
        strategies = {r["strategy"] for r in results}
        assert strategies == set(STRATEGIES)

    def test_make_experiment_papers(self):
        papers = make_experiment_papers()
        assert len(papers) == 10
        assert all(p.citation_count > 0 for p in papers)
        assert all(p.identity.doi for p in papers)

    def test_build_pipeline_single(self):
        pipeline = build_pipeline("single", Settings(app_env="test"))
        assert pipeline is not None

    def test_build_pipeline_thompson_full(self):
        pipeline = build_pipeline("thompson_full", Settings(app_env="test"))
        assert pipeline is not None

    def test_write_csv(self):
        results = [
            {"strategy": "single", "query": "test", "total_papers": 5},
            {"strategy": "multi", "query": "test", "total_papers": 8},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            write_csv(results, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["strategy"] == "single"
            assert rows[1]["strategy"] == "multi"
        finally:
            os.unlink(path)

    def test_write_csv_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            write_csv([], path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestAblationScripts:
    """Tests for run_ablation.py."""

    @pytest.mark.asyncio
    async def test_run_ablation_level_A(self):
        result = await run_ablation_level(
            "machine learning", "A", Settings(app_env="test")
        )
        assert result["level"] == "A"
        assert result["total_papers"] > 0
        assert "Baseline" in result["description"]

    @pytest.mark.asyncio
    async def test_run_ablation_level_H(self):
        result = await run_ablation_level(
            "deep learning", "H", Settings(app_env="test")
        )
        assert result["level"] == "H"
        assert result["total_papers"] > 0
        assert "full pipeline" in result["description"]

    @pytest.mark.asyncio
    async def test_run_all_ablations(self):
        results = await run_all_ablations(
            "AI research", Settings(app_env="test")
        )
        assert len(results) == len(ABLATION_LEVELS)
        levels = [r["level"] for r in results]
        assert levels == ABLATION_LEVELS

    def test_ablation_levels(self):
        assert ABLATION_LEVELS == ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_write_ablation_csv(self):
        results = [
            {"level": "A", "description": "baseline", "total_papers": 5},
            {"level": "H", "description": "full", "total_papers": 10},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            write_ablation_csv(results, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["level"] == "A"
            assert rows[1]["level"] == "H"
        finally:
            os.unlink(path)
