"""Tests for Thompson Sampling manager and source bandit."""

import pytest

from app.bandit.thompson import ThompsonSamplingManager
from app.bandit.source_bandit import SourceBanditManager
from app.models.candidate import CandidateQuery, JudgeResult, PaperJudgeResult


class TestThompsonSamplingManager:
    def test_allocate_budget_basic(self):
        mgr = ThompsonSamplingManager()
        models = ["qwen", "deepseek", "glm"]
        alloc = mgr.allocate_budget(models, "cs", "cs:survey", total_budget=10)
        assert sum(alloc.values()) == 10
        for m in models:
            assert alloc[m] >= 1

    def test_allocate_budget_exploration_floor(self):
        """Each model gets at least exploration_floor * total_budget."""
        mgr = ThompsonSamplingManager(exploration_floor=0.15)
        models = ["qwen", "deepseek", "glm"]
        alloc = mgr.allocate_budget(models, "cs", "cs:survey", total_budget=20)
        # 15% of 20 = 3, each model should get at least 3
        for m in models:
            assert alloc[m] >= 3

    def test_allocate_budget_empty_models(self):
        mgr = ThompsonSamplingManager()
        alloc = mgr.allocate_budget([], "cs", "cs:survey", total_budget=10)
        assert alloc == {}

    def test_allocate_budget_single_model(self):
        mgr = ThompsonSamplingManager()
        alloc = mgr.allocate_budget(["qwen"], "cs", "cs:survey", total_budget=5)
        assert alloc == {"qwen": 5}

    def test_compute_reward_from_judge(self):
        mgr = ThompsonSamplingManager()
        results = [
            JudgeResult(
                candidate=CandidateQuery(query="q1", proposer_model="qwen"),
                score=0.8,
            ),
            JudgeResult(
                candidate=CandidateQuery(query="q2", proposer_model="qwen"),
                score=0.6,
            ),
            JudgeResult(
                candidate=CandidateQuery(query="q3", proposer_model="deepseek"),
                score=0.9,
            ),
        ]
        rewards = mgr.compute_reward_from_judge(results)
        assert abs(rewards["qwen"] - 0.7) < 0.01
        assert abs(rewards["deepseek"] - 0.9) < 0.01

    def test_compute_reward_from_judge_empty(self):
        mgr = ThompsonSamplingManager()
        rewards = mgr.compute_reward_from_judge([])
        assert rewards == {}

    def test_compute_reward_from_papers(self):
        mgr = ThompsonSamplingManager()
        paper_results = [
            PaperJudgeResult(paper_id="p1", relevance_score=0.9),
            PaperJudgeResult(paper_id="p2", relevance_score=0.7),
            PaperJudgeResult(paper_id="p3", relevance_score=0.5),
        ]
        mapping = {
            "qwen": ["p1", "p2"],
            "deepseek": ["p3"],
            "glm": ["p_nonexistent"],  # no matching papers
        }
        rewards = mgr.compute_reward_from_papers(paper_results, mapping)
        assert abs(rewards["qwen"] - 0.8) < 0.01
        assert abs(rewards["deepseek"] - 0.5) < 0.01
        assert abs(rewards["glm"] - 0.5) < 0.01  # default

    def test_compute_reward_from_papers_empty(self):
        mgr = ThompsonSamplingManager()
        rewards = mgr.compute_reward_from_papers([], {})
        assert rewards == {}

    def test_update_state_increases_alpha(self):
        mgr = ThompsonSamplingManager()
        arm_before = mgr.state.get_arm("qwen", "cs", "cs:survey")
        alpha_before = arm_before.alpha

        mgr.update_state("qwen", "cs", "cs:survey", 1.0)
        arm_after = mgr.state.get_arm("qwen", "cs", "cs:survey")
        assert arm_after.alpha > alpha_before

    def test_update_state_batch(self):
        mgr = ThompsonSamplingManager()
        rewards = {"qwen": 0.9, "deepseek": 0.3, "glm": 0.6}
        mgr.update_state_batch(rewards, "cs", "cs:survey")

        for model in rewards:
            arm = mgr.state.get_arm(model, "cs", "cs:survey")
            assert arm.alpha > 1.0  # updated

    def test_update_state_batch_order(self):
        """Model with higher reward should have higher alpha."""
        mgr = ThompsonSamplingManager()
        rewards = {"qwen": 0.9, "deepseek": 0.1}
        mgr.update_state_batch(rewards, "cs", "cs:survey")

        qwen_arm = mgr.state.get_arm("qwen", "cs", "cs:survey")
        deepseek_arm = mgr.state.get_arm("deepseek", "cs", "cs:survey")
        assert qwen_arm.alpha > deepseek_arm.alpha

    def test_get_confidence(self):
        mgr = ThompsonSamplingManager()
        # Initially 1.0/(1.0+1.0) = 0.5
        assert abs(mgr.get_confidence("qwen", "cs", "cs:survey") - 0.5) < 0.01

        # After a positive update
        mgr.update_state("qwen", "cs", "cs:survey", 1.0)
        assert mgr.get_confidence("qwen", "cs", "cs:survey") > 0.5

    def test_get_all_confidences(self):
        mgr = ThompsonSamplingManager()
        models = ["qwen", "deepseek", "glm"]
        confs = mgr.get_all_confidences(models, "cs", "cs:survey")
        assert len(confs) == 3
        for m in models:
            assert m in confs
            assert 0.0 <= confs[m] <= 1.0

    def test_reset(self):
        mgr = ThompsonSamplingManager()
        mgr.update_state("qwen", "cs", "cs:survey", 1.0)
        mgr.reset()
        arm = mgr.state.get_arm("qwen", "cs", "cs:survey")
        assert arm.alpha == 1.0
        assert arm.beta == 1.0

    def test_sample_thetas(self):
        mgr = ThompsonSamplingManager()
        models = ["qwen", "deepseek", "glm"]
        thetas = mgr.sample_thetas(models, "cs", "cs:survey")
        assert len(thetas) == 3
        for m in models:
            assert 0.0 <= thetas[m] <= 1.0

    def test_biased_allocation_after_updates(self):
        """After many updates, better model should consistently get more budget."""
        mgr = ThompsonSamplingManager()
        # Give qwen many positive rewards
        for _ in range(100):
            mgr.update_state("qwen", "cs", "cs:survey", 0.9)
        # Give deepseek many negative rewards
        for _ in range(100):
            mgr.update_state("deepseek", "cs", "cs:survey", 0.1)

        qwen_wins = 0
        deepseek_wins = 0
        for _ in range(100):
            alloc = mgr.allocate_budget(
                ["qwen", "deepseek"], "cs", "cs:survey", total_budget=10
            )
            if alloc["qwen"] > alloc["deepseek"]:
                qwen_wins += 1
            elif alloc["deepseek"] > alloc["qwen"]:
                deepseek_wins += 1

        # Qwen should win the majority of the time
        assert qwen_wins > deepseek_wins

    def test_no_model_collapse(self):
        """Even with extreme rewards, all models should still get budget."""
        mgr = ThompsonSamplingManager()
        # Give qwen extreme positive rewards
        for _ in range(1000):
            mgr.update_state("qwen", "cs", "cs:survey", 1.0)
        # Give others extreme negative rewards
        for _ in range(1000):
            mgr.update_state("deepseek", "cs", "cs:survey", 0.0)
            mgr.update_state("glm", "cs", "cs:survey", 0.0)

        alloc = mgr.allocate_budget(
            ["qwen", "deepseek", "glm"], "cs", "cs:survey", total_budget=10
        )
        # All models should get at least 1 (exploration floor)
        for m in ["qwen", "deepseek", "glm"]:
            assert alloc[m] >= 1, f"Model {m} got {alloc[m]} — model collapse detected!"


class TestSourceBanditManager:
    def test_get_source_priorities(self):
        mgr = SourceBanditManager()
        sources = ["arxiv", "semantic_scholar", "pubmed"]
        priorities = mgr.get_source_priorities(sources, "cs")
        assert len(priorities) == 3
        assert set(priorities) == set(sources)

    def test_update_source(self):
        mgr = SourceBanditManager()
        alpha_before = mgr.state.get_arm("arxiv", "cs").alpha
        assert alpha_before == 1.0
        mgr.update_source("arxiv", "cs", 1.0)
        alpha_after = mgr.state.get_arm("arxiv", "cs").alpha
        assert alpha_after > alpha_before

    def test_get_source_confidence(self):
        mgr = SourceBanditManager()
        # Initially 0.5
        assert abs(mgr.get_source_confidence("arxiv", "cs") - 0.5) < 0.01

        # After positive updates
        for _ in range(10):
            mgr.update_source("arxiv", "cs", 1.0)
        assert mgr.get_source_confidence("arxiv", "cs") > 0.5

    def test_better_source_prioritized(self):
        """Source with more positive updates should be queried first more often."""
        mgr = SourceBanditManager()
        # arxiv: many positive
        for _ in range(50):
            mgr.update_source("arxiv", "cs", 1.0)
        # pubmed: many negative
        for _ in range(50):
            mgr.update_source("pubmed", "cs", 0.0)

        arxiv_first_count = 0
        for _ in range(100):
            priorities = mgr.get_source_priorities(["arxiv", "pubmed"], "cs")
            if priorities[0] == "arxiv":
                arxiv_first_count += 1

        assert arxiv_first_count > 50  # should win majority

    def test_reset(self):
        mgr = SourceBanditManager()
        mgr.update_source("arxiv", "cs", 1.0)
        mgr.reset()
        arm = mgr.state.get_arm("arxiv", "cs")
        assert arm.alpha == 1.0
