"""Tests for data models."""

import numpy as np

from app.models.paper import Paper, PaperIdentity, PaperList, normalize_title
from app.models.query import (
    HardFilter,
    QueryIntent,
    SearchOptions,
    SearchQuery,
    UserQuery,
)
from app.models.candidate import CandidateQuery, JudgeResult, PaperJudgeResult
from app.models.agent import AgentRun, ModelConfidence
from app.models.bandit import BanditState, BetaArm, SourceBanditState
from app.models.result import (
    GraphCluster,
    GraphEdge,
    GraphNode,
    PaperWithScores,
    PipelineMetrics,
    ResearchGraph,
    SearchResult,
    TimelineEntry,
)


# ---------- Paper & Identity ----------

class TestPaperIdentity:
    def test_doi_priority(self):
        identity = PaperIdentity(
            doi="10.1234/test",
            arxiv_id="2401.00001",
            normalized_title="test",
            year=2024,
        )
        assert identity.identity_key() == "doi:10.1234/test"

    def test_arxiv_fallback(self):
        identity = PaperIdentity(arxiv_id="2401.00001", normalized_title="test", year=2024)
        assert identity.identity_key() == "arxiv:2401.00001"

    def test_s2_fallback(self):
        identity = PaperIdentity(
            semantic_scholar_id="abc123", normalized_title="test", year=2024
        )
        assert identity.identity_key() == "s2:abc123"

    def test_title_year_fallback(self):
        identity = PaperIdentity(normalized_title="somepapertitle", year=2023)
        assert identity.identity_key() == "title_year:somepapertitle:2023"

    def test_empty_identity(self):
        identity = PaperIdentity()
        assert identity.identity_key() == ""

    def test_doi_case_insensitive(self):
        identity1 = PaperIdentity(doi="10.1234/TEST")
        identity2 = PaperIdentity(doi="10.1234/test")
        assert identity1.identity_key() == identity2.identity_key()

    def test_from_paper_data(self):
        identity = PaperIdentity.from_paper_data(
            title="Attention Is All You Need",
            year=2017,
            arxiv_id="1706.03762",
        )
        assert identity.normalized_title == "attentionisallyouneed"
        assert identity.identity_key() == "arxiv:1706.03762"


class TestNormalizeTitle:
    def test_basic(self):
        assert normalize_title("Hello, World!") == "helloworld"

    def test_unicode(self):
        assert normalize_title("café résumé") == "caferesume"

    def test_empty(self):
        assert normalize_title("") == ""

    def test_numbers_preserved(self):
        assert normalize_title("GPT-4: A Study") == "gpt4astudy"


class TestPaper:
    def test_paper_identity_key(self):
        identity = PaperIdentity(doi="10.1234/test")
        paper = Paper(paper_id="p1", identity=identity, title="Test")
        assert paper.identity_key() == "doi:10.1234/test"

    def test_paper_defaults(self):
        paper = Paper(paper_id="p1")
        assert paper.authors == []
        assert paper.citation_count == 0
        assert paper.source == "unknown"


class TestPaperList:
    def test_extend(self):
        p1 = Paper(paper_id="p1")
        p2 = Paper(paper_id="p2")
        list1 = PaperList(papers=[p1], source="arxiv")
        list2 = PaperList(papers=[p2], source="s2")
        list1.extend(list2)
        assert len(list1) == 2

    def test_iter(self):
        p1 = Paper(paper_id="p1")
        plist = PaperList(papers=[p1])
        papers = [p for p in plist]
        assert len(papers) == 1


# ---------- Query Models ----------

class TestSearchQuery:
    def test_defaults(self):
        sq = SearchQuery(original_query="test", semantic_core="semantic test")
        assert sq.domain == "general"
        assert sq.intent == QueryIntent.SURVEY
        assert sq.sub_queries == []
        assert sq.query_type == "general:survey"

    def test_hard_filters(self):
        sq = SearchQuery(
            original_query="ML papers",
            semantic_core="machine learning",
            hard_filters=HardFilter(year_start=2020, min_citations=10),
        )
        assert sq.hard_filters.year_start == 2020
        assert sq.hard_filters.min_citations == 10

    def test_options(self):
        sq = SearchQuery(
            original_query="test",
            semantic_core="test",
            options=SearchOptions(top_k=50, mode="human_review"),
        )
        assert sq.options.top_k == 50
        assert sq.options.mode == "human_review"


class TestUserQuery:
    def test_basic(self):
        uq = UserQuery(query="transformer architectures")
        assert uq.query == "transformer architectures"
        assert uq.options.top_k == 20


# ---------- Candidate & Judge Models ----------

class TestCandidateQuery:
    def test_basic(self):
        cq = CandidateQuery(query="attention mechanism", proposer_model="qwen")
        assert cq.proposer_model == "qwen"
        assert cq.logic == "OR"

    def test_judge_result_validation(self):
        cq = CandidateQuery(query="test", proposer_model="glm")
        jr = JudgeResult(candidate=cq, score=0.85)
        assert jr.score == 0.85
        assert 0.0 <= jr.coverage <= 1.0


class TestPaperJudgeResult:
    def test_basic(self):
        r = PaperJudgeResult(paper_id="p1", relevance_score=0.9)
        assert r.relevance_score == 0.9
        assert r.authority_score == 0.5


# ---------- Agent Models ----------

class TestAgentRun:
    def test_defaults(self):
        run = AgentRun(model_name="qwen")
        assert run.success is True
        assert run.model_name == "qwen"
        assert run.run_id  # auto-generated UUID


class TestModelConfidence:
    def test_mean(self):
        mc = ModelConfidence(
            model_name="qwen", domain="cs", query_type="cs:survey",
            alpha=3.0, beta=1.0,
        )
        assert abs(mc.mean() - 0.75) < 0.01


# ---------- Bandit Models ----------

class TestBetaArm:
    def test_sample_in_range(self):
        arm = BetaArm(alpha=2.0, beta=3.0)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0

    def test_update_increases_alpha_on_high_reward(self):
        arm = BetaArm(alpha=1.0, beta=1.0)
        arm.update(1.0)
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_update_increases_beta_on_low_reward(self):
        arm = BetaArm(alpha=1.0, beta=1.0)
        arm.update(0.0)
        assert arm.alpha == 1.0
        assert arm.beta == 2.0

    def test_fractional_update(self):
        arm = BetaArm(alpha=1.0, beta=1.0)
        arm.update(0.3)
        assert abs(arm.alpha - 1.3) < 0.01
        assert abs(arm.beta - 1.7) < 0.01

    def test_mean(self):
        arm = BetaArm(alpha=3.0, beta=1.0)
        assert abs(arm.mean() - 0.75) < 0.01

    def test_reset(self):
        arm = BetaArm(alpha=5.0, beta=3.0)
        arm.reset()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0


class TestBanditState:
    def test_get_arm_creates_if_missing(self):
        state = BanditState()
        arm = state.get_arm("qwen", "cs", "cs:survey")
        assert arm.alpha == 1.0
        assert arm.beta == 1.0

    def test_sample_all(self):
        state = BanditState()
        models = ["qwen", "deepseek", "glm"]
        samples = state.sample_all(models, "cs", "cs:survey")
        assert len(samples) == 3
        for m in models:
            assert 0.0 <= samples[m] <= 1.0

    def test_allocate_budget_basic(self):
        state = BanditState()
        models = ["qwen", "deepseek", "glm"]
        alloc = state.allocate_budget(models, "cs", "cs:survey", total_budget=10)
        assert sum(alloc.values()) == 10
        for m in models:
            assert alloc[m] >= 1  # exploration floor

    def test_allocate_budget_exploration_floor(self):
        state = BanditState(exploration_floor=0.10)
        models = ["qwen", "deepseek", "glm"]
        alloc = state.allocate_budget(models, "cs", "cs:survey", total_budget=10)
        # Each model should get at least 1 (10% of 10)
        for m in models:
            assert alloc[m] >= 1

    def test_allocate_budget_empty(self):
        state = BanditState()
        alloc = state.allocate_budget([], "cs", "cs:survey", total_budget=10)
        assert alloc == {}

    def test_allocate_budget_zero(self):
        state = BanditState()
        alloc = state.allocate_budget(["qwen"], "cs", "cs:survey", total_budget=0)
        assert alloc == {"qwen": 0}

    def test_update_changes_arm(self):
        state = BanditState()
        arm_before = state.get_arm("qwen", "cs", "cs:survey")
        alpha_before = arm_before.alpha
        state.update("qwen", "cs", "cs:survey", 1.0)
        arm_after = state.get_arm("qwen", "cs", "cs:survey")
        assert arm_after.alpha > alpha_before

    def test_reset(self):
        state = BanditState()
        state.update("qwen", "cs", "cs:survey", 1.0)
        state.reset()
        arm = state.get_arm("qwen", "cs", "cs:survey")
        assert arm.alpha == 1.0

    def test_biased_allocation_after_updates(self):
        """After many positive updates, a model should get more budget."""
        state = BanditState()
        # Give qwen many positive rewards
        for _ in range(50):
            state.update("qwen", "cs", "cs:survey", 1.0)
        # Give deepseek many negative rewards
        for _ in range(50):
            state.update("deepseek", "cs", "cs:survey", 0.0)

        # Sample multiple times and check average allocation
        qwen_total = 0
        deepseek_total = 0
        for _ in range(100):
            alloc = state.allocate_budget(
                ["qwen", "deepseek"], "cs", "cs:survey", total_budget=10
            )
            qwen_total += alloc["qwen"]
            deepseek_total += alloc["deepseek"]

        assert qwen_total > deepseek_total


class TestSourceBanditState:
    def test_basic(self):
        state = SourceBanditState()
        arm = state.get_arm("arxiv", "cs")
        assert arm.alpha == 1.0

    def test_sample_and_update(self):
        state = SourceBanditState()
        s = state.sample("arxiv", "cs")
        assert 0.0 <= s <= 1.0
        state.update("arxiv", "cs", 1.0)
        arm = state.get_arm("arxiv", "cs")
        assert arm.alpha == 2.0


# ---------- Result Models ----------

class TestPaperWithScores:
    def test_defaults(self):
        pws = PaperWithScores(paper=Paper(paper_id="p1"))
        assert pws.relevance_score == 0.0
        assert pws.final_score == 0.0


class TestResearchGraph:
    def test_empty(self):
        g = ResearchGraph()
        assert g.nodes == []
        assert g.edges == []

    def test_with_data(self):
        g = ResearchGraph(
            nodes=[GraphNode(paper_id="p1", title="Paper 1")],
            edges=[GraphEdge(source_id="p1", target_id="p2")],
            clusters=[GraphCluster(cluster_id=0, paper_ids=["p1"])],
            timeline=[TimelineEntry(year=2023, count=1)],
        )
        assert len(g.nodes) == 1
        assert len(g.edges) == 1


class TestSearchResult:
    def test_basic(self):
        sr = SearchResult(query="test", semantic_core="test")
        assert sr.papers == []
        assert sr.graph.nodes == []

    def test_with_papers(self):
        pws = PaperWithScores(
            paper=Paper(paper_id="p1"),
            relevance_score=0.9,
            final_score=0.85,
        )
        sr = SearchResult(query="test", semantic_core="test", papers=[pws])
        assert len(sr.papers) == 1
        assert sr.papers[0].final_score == 0.85


class TestPipelineMetrics:
    def test_defaults(self):
        m = PipelineMetrics(request_id="req1", query="test")
        assert m.llm_calls == 0
        assert m.total_latency_ms == 0.0
        assert m.models_used == []
