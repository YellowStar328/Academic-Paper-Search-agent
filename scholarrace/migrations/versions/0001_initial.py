"""Initial schema migration.

Creates all tables: papers, query_logs, agent_runs, model_confidence, feedback.

Revision ID: 0001_initial
Revises:
Create Date: 2024-01-01 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("paper_id", sa.String(36), primary_key=True),
        sa.Column("title", sa.Text, nullable=False, server_default=""),
        sa.Column("abstract", sa.Text, nullable=False, server_default=""),
        sa.Column("authors", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("year", sa.Integer, nullable=True),
        sa.Column("venue", sa.Text, nullable=False, server_default=""),
        sa.Column("doi", sa.String(255), nullable=True),
        sa.Column("arxiv_id", sa.String(50), nullable=True),
        sa.Column("semantic_scholar_id", sa.String(50), nullable=True),
        sa.Column("openalex_id", sa.String(50), nullable=True),
        sa.Column("pubmed_id", sa.String(50), nullable=True),
        sa.Column("url", sa.Text, nullable=False, server_default=""),
        sa.Column("pdf_url", sa.Text, nullable=True),
        sa.Column("citation_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("reference_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("fields_of_study", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("keywords", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("source", sa.String(50), nullable=False, server_default="unknown"),
        sa.Column("embedding", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_papers_doi", "papers", ["doi"])
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"])
    op.create_index("ix_papers_s2_id", "papers", ["semantic_scholar_id"])
    op.create_index("ix_papers_openalex_id", "papers", ["openalex_id"])
    op.create_index("ix_papers_pubmed_id", "papers", ["pubmed_id"])

    op.create_table(
        "query_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("original_query", sa.Text, nullable=False),
        sa.Column("semantic_core", sa.Text, nullable=False, server_default=""),
        sa.Column("domain", sa.String(100), nullable=False, server_default="general"),
        sa.Column("intent", sa.String(50), nullable=False, server_default="survey"),
        sa.Column("options", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("result_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_query_logs_ts", "query_logs", ["timestamp"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("query_text", sa.Text, nullable=False, server_default=""),
        sa.Column("generated_candidates", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("token_usage", sa.Integer, nullable=False, server_default="0"),
        sa.Column("success", sa.Integer, nullable=False, server_default="1"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_runs_req", "agent_runs", ["request_id"])
    op.create_index("ix_agent_runs_ts", "agent_runs", ["timestamp"])

    op.create_table(
        "model_confidence",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(100), nullable=False),
        sa.Column("query_type", sa.String(100), nullable=False),
        sa.Column("alpha", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("beta", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("total_runs", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_reward", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_mc_model", "model_confidence", ["model_name"])
    op.create_index("ix_mc_domain", "model_confidence", ["domain"])
    op.create_index("ix_mc_qtype", "model_confidence", ["query_type"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), nullable=False),
        sa.Column("paper_id", sa.String(36), nullable=True),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text, nullable=False, server_default=""),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_req", "feedback", ["request_id"])
    op.create_index("ix_feedback_ts", "feedback", ["timestamp"])


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("model_confidence")
    op.drop_table("agent_runs")
    op.drop_table("query_logs")
    op.drop_table("papers")
