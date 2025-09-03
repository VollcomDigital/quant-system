"""create unified CLI schema (runs, backtest_results, trades, symbol_aggregates, run_artifacts)

Revision ID: 0001_unified_cli_schema
Revises:
Create Date: 2025-08-27 07:09:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_unified_cli_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Runs table
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "started_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("collection_ref", sa.Text(), nullable=False),
        sa.Column("strategies_mode", sa.String(length=256), nullable=False),
        sa.Column("intervals_mode", sa.String(length=256), nullable=False),
        sa.Column("target_metric", sa.String(length=64), nullable=False),
        sa.Column("period_mode", sa.String(length=64), nullable=False),
        sa.Column("args_json", pg.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("git_sha_app", sa.String(length=64), nullable=True),
        sa.Column("git_sha_strat", sa.String(length=64), nullable=True),
        sa.Column("data_source", sa.String(length=128), nullable=True),
        sa.Column("plan_hash", sa.String(length=128), nullable=False, unique=True),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="running"
        ),
        sa.Column("error_summary", sa.Text(), nullable=True),
    )

    # Backtest results
    op.create_table(
        "backtest_results",
        sa.Column("result_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(length=36),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("symbol", sa.String(length=64), nullable=False, index=True),
        sa.Column("strategy", sa.String(length=256), nullable=False, index=True),
        sa.Column("interval", sa.String(length=32), nullable=False, index=True),
        sa.Column("start_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rank_in_symbol", sa.Integer(), nullable=True),
        sa.Column("metrics", pg.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("engine_ctx", pg.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("trades_raw", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "run_id",
            "symbol",
            "strategy",
            "interval",
            name="uq_run_symbol_strategy_interval",
        ),
    )

    # Trades
    op.create_table(
        "trades",
        sa.Column("trade_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "result_id",
            sa.String(length=36),
            sa.ForeignKey("backtest_results.result_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("trade_index", sa.Integer(), nullable=False),
        sa.Column("size", sa.String(length=64), nullable=True),
        sa.Column("entry_bar", sa.BigInteger(), nullable=True),
        sa.Column("exit_bar", sa.BigInteger(), nullable=True),
        sa.Column("entry_price", sa.String(length=64), nullable=True),
        sa.Column("exit_price", sa.String(length=64), nullable=True),
        sa.Column("pnl", sa.String(length=64), nullable=True),
        sa.Column("duration", sa.Interval(), nullable=True),
        sa.Column("tag", sa.String(length=128), nullable=True),
        sa.Column("entry_signals", sa.Text(), nullable=True),
        sa.Column("exit_signals", sa.Text(), nullable=True),
        sa.UniqueConstraint("result_id", "trade_index", name="uq_result_trade_index"),
    )

    # Symbol aggregates
    op.create_table(
        "symbol_aggregates",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(length=36),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("best_by", sa.String(length=64), nullable=False),
        sa.Column(
            "best_result",
            sa.String(length=36),
            sa.ForeignKey("backtest_results.result_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary", pg.JSONB(astext_type=sa.Text()), nullable=False),
        sa.UniqueConstraint("run_id", "symbol", "best_by", name="uq_run_symbol_bestby"),
    )

    # Run artifacts
    op.create_table(
        "run_artifacts",
        sa.Column("artifact_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(length=36),
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("path_or_uri", sa.Text(), nullable=False),
        sa.Column("meta", pg.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("run_artifacts")
    op.drop_table("symbol_aggregates")
    op.drop_table("trades")
    op.drop_table("backtest_results")
    op.drop_table("runs")
