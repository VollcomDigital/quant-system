"""
Lightweight SQLAlchemy models and helpers for the Unified CLI run lineage and results.

This module provides:
- Declarative models for runs, backtest_results, trades, symbol_aggregates, run_artifacts.
- Helper functions: create_tables, create_run_from_manifest, find_run_by_plan_hash.

It is intentionally defensive: tries to reuse src.database.db_connection.get_engine() if available,
falls back to a sqlite file-based engine when not. Designed for best-effort use by the CLI.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import (
    JSONB as PG_JSONB,  # type: ignore[import-not-found]
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker

# Prefer JSONB for Postgres, fallback to generic JSON
try:
    from sqlalchemy import JSON as SQLJSON  # type: ignore[import-not-found]
except Exception:
    SQLJSON = Text

Base = declarative_base()


# Engine/session helpers
def _get_engine():
    # Try to reuse project's db_connection engine helpers if present
    # Prefer sync engine so this module stays simple.
    # Test/CI override: force lightweight SQLite to avoid external DB dependency
    try:
        force_sqlite = False
        # Common signals for test/CI environments available at import time
        if os.environ.get("UNIFIED_MODELS_SQLITE", "").lower() in {
            "1",
            "true",
            "yes",
        } or os.environ.get("CI", "").lower() in {"1", "true", "yes"}:
            force_sqlite = True
        elif os.environ.get("PYTEST_CURRENT_TEST"):
            # Usually set by pytest while collecting/running tests
            force_sqlite = True
        elif os.environ.get("TESTING", "").lower() in {"1", "true", "yes"}:
            force_sqlite = True

        if force_sqlite:
            database_url = f"sqlite:///{os.path.abspath('quant_unified_test.db')}"
            return create_engine(database_url, echo=False, future=True)
    except Exception:
        pass
    try:
        from src.database.db_connection import (
            get_sync_engine,  # type: ignore[import-not-found]
        )

        eng = get_sync_engine()
        if eng is not None:
            return eng
    except Exception:
        pass
    try:
        # As a secondary option, try the DatabaseManager property if exported
        from src.database.db_connection import (
            db_manager,  # type: ignore[import-not-found]
        )

        eng = getattr(db_manager, "sync_engine", None)
        if eng is not None:
            return eng
    except Exception:
        pass

    # Fallback: use DATABASE_URL env var or sqlite file
    database_url = (
        os.environ.get("DATABASE_URL")
        or f"sqlite:///{os.path.abspath('quant_unified.db')}"
    )
    eng = create_engine(database_url, echo=False, future=True)
    return eng


ENGINE = _get_engine()
Session = scoped_session(
    sessionmaker(bind=ENGINE, autoflush=False, future=True, expire_on_commit=False)
)


# Helper to pick JSON type depending on DB
def JSON_TYPE():
    url = str(ENGINE.url).lower() if ENGINE and ENGINE.url else ""
    if "postgres" in url or "psql" in url:
        return PG_JSONB
    return SQLJSON


class Run(Base):
    __tablename__ = "runs"
    # Note: default schema left to DB config; migrations can add schema `quant` if desired.
    run_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at_utc = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    finished_at_utc = Column(DateTime(timezone=True), nullable=True)
    actor = Column(String(128), nullable=False)
    action = Column(String(64), nullable=False)
    collection_ref = Column(Text, nullable=False)
    strategies_mode = Column(String(256), nullable=False)
    intervals_mode = Column(String(256), nullable=False)
    target_metric = Column(String(64), nullable=False)
    period_mode = Column(String(64), nullable=False)
    args_json = Column(JSON_TYPE(), nullable=False)
    git_sha_app = Column(String(64), nullable=True)
    git_sha_strat = Column(String(64), nullable=True)
    data_source = Column(String(128), nullable=True)
    plan_hash = Column(String(128), nullable=False, unique=True, index=True)
    status = Column(String(32), nullable=False, default="running")
    error_summary = Column(Text, nullable=True)


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    result_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(
        String(36),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol = Column(String(64), nullable=False, index=True)
    strategy = Column(String(256), nullable=False, index=True)
    interval = Column(String(32), nullable=False, index=True)
    start_at_utc = Column(DateTime(timezone=True), nullable=True)
    end_at_utc = Column(DateTime(timezone=True), nullable=True)
    rank_in_symbol = Column(Integer, nullable=True)
    metrics = Column(JSON_TYPE(), nullable=False)
    engine_ctx = Column(JSON_TYPE(), nullable=True)
    trades_raw = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "symbol",
            "strategy",
            "interval",
            name="uq_run_symbol_strategy_interval",
        ),
    )

    run = relationship("Run", backref="results")


class Trade(Base):
    __tablename__ = "trades"
    trade_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    result_id = Column(
        String(36),
        ForeignKey("backtest_results.result_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trade_index = Column(Integer, nullable=False)
    # Optional timestamps for entry/exit (UTC)
    entry_time = Column(DateTime(timezone=True), nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    size = Column(String(64), nullable=True)
    entry_bar = Column(Integer, nullable=True)
    exit_bar = Column(Integer, nullable=True)
    entry_price = Column(String(64), nullable=True)
    exit_price = Column(String(64), nullable=True)
    pnl = Column(String(64), nullable=True)
    duration = Column(String(64), nullable=True)
    tag = Column(String(128), nullable=True)
    entry_signals = Column(Text, nullable=True)
    exit_signals = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("result_id", "trade_index", name="uq_result_trade_index"),
    )


class SymbolAggregate(Base):
    __tablename__ = "symbol_aggregates"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(
        String(36),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol = Column(String(64), nullable=False)
    best_by = Column(String(64), nullable=False)
    best_result = Column(
        String(36),
        ForeignKey("backtest_results.result_id", ondelete="CASCADE"),
        nullable=False,
    )
    summary = Column(JSON_TYPE(), nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "best_by", name="uq_run_symbol_bestby"),
    )


class RunArtifact(Base):
    __tablename__ = "run_artifacts"
    artifact_id = Column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    run_id = Column(
        String(36),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type = Column(String(64), nullable=False)
    path_or_uri = Column(Text, nullable=False)
    meta = Column(JSON_TYPE(), nullable=True)


class BestStrategy(Base):
    """Best performing strategy for each symbol/timeframe combination (lightweight)."""

    __tablename__ = "best_strategies"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", name="uq_best_symbol_timeframe"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(64), nullable=False, index=True)
    timeframe = Column(String(32), nullable=False, index=True)
    strategy = Column(String(256), nullable=False)

    # Performance metrics
    sortino_ratio = Column(
        SQLJSON().type if False else SQLJSON, nullable=True
    )  # keep flexible; actual usage stores numbers
    calmar_ratio = Column(SQLJSON().type if False else SQLJSON, nullable=True)
    sharpe_ratio = Column(SQLJSON().type if False else SQLJSON, nullable=True)
    total_return = Column(SQLJSON().type if False else SQLJSON, nullable=True)
    max_drawdown = Column(SQLJSON().type if False else SQLJSON, nullable=True)

    backtest_result_id = Column(String(36), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)


def create_tables():
    Base.metadata.create_all(ENGINE)
    # Best-effort migration: ensure new optional columns exist
    try:
        _ensure_trade_time_columns()
    except Exception:
        pass


def drop_tables():
    """Drop all tables for a full reset (dangerous)."""
    try:
        Base.metadata.drop_all(ENGINE)
    except Exception:
        # best-effort; caller can recreate afterwards
        pass


def _ensure_trade_time_columns() -> None:
    """Add entry_time and exit_time columns to trades if missing (best-effort).

    Uses SQLAlchemy Inspector to detect existing columns. Adds TIMESTAMPTZ for Postgres
    and TEXT for SQLite (stored as ISO strings).
    """
    try:
        from sqlalchemy import inspect, text

        insp = inspect(ENGINE)
        cols = {c.get("name") for c in insp.get_columns("trades")}
        to_add = []
        if "entry_time" not in cols:
            to_add.append("entry_time")
        if "exit_time" not in cols:
            to_add.append("exit_time")
        if not to_add:
            return
        url = str(ENGINE.url).lower() if ENGINE and ENGINE.url else ""
        with ENGINE.begin() as conn:
            for col in to_add:
                if "postgres" in url or "psql" in url:
                    conn.execute(
                        text(
                            f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col} TIMESTAMPTZ NULL"
                        )
                    )
                else:
                    # SQLite and others: check again to avoid errors, then add as TEXT
                    if col not in {c.get("name") for c in insp.get_columns("trades")}:
                        conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col} TEXT"))
    except Exception:
        # Silent; optional migration
        pass


# Convenience helpers used by CLI
def create_run_from_manifest(manifest: Dict[str, Any]) -> Optional[Run]:
    """
    Insert a Run row from manifest dict. If a run with same plan_hash exists, return it.
    """
    sess = Session()
    plan_hash = manifest.get("plan", {}).get("plan_hash")
    if not plan_hash:
        raise ValueError("Manifest missing plan.plan_hash")
    try:
        existing = sess.query(Run).filter(Run.plan_hash == plan_hash).one_or_none()
        if existing:
            return existing
        # Defensive truncation to avoid DB column length violations (e.g., long strategy lists)
        try:
            strategies_raw = manifest["plan"].get("strategies", [])
            if isinstance(strategies_raw, (list, tuple)):
                strategies_mode_raw = ",".join([str(s) for s in strategies_raw])
            else:
                strategies_mode_raw = str(strategies_raw)
        except Exception:
            strategies_mode_raw = ""

        if len(strategies_mode_raw) > 256:
            strategies_mode = strategies_mode_raw[:252] + "..."
        else:
            strategies_mode = strategies_mode_raw

        try:
            intervals_raw = manifest["plan"].get("intervals", [])
            if isinstance(intervals_raw, (list, tuple)):
                intervals_mode_raw = ",".join([str(i) for i in intervals_raw])
            else:
                intervals_mode_raw = str(intervals_raw)
        except Exception:
            intervals_mode_raw = ""

        if len(intervals_mode_raw) > 256:
            intervals_mode = intervals_mode_raw[:252] + "..."
        else:
            intervals_mode = intervals_mode_raw

        run = Run(
            actor=manifest["plan"].get("actor", "cli"),
            action=manifest["plan"].get("action", "backtest"),
            collection_ref=manifest["plan"].get("collection", ""),
            strategies_mode=strategies_mode,
            intervals_mode=intervals_mode,
            target_metric=manifest["plan"].get("metric", ""),
            period_mode=manifest["plan"].get("period_mode", ""),
            args_json=manifest["plan"],
            git_sha_app=manifest["plan"].get("git_sha_app"),
            git_sha_strat=manifest["plan"].get("git_sha_strat"),
            plan_hash=plan_hash,
            status="running",
        )
        sess.add(run)
        sess.commit()
        return run
    except IntegrityError:
        sess.rollback()
        return sess.query(Run).filter(Run.plan_hash == plan_hash).one_or_none()
    finally:
        sess.close()


def ensure_run_for_manifest(manifest: Dict[str, Any]) -> Optional[Run]:
    """
    Ensure a Run exists for the given manifest.
    Tries create_run_from_manifest first. If that fails, attempts a manual upsert.
    Returns a Run instance or None on failure.
    """
    plan_hash = manifest.get("plan", {}).get("plan_hash")
    if not plan_hash:
        return None

    # First try the existing helper which handles most common cases
    try:
        run = create_run_from_manifest(manifest)
        if run:
            return run
    except Exception:
        # fall through to manual attempt
        pass

    sess = Session()
    try:
        # Try to find existing run by plan_hash
        existing = sess.query(Run).filter(Run.plan_hash == plan_hash).one_or_none()
        if existing:
            return existing

        # Build a minimal Run object from manifest safely
        plan = manifest.get("plan", {}) or {}
        # Defensive truncation for safety when constructing Run from manifest 'plan'
        try:
            strategies_raw = plan.get("strategies", [])
            if isinstance(strategies_raw, (list, tuple)):
                strategies_mode_raw = ",".join([str(s) for s in strategies_raw])
            else:
                strategies_mode_raw = str(strategies_raw)
        except Exception:
            strategies_mode_raw = ""

        if len(strategies_mode_raw) > 256:
            strategies_mode = strategies_mode_raw[:252] + "..."
        else:
            strategies_mode = strategies_mode_raw

        try:
            intervals_raw = plan.get("intervals", [])
            if isinstance(intervals_raw, (list, tuple)):
                intervals_mode_raw = ",".join([str(i) for i in intervals_raw])
            else:
                intervals_mode_raw = str(intervals_raw)
        except Exception:
            intervals_mode_raw = ""

        if len(intervals_mode_raw) > 256:
            intervals_mode = intervals_mode_raw[:252] + "..."
        else:
            intervals_mode = intervals_mode_raw

        run = Run(
            actor=plan.get("actor", "cli"),
            action=plan.get("action", "backtest"),
            collection_ref=plan.get("collection", ""),
            strategies_mode=strategies_mode,
            intervals_mode=intervals_mode,
            target_metric=plan.get("metric", ""),
            period_mode=plan.get("period_mode", ""),
            args_json=plan,
            plan_hash=plan_hash,
            status="running",
        )
        sess.add(run)
        sess.commit()
        return run
    except IntegrityError:
        # If another process inserted concurrently, return that row
        try:
            sess.rollback()
            return sess.query(Run).filter(Run.plan_hash == plan_hash).one_or_none()
        except Exception:
            sess.rollback()
            return None
    except Exception:
        try:
            sess.rollback()
        except Exception:
            pass
        return None
    finally:
        sess.close()


def find_run_by_plan_hash(plan_hash: str) -> Optional[Run]:
    sess = Session()
    try:
        return sess.query(Run).filter(Run.plan_hash == plan_hash).one_or_none()
    finally:
        sess.close()
