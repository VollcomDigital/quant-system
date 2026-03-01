from __future__ import annotations

import importlib
import inspect
import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import CollectionConfig, Config
from ..data.alpaca_source import AlpacaSource
from ..data.alphavantage_source import AlphaVantageSource
from ..data.base import DataSource
from ..data.ccxt_source import CCXTSource
from ..data.finnhub_source import FinnhubSource
from ..data.polygon_source import PolygonSource
from ..data.tiingo_source import TiingoSource
from ..data.twelvedata_source import TwelveDataSource
from ..data.yfinance_source import YFinanceSource
from ..strategies.base import BaseStrategy
from ..strategies.registry import discover_external_strategies
from ..utils.telemetry import get_logger, log_json, time_block
from .metrics import (
    omega_ratio,
    pain_index,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
)
from .results_cache import ResultsCache

StageName = Literal[
    "created",
    "collection_validation",
    "data_fetch",
    "data_validation",
    "data_preparation",
    "strategy_optimization",
    "strategy_validation",
]


@dataclass
class BestResult:
    collection: str
    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    stats: dict[str, Any]


@dataclass
class JobContext:
    collection: CollectionConfig
    symbol: str
    timeframe: str
    source: str


@dataclass
class GateDecision:
    passed: bool
    action: Literal["continue", "skip_job", "skip_optimization", "reject_result"]
    reasons: list[str]
    stage: StageName


@dataclass
class JobState:
    job: JobContext
    current_stage: StageName = "created"
    policy_skip_optimization: bool = False
    decisions: dict[StageName, GateDecision] = field(default_factory=dict)
    reasons_by_stage: dict[StageName, list[str]] = field(default_factory=dict)


@dataclass
class FetchedData:
    raw_df: pd.DataFrame


@dataclass
class ValidatedData:
    raw_df: pd.DataFrame
    continuity: dict[str, float | int]
    reliability_on_fail: str
    reliability_reasons: list[str]


@dataclass
class ExecutionPreparedData:
    data_frame: pd.DataFrame
    dates: pd.DatetimeIndex
    fees: float
    slippage: float
    fractional: bool
    bars_per_year: int
    fingerprint: str


@dataclass
class StrategyPlan:
    strategy: BaseStrategy
    fixed_params: dict[str, Any]
    search_space: dict[str, list[Any]]
    search_method: str
    trials_target: int
    skip_optimization: bool = False
    optimization_skip_reasons: list[str] = field(default_factory=list)
    optimization_skip_reason: str | None = None
    optimization_details: dict[str, Any] | None = None
    best_val: float = float("-inf")
    best_params: dict[str, Any] | None = None
    best_stats: dict[str, Any] | None = None
    evaluations: int = 0


@dataclass
class StrategyEvalOutcome:
    best_val: float
    best_params: dict[str, Any] | None
    best_stats: dict[str, Any] | None
    evaluations: int
    skipped_reason: str | None
    strategy: str
    job: JobContext


class BacktestRunner:
    def __init__(self, cfg: Config, strategies_root: Path, run_id: str | None = None):
        self.cfg = cfg
        self.strategies_root = strategies_root
        self.external_index = discover_external_strategies(strategies_root)
        self.results_cache = ResultsCache(Path(self.cfg.cache_dir).parent / "results")
        self.run_id = run_id
        self.logger = get_logger()
        self._pybroker_components: tuple[Any, ...] | None = None
        self._cache_write_failures = 0
        self._strategy_overrides: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []

    def _ensure_pybroker(self) -> tuple[Any, ...]:
        if self._pybroker_components is None:
            try:
                pybroker = importlib.import_module("pybroker")
                common = importlib.import_module("pybroker.common")
                strategy_cls = pybroker.Strategy
                config_cls = pybroker.StrategyConfig
                fee_mode_cls = common.FeeMode
                price_type_cls = common.PriceType
                data_col_enum = common.DataCol
                try:
                    pybroker.disable_logging()
                except Exception:
                    pass
                try:
                    pybroker.disable_progress_bar()
                except Exception:
                    pass
                self._pybroker_components = (
                    strategy_cls,
                    config_cls,
                    fee_mode_cls,
                    price_type_cls,
                    data_col_enum,
                )
            except Exception as exc:  # pragma: no cover - sanity guard
                raise RuntimeError("PyBroker must be installed to run backtests.") from exc
        return self._pybroker_components

    def _cache_set(self, **kwargs: Any) -> None:
        try:
            self.results_cache.set(**kwargs)
        except Exception as exc:
            self._cache_write_failures += 1
            if self._cache_write_failures <= 3:
                self.logger.warning("results cache write failed", exc_info=exc)
            elif self._cache_write_failures == 4:
                self.logger.warning(
                    "results cache write failures continuing; suppressing further warnings"
                )

    def _make_source(self, col: CollectionConfig) -> DataSource:
        cache_dir = Path(self.cfg.cache_dir)
        src = col.source.lower()
        if src == "yfinance":
            return YFinanceSource(cache_dir)
        if src in ("ccxt", "binance", "bybit"):
            if not col.exchange:
                exchange = src if src != "ccxt" else None
                if not exchange:
                    raise ValueError("exchange is required for ccxt collection")
                return CCXTSource(exchange, cache_dir)
            return CCXTSource(col.exchange, cache_dir)
        if src == "polygon":
            return PolygonSource(cache_dir)
        if src == "tiingo":
            return TiingoSource(cache_dir)
        if src == "alpaca":
            return AlpacaSource(cache_dir)
        if src == "finnhub":
            return FinnhubSource(cache_dir)
        if src == "twelvedata":
            return TwelveDataSource(cache_dir)
        if src == "alphavantage":
            return AlphaVantageSource(cache_dir)
        raise ValueError(f"Unsupported data source: {col.source}")

    def _fees_slippage_for(self, col: CollectionConfig) -> tuple[float, float]:
        if col.fees is not None or col.slippage is not None:
            return (
                col.fees if col.fees is not None else self.cfg.fees,
                col.slippage if col.slippage is not None else self.cfg.slippage,
            )
        src = col.source.lower()
        if src in ("binance", "bybit", "ccxt"):
            return (0.0006, 0.0005)
        return (0.0005, 0.0005)

    @staticmethod
    def _bars_per_year(timeframe: str) -> int:
        tf = timeframe.strip().lower()
        if not tf:
            return 252
        digits = "".join(ch for ch in tf if ch.isdigit())
        unit = tf[len(digits) :].strip() or "d"
        value = int(digits) if digits else 1
        value = max(1, value)
        if unit in {"d", "day", "days"}:
            return max(1, int(round(252 / value)))
        if unit in {"w", "week", "weeks"}:
            return max(1, int(round(52 / value)))
        if unit in {"h", "hour", "hours"}:
            return max(1, int(round((24 * 365) / value)))
        if unit in {"m", "min", "minute", "minutes"}:
            return max(1, int(round((60 * 24 * 365) / value)))
        if unit in {"s", "sec", "second", "seconds"}:
            return max(1, int(round((60 * 60 * 24 * 365) / value)))
        return 252

    @staticmethod
    def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta | None:
        tf = (timeframe or "").strip().lower()
        if not tf:
            return None
        digits = "".join(ch for ch in tf if ch.isdigit())
        unit = tf[len(digits) :].strip() or "d"
        value = int(digits) if digits else 1
        value = max(1, value)
        if unit in {"d", "day", "days"}:
            return pd.Timedelta(days=value)
        if unit in {"w", "week", "weeks"}:
            return pd.Timedelta(weeks=value)
        if unit in {"h", "hour", "hours"}:
            return pd.Timedelta(hours=value)
        if unit in {"m", "min", "minute", "minutes"}:
            return pd.Timedelta(minutes=value)
        if unit in {"s", "sec", "second", "seconds"}:
            return pd.Timedelta(seconds=value)
        return None

    @classmethod
    def compute_continuity_score(
        cls,
        df: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, float | int]:
        raw_idx = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values()
        actual_bars = int(len(raw_idx))
        idx = raw_idx
        if idx.has_duplicates:
            idx = idx[~idx.duplicated(keep="first")]
        unique_bars = int(len(idx))
        duplicate_bars = max(0, actual_bars - unique_bars)

        if unique_bars <= 1:
            raise ValueError(
                "insufficient_bars_for_continuity: at least 2 bars are required"
            )

        expected_delta = cls._timeframe_to_timedelta(timeframe)
        if expected_delta is None:
            raise ValueError(f"unsupported_timeframe_for_continuity: {timeframe}")

        diffs = pd.Series(idx[1:] - idx[:-1])
        missing_bars = 0
        largest_gap_bars = 0
        for diff in diffs:
            if diff <= expected_delta:
                continue
            gap_bars = int(diff / expected_delta) - 1
            if gap_bars <= 0:
                continue
            missing_bars += gap_bars
            largest_gap_bars = max(largest_gap_bars, gap_bars)

        expected_bars = unique_bars + missing_bars
        coverage_ratio = 1.0 if expected_bars <= 0 else float(unique_bars / expected_bars)
        coverage_ratio = max(0.0, min(1.0, coverage_ratio))
        missing_ratio = 0.0 if expected_bars <= 0 else (missing_bars / expected_bars)
        largest_gap_ratio = 0.0 if expected_bars <= 0 else (largest_gap_bars / expected_bars)
        duplicate_ratio = 0.0 if actual_bars <= 0 else (duplicate_bars / actual_bars)
        score = max(
            0.0, 1.0 - (0.60 * missing_ratio + 0.25 * largest_gap_ratio + 0.15 * duplicate_ratio)
        )

        return {
            "score": float(score),
            "coverage_ratio": float(coverage_ratio),
            "expected_bars": int(expected_bars),
            "actual_bars": int(actual_bars),
            "unique_bars": int(unique_bars),
            "duplicate_bars": int(duplicate_bars),
            "missing_bars": int(missing_bars),
            "largest_gap_bars": int(largest_gap_bars),
        }

    @staticmethod
    def _sample_series(series: pd.Series, max_points: int = 500) -> list[dict[str, float]]:
        if series.empty:
            return []
        cleaned = series.dropna()
        if cleaned.empty:
            return []
        step = max(1, len(cleaned) // max_points)
        sampled: list[dict[str, float]] = []
        for idx in range(0, len(cleaned), step):
            ts = cleaned.index[idx]
            ts_val = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            sampled.append({"ts": ts_val, "value": float(cleaned.iloc[idx])})
        last_ts = cleaned.index[-1]
        last_val = cleaned.iloc[-1]
        last_key = last_ts.isoformat() if isinstance(last_ts, datetime) else str(last_ts)
        if not sampled or sampled[-1]["ts"] != last_key:
            sampled.append({"ts": last_key, "value": float(last_val)})
        return sampled

    @staticmethod
    def _convert_decimal(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        return value

    @staticmethod
    def _fractional_enabled(col: CollectionConfig, symbol: str) -> bool:
        src = col.source.lower()
        if src in {"binance", "bybit", "ccxt"}:
            return True
        return "/" in symbol or symbol.endswith("USDT")

    @staticmethod
    def _prepare_pybroker_frame(
        df: pd.DataFrame, symbol: str, data_col_enum: Any
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        if df.empty:
            raise ValueError("No price data available for backtest.")
        working = df.copy()
        if not working.index.is_monotonic_increasing:
            working = working.sort_index()
        rename_map = {}
        for column in working.columns:
            lowered = column.lower()
            if lowered in {"open", "high", "low", "close", "volume"}:
                rename_map[column] = lowered
        working = working.rename(columns=rename_map)
        required = {"open", "high", "low", "close"}
        missing = sorted(required - set(map(str.lower, working.columns)))
        if missing:
            raise ValueError(f"Missing price columns: {', '.join(missing)}")
        if "volume" not in map(str.lower, working.columns):
            working["volume"] = 0.0
        dates = pd.to_datetime(working.index)
        working[data_col_enum.DATE.value] = dates
        working[data_col_enum.SYMBOL.value] = symbol
        ordered_cols = [
            data_col_enum.SYMBOL.value,
            data_col_enum.DATE.value,
            data_col_enum.OPEN.value,
            data_col_enum.HIGH.value,
            data_col_enum.LOW.value,
            data_col_enum.CLOSE.value,
            data_col_enum.VOLUME.value,
        ]
        missing_cols = [col for col in ordered_cols if col not in working.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns after normalization: {missing_cols}")
        ordered = pd.DataFrame({col: working[col].to_numpy() for col in ordered_cols})
        return ordered, dates

    def _compute_cagr(
        self,
        equity_curve: pd.Series,
        dates: pd.DatetimeIndex,
        periods_per_year: int,
    ) -> float:
        if equity_curve.empty or len(equity_curve) < 2:
            return float("nan")
        start = equity_curve.iloc[0]
        end = equity_curve.iloc[-1]
        if start <= 0 or end <= 0:
            return float("nan")
        delta_days = (dates[-1] - dates[0]).days
        years = delta_days / 365.25 if delta_days > 0 else 0.0
        if years <= 0 and periods_per_year > 0:
            years = len(equity_curve) / periods_per_year
        if years <= 0:
            return float("nan")
        try:
            return float(end ** (1 / years) - 1.0)
        except Exception:  # pragma: no cover - guard against numerical issues
            return float("nan")

    def _grid(self, grid: dict[str, Iterable[Any]]):
        if not grid:
            yield {}
            return
        keys = list(grid.keys())
        for values in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, values, strict=False))

    def _evaluate_metric(
        self,
        metric: str,
        returns: pd.Series,
        equity: pd.Series,
        periods_per_year: int,
    ) -> float:
        metric = metric.lower()
        if metric == "sharpe":
            return sharpe_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        if metric == "sortino":
            return sortino_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        if metric == "profit":
            return total_return(equity)
        raise ValueError(f"Unknown metric: {metric}")

    def _run_pybroker_simulation(
        self,
        data: pd.DataFrame,
        dates: pd.DatetimeIndex,
        symbol: str,
        entries: pd.Series,
        exits: pd.Series,
        fee_percent: float,
        slippage_percent: float,
        timeframe: str,
        fractional: bool,
        periods_per_year: int,
    ) -> tuple[pd.Series, pd.Series, dict[str, Any]] | None:
        strategy_cls, config_cls, fee_mode_cls, price_type_cls, data_col_enum = (
            self._ensure_pybroker()
        )
        fee_total = max(0.0, fee_percent + slippage_percent)
        config_kwargs: dict[str, Any] = {
            "initial_cash": 10_000.0,
            "subtract_fees": True,
            "enable_fractional_shares": fractional,
            "exit_on_last_bar": True,
            "exit_cover_fill_price": price_type_cls.CLOSE,
            "exit_sell_fill_price": price_type_cls.CLOSE,
            "bars_per_year": periods_per_year,
        }
        if fee_total > 0:
            config_kwargs["fee_mode"] = fee_mode_cls.ORDER_PERCENT
            config_kwargs["fee_amount"] = fee_total
        try:
            allowed = set(inspect.signature(config_cls).parameters)
            config_kwargs = {k: v for k, v in config_kwargs.items() if k in allowed}
        except (TypeError, ValueError):
            pass
        strategy = strategy_cls(
            data,
            start_date=data[data_col_enum.DATE.value].iloc[0],
            end_date=data[data_col_enum.DATE.value].iloc[-1],
            config=config_cls(**config_kwargs),
        )

        aligned_entries = entries.reindex(dates, fill_value=False).astype(bool).to_numpy()
        aligned_exits = exits.reindex(dates, fill_value=False).astype(bool).to_numpy()

        def exec_fn(ctx):
            idx = ctx.bars - 1
            if idx < 0 or idx >= len(aligned_entries):
                return
            if aligned_entries[idx] and ctx.long_pos() is None:
                ctx.buy_fill_price = price_type_cls.CLOSE
                ctx.buy_shares = ctx.calc_target_shares(target_size=1.0)
            if aligned_exits[idx] and ctx.long_pos() is not None:
                ctx.sell_fill_price = price_type_cls.CLOSE
                ctx.sell_all_shares()

        strategy.add_execution(exec_fn, symbol)

        try:
            result = strategy.backtest(calc_bootstrap=False)
        except Exception:
            return None

        portfolio_df = result.portfolio
        if not isinstance(portfolio_df, pd.DataFrame):
            portfolio_df = pd.DataFrame(portfolio_df)
        if portfolio_df.empty:
            return None
        portfolio_df = portfolio_df.copy()
        if "date" not in portfolio_df.columns:
            if isinstance(portfolio_df.index, pd.DatetimeIndex):
                portfolio_df = portfolio_df.reset_index()
                if "index" in portfolio_df.columns and "date" not in portfolio_df.columns:
                    portfolio_df = portfolio_df.rename(columns={"index": "date"})
            else:
                portfolio_df["date"] = pd.to_datetime(dates[: len(portfolio_df)])
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        equity_series = (
            portfolio_df.set_index("date")["equity"].astype(float).groupby(level=0).last()
        )
        equity_series = equity_series.reindex(pd.to_datetime(dates))
        equity_series = equity_series.ffill().bfill()
        if equity_series.isna().all():
            return None
        equity_series.index = dates
        returns = equity_series.pct_change().fillna(0.0)
        equity_curve = (1.0 + returns).cumprod()
        drawdown_series = equity_curve / equity_curve.cummax() - 1.0

        trades_df = result.trades
        if isinstance(trades_df, pd.DataFrame):
            trades_frame = trades_df.copy()
        elif isinstance(trades_df, list):
            trades_frame = pd.DataFrame(trades_df)
        else:
            trades_frame = pd.DataFrame()

        trades_records: list[dict[str, Any]] = []
        if not trades_frame.empty:
            for column in trades_frame.columns:
                trades_frame[column] = trades_frame[column].map(self._convert_decimal)
                if pd.api.types.is_datetime64_any_dtype(trades_frame[column]):
                    trades_frame[column] = trades_frame[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
            trades_records = trades_frame.head(50).to_dict("records")

        trade_count = int(getattr(result.metrics, "trade_count", len(trades_frame)))
        omega = float(omega_ratio(returns))
        tail = float(tail_ratio(returns))
        sharpe_val = float(
            sharpe_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )
        sortino_val = float(
            sortino_ratio(
                returns,
                risk_free_rate=self.cfg.risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )
        profit = float(total_return(equity_curve))
        pain = float(pain_index(equity_curve))
        max_dd = float(drawdown_series.min()) if not drawdown_series.empty else float("nan")
        cagr = self._compute_cagr(equity_curve, dates, periods_per_year)
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 and np.isfinite(cagr) else float("nan")

        stats = {
            "sharpe": sharpe_val,
            "sortino": sortino_val,
            "omega": omega,
            "tail_ratio": tail,
            "profit": profit,
            "pain_index": pain,
            "trades": trade_count,
            "max_drawdown": max_dd,
            "cagr": cagr,
            "calmar": calmar,
            "equity_curve": self._sample_series(equity_curve),
            "drawdown_curve": self._sample_series(drawdown_series),
            "trades_log": trades_records,
        }
        return returns, equity_curve, stats

    def _failure_record(self, payload: dict[str, Any]) -> None:
        self.failures.append(payload)

    @staticmethod
    def _job_log_context(job: JobContext) -> dict[str, Any]:
        return {
            "collection": job.collection.name,
            "symbol": job.symbol,
            "timeframe": job.timeframe,
            "source": job.source,
        }

    def _gate_log(self, stage: StageName, decision: GateDecision, context: dict[str, Any]) -> None:
        log_json(
            self.logger,
            f"{stage}_gate",
            passed=decision.passed,
            action=decision.action,
            reasons=decision.reasons,
            **context,
        )

    def _apply_gate_to_state(self, state: JobState, decision: GateDecision) -> None:
        state.current_stage = decision.stage
        state.decisions[decision.stage] = decision
        state.reasons_by_stage[decision.stage] = list(decision.reasons)
        if decision.action == "skip_optimization":
            state.policy_skip_optimization = True

    def _handle_gate_decision(
        self,
        state: JobState,
        decision: GateDecision,
        context_extra: dict[str, Any] | None = None,
        record_failure: bool = True,
    ) -> GateDecision:
        context = self._job_log_context(state.job)
        if context_extra:
            context |= context_extra
        self._apply_gate_to_state(state, decision)
        if not decision.passed or decision.action != "continue":
            self._gate_log(decision.stage, decision, context)
            if record_failure and decision.action in {"skip_job", "reject_result"}:
                failure: dict[str, Any] = {
                    **self._job_log_context(state.job),
                    "stage": decision.stage,
                    "error": "; ".join(decision.reasons) if decision.reasons else "gate_failed",
                }
                if context_extra and "strategy" in context_extra:
                    failure["strategy"] = context_extra["strategy"]
                self._failure_record(failure)
        return decision

    @staticmethod
    def _plan_add_skip_reason(plan: StrategyPlan, reason: str) -> None:
        if reason not in plan.optimization_skip_reasons:
            plan.optimization_skip_reasons.append(reason)
        plan.skip_optimization = bool(plan.optimization_skip_reasons)
        plan.optimization_skip_reason = (
            "; ".join(plan.optimization_skip_reasons) if plan.optimization_skip_reasons else None
        )

    def _apply_policy_constraints_to_plan(
        self,
        state: JobState,
        validated_data: ValidatedData,
        plan: StrategyPlan,
    ) -> None:
        if not state.policy_skip_optimization:
            return
        self._plan_add_skip_reason(plan, "reliability_threshold_skip_optimization")
        if not isinstance(plan.optimization_details, dict):
            plan.optimization_details = {}
        plan.optimization_details["reliability_reasons"] = list(validated_data.reliability_reasons)

    def _create_job_list(self) -> list[JobContext]:
        # Expand config into executable collection/symbol/timeframe jobs.
        jobs: list[JobContext] = []
        for col in self.cfg.collections:
            for symbol in col.symbols:
                for timeframe in self.cfg.timeframes:
                    jobs.append(
                        JobContext(collection=col, symbol=symbol, timeframe=timeframe, source=col.source)
                    )
        return jobs

    def _collection_validation(self, state: JobState) -> GateDecision:
        # Validate collection-level prerequisites once before processing jobs.
        try:
            self._make_source(state.job.collection)
            return GateDecision(True, "continue", [], "collection_validation")
        except Exception as exc:
            return GateDecision(False, "skip_job", [str(exc)], "collection_validation")

    def _data_fetch(self, job: JobContext, only_cached: bool) -> tuple[GateDecision, FetchedData | None]:
        # Fetch raw market data for a single job from source/cache
        try:
            source = self._make_source(job.collection)
        except Exception as exc:
            decision = GateDecision(False, "skip_job", [str(exc)], "data_fetch")
            return decision, None

        with time_block(
            self.logger,
            "data_fetch",
            collection=job.collection.name,
            symbol=job.symbol,
            timeframe=job.timeframe,
            source=job.source,
        ):
            try:
                df = source.fetch(job.symbol, job.timeframe, only_cached=only_cached)
                decision = GateDecision(True, "continue", [], "data_fetch")
            except Exception as exc:
                decision = GateDecision(False, "skip_job", [str(exc)], "data_fetch")
                return decision, None
        return decision, FetchedData(raw_df=df)

    def _data_validation(
        self, job: JobContext, fetched_data: FetchedData
    ) -> tuple[GateDecision, ValidatedData | None]:
        # Compute continuity and reliability policy decisions.
        if fetched_data.raw_df.empty:
            return GateDecision(False, "skip_job", ["empty_dataframe"], "data_validation"), None

        try:
            continuity = self.compute_continuity_score(fetched_data.raw_df, job.timeframe)
        except ValueError as exc:
            return GateDecision(False, "skip_job", [str(exc)], "data_validation"), None

        reliability_cfg = self.cfg.reliability_thresholds
        reliability_on_fail = "skip_optimization"
        min_data_points = None
        min_continuity_score = None
        if reliability_cfg is not None:
            reliability_on_fail = str(reliability_cfg.on_fail).strip().lower()
            min_data_points = reliability_cfg.min_data_points
            min_continuity_score = reliability_cfg.min_continuity_score
        reliability_reasons: list[str] = []
        if min_continuity_score is not None:
            threshold = float(min_continuity_score)
            continuity_score = float(continuity.get("score", 0.0))
            if continuity_score < threshold:
                reliability_reasons.append(
                    "min_continuity_score_not_met("
                    f"required={threshold}, available={continuity_score})"
                )
        if min_data_points is not None and len(fetched_data.raw_df) < int(min_data_points):
            reliability_reasons.append(
                f"min_data_points_not_met(required={int(min_data_points)}, available={len(fetched_data.raw_df)})"
            )

        if reliability_reasons and reliability_on_fail in {"skip_evaluation", "skip_job"}:
            decision = GateDecision(False, "skip_job", reliability_reasons, "data_validation")
        elif reliability_reasons:
            decision = GateDecision(True, "skip_optimization", reliability_reasons, "data_validation")
        else:
            decision = GateDecision(True, "continue", [], "data_validation")
        # Keep validated diagnostics available even when decision is skip_job.
        validated_data = ValidatedData(
            raw_df=fetched_data.raw_df,
            continuity=continuity,
            reliability_on_fail=reliability_on_fail,
            reliability_reasons=list(reliability_reasons),
        )
        return decision, validated_data

    def _execution_context_prepare(
        self, job: JobContext, validated_data: ValidatedData
    ) -> tuple[GateDecision, ExecutionPreparedData | None]:
        # Prepare pybroker-ready frame and execution metadata for strategy stages.
        try:
            _, _, _, _, data_col_enum = self._ensure_pybroker()
            data_frame, dates = self._prepare_pybroker_frame(validated_data.raw_df, job.symbol, data_col_enum)
            fractional = self._fractional_enabled(job.collection, job.symbol)
            bars_per_year = self._bars_per_year(job.timeframe)
            price = validated_data.raw_df[
                "Close" if "Close" in validated_data.raw_df.columns else "close"
            ].astype(float)
            fingerprint = (
                f"{len(validated_data.raw_df)}:{validated_data.raw_df.index[-1].isoformat()}:{float(price.iloc[-1])}"
            )
            fees, slippage = self._fees_slippage_for(job.collection)
            decision = GateDecision(True, "continue", [], "data_preparation")
        except Exception as exc:
            return GateDecision(False, "skip_job", [str(exc)], "data_preparation"), None
        prepared = ExecutionPreparedData(
            data_frame=data_frame,
            dates=dates,
            fees=fees,
            slippage=slippage,
            fractional=fractional,
            bars_per_year=bars_per_year,
            fingerprint=fingerprint,
        )
        return decision, prepared

    def _strategy_create_plan(
        self,
        _state: JobState,
        strat_name: str,
    ) -> StrategyPlan:
        # Build fixed/search params and search configuration for a strategy.
        StrategyClass = self.external_index[strat_name]
        strategy: BaseStrategy = StrategyClass()
        base_params = self._strategy_overrides.get(strat_name, {})
        grid_override = base_params.get("grid") if isinstance(base_params, dict) else None
        if isinstance(grid_override, dict):
            grid = grid_override
            static_params = {k: v for k, v in base_params.items() if k != "grid"}
        else:
            grid = strategy.param_grid() | base_params
            static_params = {}

        fixed_params = dict(static_params)
        search_space: dict[str, list[Any]] = {}
        for name, values in grid.items():
            options = list(values) if isinstance(values, set | tuple | list) else [values]
            if len(options) <= 1:
                if options:
                    fixed_params[name] = options[0]
            else:
                search_space[name] = options

        search_method = getattr(self.cfg, "param_search", "grid") or "grid"
        trials_target = max(1, int(getattr(self.cfg, "param_trials", 25)))

        return StrategyPlan(
            strategy=strategy,
            fixed_params=fixed_params,
            search_space=search_space,
            search_method=search_method,
            trials_target=trials_target,
        )

    def _strategy_validate_plan(
        self,
        _state: JobState,
        validated_data: ValidatedData,
        plan: StrategyPlan,
    ) -> GateDecision:
        # Decide whether to skip optimization for this strategy plan.
        n_params = len(plan.search_space)
        min_bars_for_optimization = max(self.cfg.param_min_bars, self.cfg.param_dof_multiplier * n_params)
        insufficient_bars = (
            bool(plan.search_space) and len(validated_data.raw_df) < min_bars_for_optimization
        )

        skip_reasons = list(plan.optimization_skip_reasons)
        if insufficient_bars:
            skip_reasons.append("insufficient_bars_for_optimization")

        if skip_reasons:
            plan.skip_optimization = True
            plan.optimization_skip_reasons = skip_reasons
            plan.optimization_skip_reason = "; ".join(skip_reasons)
            if not isinstance(plan.optimization_details, dict):
                plan.optimization_details = {}
            plan.optimization_details["skipped"] = True
            plan.optimization_details["reason"] = skip_reasons[0]
            plan.optimization_details["reasons"] = skip_reasons
            if insufficient_bars:
                plan.optimization_details["min_bars_required"] = min_bars_for_optimization
                plan.optimization_details["bars_available"] = len(validated_data.raw_df)
            return GateDecision(
                passed=True,
                action="skip_optimization",
                reasons=skip_reasons,
                stage="strategy_optimization",
            )

        plan.skip_optimization = False
        plan.optimization_skip_reasons = []
        plan.optimization_skip_reason = None
        plan.optimization_details = None
        return GateDecision(
            passed=True,
            action="continue",
            reasons=[],
            stage="strategy_optimization",
        )

    def _strategy_evaluation(
        self,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
        params: dict[str, Any],
    ) -> float:
        # Evaluate one parameter set, using cache when available.
        full_params = {**plan.fixed_params, **params}
        call_params = full_params.copy()
        try:
            entries, exits = plan.strategy.generate_signals(validated_data.raw_df, call_params)
        except Exception as exc:
            self._failure_record(
                {
                    "collection": state.job.collection.name,
                    "symbol": state.job.symbol,
                    "timeframe": state.job.timeframe,
                    "source": state.job.source,
                    "strategy": plan.strategy.name,
                    "params": full_params,
                    "stage": "generate_signals",
                    "error": str(exc),
                }
            )
            return float("nan")
        entries = entries.reindex(validated_data.raw_df.index, fill_value=False)
        exits = exits.reindex(validated_data.raw_df.index, fill_value=False)

        cached = self.results_cache.get(
            collection=state.job.collection.name,
            symbol=state.job.symbol,
            timeframe=state.job.timeframe,
            strategy=plan.strategy.name,
            params=full_params,
            metric_name=self.cfg.metric,
            data_fingerprint=prepared.fingerprint,
            fees=prepared.fees,
            slippage=prepared.slippage,
        )
        if cached is not None:
            self.metrics["result_cache_hits"] += 1
            plan.evaluations += 1
            val_cached = float(cached["metric_value"])
            cached_stats = dict(cached["stats"])
            self._cache_set(
                collection=state.job.collection.name,
                symbol=state.job.symbol,
                timeframe=state.job.timeframe,
                strategy=plan.strategy.name,
                params=full_params,
                metric_name=self.cfg.metric,
                metric_value=val_cached,
                stats=cached_stats,
                data_fingerprint=prepared.fingerprint,
                fees=prepared.fees,
                slippage=prepared.slippage,
                run_id=self.run_id,
            )
            if val_cached > plan.best_val:
                plan.best_val = val_cached
                plan.best_params = full_params.copy()
                plan.best_stats = cached_stats
            return val_cached

        self.metrics["result_cache_misses"] += 1
        sim_result = self._run_pybroker_simulation(
            prepared.data_frame,
            prepared.dates,
            state.job.symbol,
            entries,
            exits,
            prepared.fees,
            prepared.slippage,
            state.job.timeframe,
            prepared.fractional,
            prepared.bars_per_year,
        )
        if sim_result is None:
            return float("-inf")
        returns, equity_curve, stats = sim_result
        stats = dict(stats)
        if plan.optimization_details is not None:
            stats["optimization"] = plan.optimization_details
        reliability = dict(stats.get("data_reliability", {}))
        reliability["continuity"] = validated_data.continuity
        stats["data_reliability"] = reliability
        self.metrics["param_evals"] += 1
        plan.evaluations += 1
        metric_val = self._evaluate_metric(self.cfg.metric, returns, equity_curve, prepared.bars_per_year)
        if not np.isfinite(metric_val):
            return float("-inf")
        self._cache_set(
            collection=state.job.collection.name,
            symbol=state.job.symbol,
            timeframe=state.job.timeframe,
            strategy=plan.strategy.name,
            params=full_params,
            metric_name=self.cfg.metric,
            metric_value=float(metric_val),
            stats=stats,
            data_fingerprint=prepared.fingerprint,
            fees=prepared.fees,
            slippage=prepared.slippage,
            run_id=self.run_id,
        )
        if metric_val > plan.best_val:
            plan.best_val = metric_val
            plan.best_params = full_params.copy()
            plan.best_stats = stats
        return float(metric_val)

    def _strategy_run(
        self,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
    ) -> StrategyEvalOutcome | None:
        # Run search (optuna/grid) or baseline evaluation and return the best candidate.
        try:
            space_items = list(plan.search_space.items())
            if plan.search_space and not plan.skip_optimization:
                search_method = plan.search_method
                if search_method == "optuna":
                    try:
                        import optuna
                    except Exception:
                        search_method = "grid"
                if search_method == "optuna":

                    def objective(trial, space=space_items):
                        var_params = {
                            name: trial.suggest_categorical(name, options) for name, options in space
                        }
                        result = self._strategy_evaluation(plan, state, validated_data, prepared, var_params)
                        return result if np.isfinite(result) else float("-inf")

                    total_combos = 1
                    for options in plan.search_space.values():
                        total_combos *= max(1, len(options))
                    n_trials = min(plan.trials_target, max(1, total_combos))
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_trials)
                else:
                    for params in self._grid(plan.search_space):
                        self._strategy_evaluation(plan, state, validated_data, prepared, params)
            else:
                self._strategy_evaluation(plan, state, validated_data, prepared, {})
            return StrategyEvalOutcome(
                best_val=float(plan.best_val),
                best_params=plan.best_params,
                best_stats=plan.best_stats,
                evaluations=plan.evaluations,
                skipped_reason=plan.optimization_skip_reason,
                strategy=plan.strategy.name,
                job=state.job,
            )
        except Exception as exc:
            self._failure_record(
                {
                    "collection": state.job.collection.name,
                    "symbol": state.job.symbol,
                    "timeframe": state.job.timeframe,
                    "source": state.job.source,
                    "strategy": plan.strategy.name,
                    "stage": "strategy_optimization",
                    "error": str(exc),
                }
            )
            return None

    def _strategy_validate_results(self, state: JobState, outcome: StrategyEvalOutcome) -> GateDecision:
        # Keep this gate lightweight for now; stricter schema checks can be added later.
        reasons: list[str] = []
        if not np.isfinite(outcome.best_val):
            reasons.append("best_metric_not_finite")
        if outcome.best_params is None:
            reasons.append("missing_best_params")
        if not isinstance(outcome.best_stats, dict) or not outcome.best_stats:
            reasons.append("missing_best_stats")
        if reasons:
            decision = GateDecision(False, "reject_result", reasons, "strategy_validation")
        else:
            decision = GateDecision(True, "continue", [], "strategy_validation")
        return decision

    def run_all(self, only_cached: bool = False) -> list[BestResult]:
        best_results: list[BestResult] = []
        # Initialize per-run counters and transient state.
        self.metrics = {
            "result_cache_hits": 0,
            "result_cache_misses": 0,
            "param_evals": 0,
            "symbols_tested": 0,
            "strategies_used": set(),
        }
        self.failures = []
        self._cache_write_failures = 0
        self._strategy_overrides = (
            {s.name: s.params for s in self.cfg.strategies} if self.cfg.strategies else {}
        )

        jobs = self._create_job_list()
        # Cache collection gate decisions so each collection is validated once per run.
        validated_collections: dict[str, GateDecision] = {}

        for job in jobs:
            state = JobState(job=job)
            collection_key = job.collection.name
            collection_decision = validated_collections.get(collection_key)
            if collection_decision is None:
                collection_decision = self._collection_validation(state)
                collection_decision = self._handle_gate_decision(
                    state,
                    collection_decision,
                )
                validated_collections[collection_key] = collection_decision
            else:
                # Apply cached gate state without re-emitting logs/failure side effects.
                self._apply_gate_to_state(state, collection_decision)
            if not collection_decision.passed:
                continue

            data_fetch_decision, fetched_data = self._data_fetch(state.job, only_cached=only_cached)
            data_fetch_decision = self._handle_gate_decision(
                state,
                data_fetch_decision,
            )
            if not data_fetch_decision.passed or fetched_data is None:
                continue

            data_decision, validated_data = self._data_validation(state.job, fetched_data)
            data_decision = self._handle_gate_decision(
                state,
                data_decision,
            )
            if not data_decision.passed or validated_data is None:
                continue

            prep_decision, prepared = self._execution_context_prepare(state.job, validated_data)
            prep_decision = self._handle_gate_decision(
                state,
                prep_decision,
            )
            if not prep_decision.passed or prepared is None:
                continue

            for strat_name in self.external_index.keys():
                # Strategy stage: create plan -> validate plan -> run -> validate results.
                plan = self._strategy_create_plan(state, strat_name)
                self._apply_policy_constraints_to_plan(state, validated_data, plan)
                plan_decision = self._strategy_validate_plan(state, validated_data, plan)
                _ = self._handle_gate_decision(
                    state,
                    plan_decision,
                    context_extra={
                        "strategy": plan.strategy.name,
                        "search_method": plan.search_method,
                    },
                )  # routing handled by plan.optimization_skip_reason in _strategy_run
                self.metrics["symbols_tested"] += 1
                self.metrics["strategies_used"].add(plan.strategy.name)
                outcome = self._strategy_run(plan, state, validated_data, prepared)
                if outcome is None:
                    continue
                validation_decision = self._handle_gate_decision(
                    state,
                    self._strategy_validate_results(state, outcome),
                    context_extra={"strategy": outcome.strategy},
                )
                if not validation_decision.passed:
                    continue

                best_results.append(
                    BestResult(
                        collection=state.job.collection.name,
                        symbol=state.job.symbol,
                        timeframe=state.job.timeframe,
                        strategy=plan.strategy.name,
                        params=outcome.best_params or {},
                        metric_name=self.cfg.metric,
                        metric_value=float(outcome.best_val),
                        stats=outcome.best_stats or {},
                    )
                )

        if isinstance(self.metrics.get("strategies_used"), set):
            self.metrics["strategies_count"] = len(self.metrics["strategies_used"])  # type: ignore
            self.metrics.pop("strategies_used", None)
        return best_results
