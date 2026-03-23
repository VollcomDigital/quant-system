from __future__ import annotations

import importlib
import inspect
import itertools
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import (
    CollectionConfig,
    Config,
    ResultConsistencyExecutionPriceVarianceConfig,
    ResultConsistencyOutlierDependencyConfig,
    ValidationContinuityConfig,
    ValidationOutlierDetectionConfig,
    ResultConsistencyConfig,
)
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
from .evaluation.adapters import normalized_rows_to_legacy_rows
from .evaluation.contracts import (
    EvaluationCacheRecord,
    EvaluationModeConfig,
    EvaluationRequest,
    ResultRecord,
)
from .evaluation.evaluator import BacktestEvaluator
from .evaluation.store import EvaluationCache, ResultStore
from .metrics import (
    omega_ratio,
    pain_index,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
)
from .results_cache import ResultsCache, ResultsCacheRecord

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
    action: Literal[
        "continue",
        "baseline_only",
        "skip_job",
        "skip_collection",
        "skip_optimization",
        "reject_result",
    ]
    reasons: list[str]
    stage: StageName


@dataclass
class JobState:
    job: JobContext
    current_stage: StageName = "created"
    policy_skip_optimization: bool = False
    validation_config_hash: str = ""
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
    has_valid_candidate: bool
    evaluations: int
    skipped_reason: str | None
    strategy: str
    job: JobContext


@dataclass
class ValidationContext:
    stage: StageName
    state: JobState
    mode: str
    job: JobContext
    fetched_data: FetchedData | None = None
    validated_data: ValidatedData | None = None
    prepared_data: ExecutionPreparedData | None = None
    plan: StrategyPlan | None = None
    outcome: StrategyEvalOutcome | None = None


class BacktestRunner:
    _CRYPTO_SOURCE_NAMES = {"binance", "bybit", "ccxt", "coinbase", "kraken", "okx", "kucoin"}
    _VALIDATION_GATE_IDS = (
        "data_quality.min_required_bars",
        "data_quality.min_data_points",
        "data_quality.continuity.min_score",
        "data_quality.continuity.max_missing_bar_pct",
        "data_quality.kurtosis",
        "data_quality.outlier_detection",
        "optimization.feasibility",
        "result_consistency.outlier_dependency",
        "result_consistency.execution_price_variance",
    )

    def __init__(
        self,
        cfg: Config,
        strategies_root: Path,
        run_id: str | None = None,
        evaluation_mode: str | None = None,
    ):
        self.cfg = cfg
        self.strategies_root = strategies_root
        self.external_index = discover_external_strategies(strategies_root)
        self.results_cache = ResultsCache(Path(self.cfg.cache_dir).parent / "results")
        eval_root = Path(self.cfg.cache_dir).parent / "evaluation"
        self.evaluation_cache = EvaluationCache(eval_root)
        self.result_store = ResultStore(eval_root)
        self.run_id = run_id
        self.logger = get_logger()
        self._pybroker_components: tuple[Any, ...] | None = None
        self._cache_write_failures = 0
        self._strategy_overrides: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []
        requested_mode = (evaluation_mode or getattr(self.cfg, "evaluation_mode", "backtest")).strip().lower()
        if requested_mode not in {"backtest", "walk_forward"}:
            raise ValueError(
                f"Invalid evaluation mode: {requested_mode}. Expected 'backtest' or 'walk_forward'."
            )
        if requested_mode == "walk_forward":
            raise NotImplementedError(
                "evaluation_mode=walk_forward is configured but not implemented yet. "
                "Use evaluation_mode=backtest for now."
            )
        self.evaluation_mode = requested_mode
        self.mode_config = EvaluationModeConfig(mode=requested_mode, payload={})
        self.mode_config_hash = self.evaluation_cache.hash_mode_config(self.mode_config)
        self._result_store_write_failures = 0
        self._evaluation_cache_write_failures = 0
        self.validation_metadata: dict[str, Any] = {}
        self.active_validation_gates: list[str] = []
        self.inactive_validation_gates: list[str] = []
        self._evaluator: BacktestEvaluator | None = None

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
            if isinstance(self.results_cache, ResultsCache):
                record = ResultsCacheRecord.from_mapping(kwargs)
                self.results_cache.set(record=record)
            else:
                self.results_cache.set(**kwargs)
        except Exception as exc:
            self._cache_write_failures += 1
            if self._cache_write_failures <= 3:
                self.logger.warning("results cache write failed", exc_info=exc)
            elif self._cache_write_failures == 4:
                self.logger.warning(
                    "results cache write failures continuing; suppressing further warnings"
                )

    def _evaluation_cache_set(self, **kwargs: Any) -> None:
        try:
            if isinstance(self.evaluation_cache, EvaluationCache):
                record = EvaluationCacheRecord.from_mapping(kwargs)
                self.evaluation_cache.set(record=record)
            else:
                self.evaluation_cache.set(**kwargs)
        except Exception as exc:
            self._evaluation_cache_write_failures += 1
            if self._evaluation_cache_write_failures <= 3:
                self.logger.warning("evaluation cache write failed", exc_info=exc)
            elif self._evaluation_cache_write_failures == 4:
                self.logger.warning(
                    "evaluation cache write failures continuing; suppressing further warnings"
                )

    def _result_store_insert(self, record: ResultRecord) -> None:
        try:
            self.result_store.insert(record)
        except Exception as exc:
            self._result_store_write_failures += 1
            if self._result_store_write_failures <= 3:
                self.logger.warning("result store write failed", exc_info=exc)
            elif self._result_store_write_failures == 4:
                self.logger.warning(
                    "result store write failures continuing; suppressing further warnings"
                )

    def _result_store_upsert_run_metadata(
        self,
        *,
        validation_profile: dict[str, Any],
        active_gates: list[str],
        inactive_gates: list[str],
    ) -> None:
        if self.run_id is None:
            return
        try:
            self.result_store.upsert_run_metadata(
                run_id=self.run_id,
                evaluation_mode=self.mode_config.mode,
                mode_config_hash=self.mode_config_hash,
                validation_profile=validation_profile,
                active_gates=active_gates,
                inactive_gates=inactive_gates,
            )
        except Exception as exc:
            self._result_store_write_failures += 1
            if self._result_store_write_failures <= 3:
                self.logger.warning("run metadata write failed", exc_info=exc)
            elif self._result_store_write_failures == 4:
                self.logger.warning(
                    "result store write failures continuing; suppressing further warnings"
                )

    @staticmethod
    def _serialize_data_quality_profile(data_quality: Any) -> dict[str, Any] | None:
        if data_quality is None:
            return None
        continuity = getattr(data_quality, "continuity", None)
        calendar = getattr(continuity, "calendar", None) if continuity is not None else None
        calendar_payload = None
        if calendar is not None:
            calendar_payload = {
                "kind": getattr(calendar, "kind", None),
                "exchange": getattr(calendar, "exchange", None),
                "timezone": getattr(calendar, "timezone", None),
            }
        outlier = getattr(data_quality, "outlier_detection", None)
        return {
            "on_fail": getattr(data_quality, "on_fail", None),
            "min_data_points": getattr(data_quality, "min_data_points", None),
            "continuity": (
                {
                    "min_score": getattr(continuity, "min_score", None),
                    "max_missing_bar_pct": getattr(continuity, "max_missing_bar_pct", None),
                    "calendar": calendar_payload,
                }
                if continuity is not None
                else None
            ),
            "kurtosis": getattr(data_quality, "kurtosis", None),
            "outlier_detection": (
                {
                    "max_outlier_pct": getattr(outlier, "max_outlier_pct", None),
                    "method": getattr(outlier, "method", None),
                    "zscore_threshold": getattr(outlier, "zscore_threshold", None),
                }
                if outlier is not None
                else None
            ),
        }

    @staticmethod
    def _serialize_optimization_profile(optimization: Any) -> dict[str, Any] | None:
        if optimization is None:
            return None
        return {
            "on_fail": getattr(optimization, "on_fail", None),
            "min_bars": getattr(optimization, "min_bars", None),
            "dof_multiplier": getattr(optimization, "dof_multiplier", None),
        }

    @staticmethod
    def _serialize_result_consistency_profile(result_consistency: Any) -> dict[str, Any] | None:
        if result_consistency is None:
            return None
        outlier_dependency = getattr(result_consistency, "outlier_dependency", None)
        execution_price_variance = getattr(result_consistency, "execution_price_variance", None)
        return {
            "outlier_dependency": (
                {
                    "slices": getattr(outlier_dependency, "slices", None),
                    "profit_share_threshold": getattr(outlier_dependency, "profit_share_threshold", None),
                    "trade_share_threshold": getattr(outlier_dependency, "trade_share_threshold", None),
                }
                if outlier_dependency is not None
                else None
            ),
            "execution_price_variance": (
                {
                    "price_tolerance_bps": getattr(execution_price_variance, "price_tolerance_bps", None),
                }
                if execution_price_variance is not None
                else None
            ),
        }

    @staticmethod
    def _active_data_quality_gates(data_quality: Any) -> set[str]:
        active: set[str] = set()
        if data_quality is None:
            return active
        # Continuity precondition checks (<2 bars, timeframe sanity) run for any
        # configured data-quality policy because continuity diagnostics are always computed.
        active.add("data_quality.min_required_bars")
        if getattr(data_quality, "min_data_points", None) is not None:
            active.add("data_quality.min_data_points")
        continuity = getattr(data_quality, "continuity", None)
        if continuity is not None:
            if getattr(continuity, "min_score", None) is not None:
                active.add("data_quality.continuity.min_score")
            if getattr(continuity, "max_missing_bar_pct", None) is not None:
                active.add("data_quality.continuity.max_missing_bar_pct")
        if getattr(data_quality, "kurtosis", None) is not None:
            active.add("data_quality.kurtosis")
        if getattr(data_quality, "outlier_detection", None) is not None:
            active.add("data_quality.outlier_detection")
        return active

    @staticmethod
    def _active_optimization_gates(optimization: Any) -> set[str]:
        if optimization is None:
            return set()
        return {"optimization.feasibility"}

    @staticmethod
    def _active_result_consistency_gates(result_consistency: Any) -> set[str]:
        if result_consistency is None:
            return set()
        active: set[str] = set()
        if getattr(result_consistency, "outlier_dependency", None) is not None:
            active.add("result_consistency.outlier_dependency")
        if getattr(result_consistency, "execution_price_variance", None) is not None:
            active.add("result_consistency.execution_price_variance")
        return active

    def _build_validation_metadata(self) -> dict[str, Any]:
        collection_profiles: list[dict[str, Any]] = []
        active_gates_union: set[str] = set()

        for collection in self.cfg.collections:
            collection_validation = getattr(collection, "validation", None)
            collection_dq = getattr(collection_validation, "data_quality", None)
            collection_optimization = getattr(collection_validation, "optimization", None)
            collection_result_consistency = getattr(collection_validation, "result_consistency", None)
            collection_active = self._active_data_quality_gates(collection_dq)
            collection_active.update(self._active_optimization_gates(collection_optimization))
            collection_active.update(
                self._active_result_consistency_gates(collection_result_consistency)
            )
            active_gates_union.update(collection_active)
            collection_profiles.append(
                {
                    "collection": collection.name,
                    "source": collection.source,
                    "data_quality": self._serialize_data_quality_profile(collection_dq),
                    "optimization": self._serialize_optimization_profile(collection_optimization),
                    "result_consistency": self._serialize_result_consistency_profile(
                        collection_result_consistency
                    ),
                    "active_gates": sorted(collection_active),
                }
            )

        active_gates = sorted(active_gates_union)
        inactive_gates = sorted(set(self._VALIDATION_GATE_IDS).difference(active_gates_union))
        profile = {
            "collections": collection_profiles,
        }
        return {
            "profile": profile,
            "active_gates": active_gates,
            "inactive_gates": inactive_gates,
        }

    def _build_job_validation_profile(self, collection: CollectionConfig) -> dict[str, Any]:
        collection_validation = getattr(collection, "validation", None)
        collection_dq = getattr(collection_validation, "data_quality", None)
        collection_optimization = getattr(collection_validation, "optimization", None)
        collection_result_consistency = getattr(collection_validation, "result_consistency", None)
        return {
            "data_quality": self._serialize_data_quality_profile(collection_dq),
            "optimization": self._serialize_optimization_profile(collection_optimization),
            "result_consistency": self._serialize_result_consistency_profile(
                collection_result_consistency
            ),
        }

    @staticmethod
    def _hash_validation_profile(profile: dict[str, Any]) -> str:
        payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_evaluator(self) -> BacktestEvaluator:
        if self._evaluator is None:
            # Lazily resolve evaluator once per run to keep monkeypatch flexibility
            # while avoiding per-evaluation object churn.
            self._evaluator = BacktestEvaluator(self._run_pybroker_simulation, self._evaluate_metric)
        return self._evaluator

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
        if unit in {"w", "wk", "wks", "week", "weeks"}:
            return max(1, int(round(52 / value)))
        if unit in {"mo", "mon", "month", "months"}:
            return max(1, int(round(12 / value)))
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
        if unit in {"w", "wk", "wks", "week", "weeks"}:
            return pd.Timedelta(weeks=value)
        if unit in {"mo", "mon", "month", "months"}:
            return pd.Timedelta(days=30 * value)
        if unit in {"h", "hour", "hours"}:
            return pd.Timedelta(hours=value)
        if unit in {"m", "min", "minute", "minutes"}:
            return pd.Timedelta(minutes=value)
        if unit in {"s", "sec", "second", "seconds"}:
            return pd.Timedelta(seconds=value)
        return None

    @staticmethod
    def _parse_timeframe_shape(timeframe: str) -> tuple[int, str, bool, bool]:
        tf = (timeframe or "").strip().lower()
        digits = "".join(ch for ch in tf if ch.isdigit())
        unit = tf[len(digits) :].strip() or "d"
        value = int(digits) if digits else 1
        value = max(1, value)
        is_eod_like = unit in {
            "d",
            "day",
            "days",
            "w",
            "wk",
            "wks",
            "week",
            "weeks",
            "mo",
            "mon",
            "month",
            "months",
        }
        is_daily = unit in {"d", "day", "days"}
        return value, unit, is_eod_like, is_daily

    @staticmethod
    def _count_missing_from_expected_index(
        expected_idx: pd.DatetimeIndex,
        actual_idx: pd.DatetimeIndex,
        *,
        normalize_before_compare: bool = False,
    ) -> tuple[int, int, int]:
        expected_bars = int(len(expected_idx))
        if expected_bars <= 0:
            return 0, 0, 0
        lhs = expected_idx.normalize() if normalize_before_compare else expected_idx
        rhs = actual_idx.normalize() if normalize_before_compare else actual_idx
        present_mask = lhs.isin(rhs)
        missing_mask = np.asarray(~present_mask, dtype=np.int8)
        missing_bars = int(missing_mask.sum())
        if missing_bars == 0:
            largest_gap_bars = 0
        else:
            # Find contiguous runs of missing bars from edge transitions.
            padded = np.concatenate(([0], missing_mask, [0]))
            transitions = np.diff(padded)
            starts = np.nonzero(transitions == 1)[0]
            ends = np.nonzero(transitions == -1)[0]
            largest_gap_bars = int((ends - starts).max())
        return expected_bars, missing_bars, largest_gap_bars

    @staticmethod
    def _count_missing_by_fixed_delta(
        idx: pd.DatetimeIndex, expected_delta: pd.Timedelta
    ) -> tuple[int, int]:
        diffs = idx[1:] - idx[:-1]
        gap_diffs = diffs[diffs > expected_delta]
        if len(gap_diffs) == 0:
            return 0, 0
        gap_bars = np.asarray(gap_diffs // expected_delta, dtype=int) - 1
        positive_gap_bars = gap_bars[gap_bars > 0]
        if positive_gap_bars.size == 0:
            return 0, 0
        return int(positive_gap_bars.sum()), int(positive_gap_bars.max())

    @staticmethod
    def _weekday_expected_index(
        idx: pd.DatetimeIndex, expected_delta: pd.Timedelta
    ) -> pd.DatetimeIndex:
        expected_idx = pd.date_range(start=idx[0], end=idx[-1], freq=expected_delta)
        return expected_idx[expected_idx.dayofweek < 5]

    @classmethod
    def _exchange_daily_expected_counts(
        cls,
        idx: pd.DatetimeIndex,
        *,
        value: int,
        exchange_calendar: str,
        expected_delta: pd.Timedelta,
    ) -> tuple[int, int, int]:
        try:
            import exchange_calendars as xcals

            calendar = xcals.get_calendar(exchange_calendar)
            start_date = pd.Timestamp(idx[0]).date().isoformat()
            end_date = pd.Timestamp(idx[-1]).date().isoformat()
            expected_idx = pd.DatetimeIndex(calendar.sessions_in_range(start_date, end_date))
            if expected_idx.tz is not None:
                expected_idx = expected_idx.tz_localize(None)
            if value > 1 and len(expected_idx) > 0:
                expected_idx = expected_idx[::value]
            actual_idx = idx
            if actual_idx.tz is not None:
                actual_idx = actual_idx.tz_localize(None)
            return cls._count_missing_from_expected_index(
                expected_idx,
                actual_idx,
                normalize_before_compare=True,
            )
        except ModuleNotFoundError:
            # Fallback keeps continuity gate functional when exchange_calendars
            # is not installed; behaves like weekday expectations.
            expected_idx = cls._weekday_expected_index(idx, expected_delta)
            return cls._count_missing_from_expected_index(expected_idx, idx)
        except Exception as exc:
            # Surface invalid exchange-calendar usage as a validation-style error
            # so data-validation gates can convert it to skip decisions.
            raise ValueError(
                f"Failed to use exchange calendar '{exchange_calendar}': {exc}"
            ) from exc

    @classmethod
    def _expected_missing_counts(
        cls,
        idx: pd.DatetimeIndex,
        *,
        timeframe: str,
        calendar_kind: str,
        exchange_calendar: str | None,
        expected_delta: pd.Timedelta,
        unique_bars: int,
    ) -> tuple[int, int, int]:
        value, _, _, is_daily = cls._parse_timeframe_shape(timeframe)
        if calendar_kind == "exchange" and is_daily and exchange_calendar:
            return cls._exchange_daily_expected_counts(
                idx,
                value=value,
                exchange_calendar=exchange_calendar,
                expected_delta=expected_delta,
            )
        if calendar_kind in {"weekday", "exchange"} and is_daily:
            expected_idx = cls._weekday_expected_index(idx, expected_delta)
            return cls._count_missing_from_expected_index(expected_idx, idx)
        missing_bars, largest_gap_bars = cls._count_missing_by_fixed_delta(idx, expected_delta)
        expected_bars = unique_bars + missing_bars
        return expected_bars, missing_bars, largest_gap_bars

    @staticmethod
    def _parse_calendar_timezone(calendar_timezone: str | None) -> timezone | None:
        if calendar_timezone is None:
            return None
        normalized = calendar_timezone.strip().upper()
        if normalized == "UTC":
            return timezone.utc
        try:
            sign = 1 if normalized[3] == "+" else -1
            hours = int(normalized[4:6])
            minutes = int(normalized[7:9])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"unsupported_calendar_timezone: {calendar_timezone}") from exc
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

    @classmethod
    def _normalize_for_calendar_timezone(
        cls, idx: pd.DatetimeIndex, calendar_timezone: str | None
    ) -> pd.DatetimeIndex:
        parsed_timezone = cls._parse_calendar_timezone(calendar_timezone)
        if parsed_timezone is None:
            return idx
        localized_idx = idx.tz_localize(timezone.utc) if idx.tz is None else idx
        return localized_idx.tz_convert(parsed_timezone).tz_localize(None)

    @classmethod
    def compute_continuity_score(
        cls,
        df: pd.DataFrame,
        timeframe: str,
        calendar_kind: str = "crypto_24_7",
        exchange_calendar: str | None = None,
        calendar_timezone: str | None = None,
    ) -> dict[str, float | int]:
        raw_idx = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values()
        idx = cls._normalize_for_calendar_timezone(raw_idx, calendar_timezone).sort_values()
        actual_bars = int(len(idx))
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

        expected_bars, missing_bars, largest_gap_bars = cls._expected_missing_counts(
            idx,
            timeframe=timeframe,
            calendar_kind=calendar_kind,
            exchange_calendar=exchange_calendar,
            expected_delta=expected_delta,
            unique_bars=unique_bars,
        )

        if expected_bars <= 0:
            expected_bars = unique_bars
        coverage_ratio = float(unique_bars / expected_bars)
        coverage_ratio = max(0.0, min(1.0, coverage_ratio))
        missing_ratio = missing_bars / expected_bars
        largest_gap_ratio = largest_gap_bars / expected_bars
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
    def _enrich_evaluation_stats(
        stats: dict[str, Any],
        plan: StrategyPlan,
        validated_data: ValidatedData,
    ) -> dict[str, Any]:
        enriched_stats = dict(stats)
        enriched_stats.pop("optimization", None)
        if plan.optimization_details is not None:
            enriched_stats["optimization"] = dict(plan.optimization_details)
        reliability = dict(enriched_stats.get("data_reliability", {}))
        reliability["continuity"] = validated_data.continuity
        enriched_stats["data_reliability"] = reliability
        return enriched_stats

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
    ) -> tuple[pd.Series, pd.Series, dict[str, Any], pd.DataFrame] | None:
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

        if not trades_frame.empty:
            for column in trades_frame.columns:
                trades_frame[column] = trades_frame[column].map(self._convert_decimal)

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
        }
        return returns, equity_curve, stats, trades_frame

    def _failure_record(self, payload: dict[str, Any]) -> None:
        self.failures.append(payload)

    def _strategy_failure_payload(
        self,
        state: JobState,
        stage: str,
        error: str,
        strategy: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a normalized strategy failure record for summary/report outputs."""
        payload: dict[str, Any] = {
            **self._job_log_context(state.job),
            "strategy": strategy,
            "stage": stage,
            "error": error,
        }
        if params is not None:
            payload["params"] = params
        return payload

    @staticmethod
    def _job_log_context(job: JobContext) -> dict[str, Any]:
        return {
            "collection": job.collection.name,
            "symbol": job.symbol,
            "timeframe": job.timeframe,
            "source": job.source,
        }

    @staticmethod
    def _collection_gate_key(collection: CollectionConfig) -> int:
        # Use object identity so collections sharing a name do not collide.
        return id(collection)

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
        # Job-level data-quality gate can disable optimization for all strategies on this job.
        if decision.action == "skip_optimization":
            state.policy_skip_optimization = True

    def _handle_gate_decision(
        self,
        state: JobState,
        decision: GateDecision,
        context_extra: dict[str, Any] | None = None,
        record_failure: bool = True,
        blocked_collections: set[int] | None = None,
    ) -> GateDecision:
        """Apply, log, and optionally record a non-continue gate decision."""
        context = self._job_log_context(state.job)
        if context_extra:
            context |= context_extra
        self._apply_gate_to_state(state, decision)
        if decision.action == "skip_collection" and blocked_collections is not None:
            blocked_collections.add(self._collection_gate_key(state.job.collection))
        if not decision.passed or decision.action != "continue":
            self._gate_log(decision.stage, decision, context)
            if record_failure and decision.action in {"skip_job", "skip_collection", "reject_result"}:
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
        if not state.policy_skip_optimization or not plan.search_space:
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

    @staticmethod
    def _gate_action_rank(action: str) -> int:
        order = {
            "continue": 0,
            "baseline_only": 1,
            "skip_optimization": 2,
            "skip_job": 3,
            "reject_result": 4,
            "skip_collection": 5,
        }
        return order.get(action, -1)

    def _compose_gate_decisions(self, stage: StageName, *decisions: GateDecision) -> GateDecision:
        reasons: list[str] = []
        chosen = GateDecision(True, "continue", [], stage)
        for decision in decisions:
            if decision.stage != stage:
                continue
            if self._gate_action_rank(decision.action) > self._gate_action_rank(chosen.action):
                chosen = decision
            for reason in decision.reasons:
                if reason not in reasons:
                    reasons.append(reason)
        return GateDecision(
            passed=chosen.passed,
            action=chosen.action,
            reasons=reasons,
            stage=stage,
        )

    def _collection_validation(self, state: JobState) -> tuple[GateDecision, DataSource | None]:
        context = ValidationContext(
            stage="collection_validation",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
        )
        common_decision, source = self._collection_validation_common(context)
        mode_decision = (
            self._collection_validation_backtest(context)
            if context.mode == "backtest"
            else self._collection_validation_walk_forward(context)
        )
        decision = self._compose_gate_decisions(
            "collection_validation", common_decision, mode_decision
        )
        return decision, source

    def _collection_validation_common(
        self, context: ValidationContext
    ) -> tuple[GateDecision, DataSource | None]:
        try:
            source = self._make_source(context.job.collection)
            return GateDecision(True, "continue", [], "collection_validation"), source
        except Exception as exc:
            return GateDecision(False, "skip_job", [str(exc)], "collection_validation"), None

    @staticmethod
    def _collection_validation_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "collection_validation")

    @staticmethod
    def _collection_validation_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "skip_job",
            ["walk_forward_collection_validation_not_implemented"],
            "collection_validation",
        )

    def _data_fetch(
        self,
        state: JobState,
        only_cached: bool,
        source: DataSource | None = None,
    ) -> tuple[GateDecision, FetchedData | None]:
        context = ValidationContext(
            stage="data_fetch",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
        )
        common_decision, fetched_data = self._data_fetch_common(
            context, only_cached=only_cached, source=source
        )
        mode_decision = (
            self._data_fetch_backtest(context)
            if context.mode == "backtest"
            else self._data_fetch_walk_forward(context)
        )
        decision = self._compose_gate_decisions("data_fetch", common_decision, mode_decision)
        return decision, fetched_data

    def _data_fetch_common(
        self,
        context: ValidationContext,
        only_cached: bool,
        source: DataSource | None = None,
    ) -> tuple[GateDecision, FetchedData | None]:
        """Fetch market data for one job and translate failures into gate decisions."""
        job = context.job
        try:
            resolved_source = source or self._make_source(job.collection)
        except Exception as exc:
            return GateDecision(False, "skip_job", [str(exc)], "data_fetch"), None

        with time_block(
            self.logger,
            "data_fetch",
            collection=job.collection.name,
            symbol=job.symbol,
            timeframe=job.timeframe,
            source=job.source,
        ):
            try:
                df = resolved_source.fetch(job.symbol, job.timeframe, only_cached=only_cached)
            except Exception as exc:
                return GateDecision(False, "skip_job", [str(exc)], "data_fetch"), None
        return GateDecision(True, "continue", [], "data_fetch"), FetchedData(raw_df=df)

    @staticmethod
    def _data_fetch_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "data_fetch")

    @staticmethod
    def _data_fetch_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "skip_job",
            ["walk_forward_data_fetch_not_implemented"],
            "data_fetch",
        )

    def _data_validation(
        self, state: JobState, fetched_data: FetchedData
    ) -> tuple[GateDecision, ValidatedData | None]:
        context = ValidationContext(
            stage="data_validation",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
            fetched_data=fetched_data,
        )
        common_decision, validated_data = self._data_validation_common(context)
        mode_decision = (
            self._data_validation_backtest(context)
            if context.mode == "backtest"
            else self._data_validation_walk_forward(context)
        )
        decision = self._compose_gate_decisions("data_validation", common_decision, mode_decision)
        return decision, validated_data

    def _data_validation_common(
        self, context: ValidationContext
    ) -> tuple[GateDecision, ValidatedData | None]:
        # `validation.data_quality` drives the job-level data gate decisions.
        fetched_data = context.fetched_data
        if fetched_data is None:
            return GateDecision(False, "skip_job", ["missing_fetched_data"], "data_validation"), None
        if fetched_data.raw_df.empty:
            return GateDecision(False, "skip_job", ["empty_dataframe"], "data_validation"), None

        global_validation = getattr(self.cfg, "validation", None)
        collection_validation = getattr(context.job.collection, "validation", None)
        has_data_quality_policy = (
            getattr(global_validation, "data_quality", None) is not None
            or getattr(collection_validation, "data_quality", None) is not None
        )

        (
            reliability_on_fail,
            min_data_points_cfg,
            continuity_cfg,
            kurtosis_cfg,
            outlier_detection,
            calendar_kind,
            calendar_exchange,
            calendar_timezone,
        ) = (
            self._load_data_quality_policy(context.job.collection)
        )
        # `on_fail` is required by config whenever a data-quality policy exists,
        # so `reliability_on_fail is None` means data-quality policy is unset.
        if reliability_on_fail is None:
            # Even with policy disabled, we keep best-effort continuity diagnostics in result stats.
            continuity: dict[str, float | int] = {}
            default_calendar_kind, default_calendar_exchange, _ = self._resolve_calendar_policy(
                None, context.job.collection.source
            )
            try:
                continuity = self.compute_continuity_score(
                    fetched_data.raw_df,
                    context.job.timeframe,
                    calendar_kind=default_calendar_kind,
                    exchange_calendar=default_calendar_exchange,
                )
            except ValueError:
                # No policy is configured, so continuity diagnostics are best-effort only.
                continuity = {}
            validated_data = ValidatedData(
                raw_df=fetched_data.raw_df,
                continuity=continuity,
                reliability_on_fail="continue",
                reliability_reasons=[],
            )
            return GateDecision(True, "continue", [], "data_validation"), validated_data

        continuity: dict[str, float | int] = {}
        continuity_calendar_kind = calendar_kind
        continuity_calendar_exchange = calendar_exchange
        if continuity_calendar_kind is None:
            continuity_calendar_kind, continuity_calendar_exchange, _ = self._resolve_calendar_policy(
                None, context.job.collection.source
            )
        try:
            continuity = self.compute_continuity_score(
                fetched_data.raw_df,
                context.job.timeframe,
                calendar_kind=continuity_calendar_kind,
                exchange_calendar=continuity_calendar_exchange,
                calendar_timezone=calendar_timezone,
            )
        except ValueError as exc:
            # Preserve pre-validation behavior unless a data-quality policy is configured.
            if has_data_quality_policy:
                return GateDecision(False, "skip_job", [str(exc)], "data_validation"), None
            continuity = {}
        reliability_reasons = self._collect_reliability_reasons(
            raw_df=fetched_data.raw_df,
            continuity=continuity,
            min_data_points_cfg=min_data_points_cfg,
            continuity_cfg=continuity_cfg,
            kurtosis_cfg=kurtosis_cfg,
            outlier_detection=outlier_detection,
        )
        if not reliability_reasons:
            decision = GateDecision(True, "continue", [], "data_validation")
        elif reliability_on_fail == "skip_job":
            decision = GateDecision(False, "skip_job", reliability_reasons, "data_validation")
        elif reliability_on_fail == "skip_collection":
            decision = GateDecision(False, "skip_collection", reliability_reasons, "data_validation")
        else:
            decision = GateDecision(True, "skip_optimization", reliability_reasons, "data_validation")
        # Keep validated diagnostics available even when decision is skip_job.
        validated_data = ValidatedData(
            raw_df=fetched_data.raw_df,
            continuity=continuity,
            reliability_on_fail=reliability_on_fail,
            reliability_reasons=list(reliability_reasons),
        )
        return decision, validated_data

    def _load_data_quality_policy(
        self, collection: CollectionConfig
    ) -> tuple[
        str | None,
        int | None,
        ValidationContinuityConfig | None,
        float | None,
        ValidationOutlierDetectionConfig | None,
        str | None,
        str | None,
        str | None,
    ]:
        collection_validation = getattr(collection, "validation", None)
        resolved_dq = getattr(collection_validation, "data_quality", None) if collection_validation else None
        if resolved_dq is None:
            # Sentinel for "no data-quality policy configured for this collection".
            return None, None, None, None, None, None, None, None
        on_fail = resolved_dq.on_fail
        min_data_points_cfg = resolved_dq.min_data_points
        continuity_cfg = resolved_dq.continuity
        kurtosis_cfg = resolved_dq.kurtosis
        outlier_detection = resolved_dq.outlier_detection
        calendar_kind: str | None = None
        calendar_exchange: str | None = None
        calendar_timezone: str | None = None
        if continuity_cfg is not None:
            calendar_kind, calendar_exchange, calendar_timezone = self._resolve_calendar_policy(
                continuity_cfg.calendar, collection.source
            )
        return (
            on_fail,
            min_data_points_cfg,
            continuity_cfg,
            kurtosis_cfg,
            outlier_detection,
            calendar_kind,
            calendar_exchange,
            calendar_timezone,
        )

    def _resolve_calendar_policy(
        self,
        calendar_cfg: Any,
        source: str,
    ) -> tuple[str, str | None, str | None]:
        if calendar_cfg is None:
            calendar_kind = (
                "crypto_24_7"
                if source.strip().lower() in self._CRYPTO_SOURCE_NAMES
                else "weekday"
            )
            calendar_exchange = None
            calendar_timezone = None
            return calendar_kind, calendar_exchange, calendar_timezone
        calendar_kind = calendar_cfg.kind
        calendar_exchange = calendar_cfg.exchange
        calendar_timezone = calendar_cfg.timezone
        if calendar_kind == "auto":
            calendar_kind = (
                "crypto_24_7" if source.strip().lower() in self._CRYPTO_SOURCE_NAMES else "weekday"
            )
        return calendar_kind, calendar_exchange, calendar_timezone

    def _load_optimization_policy(self, collection: CollectionConfig) -> tuple[str, int, int] | None:
        # `validation.optimization` is optional per collection; when omitted no feasibility gate is applied.
        collection_validation = getattr(collection, "validation", None)
        policy = getattr(collection_validation, "optimization", None)
        if policy is None:
            return None
        return policy.on_fail, policy.min_bars, policy.dof_multiplier

    @staticmethod
    def _load_result_consistency_policy(collection: CollectionConfig) -> ResultConsistencyConfig | None:
        collection_validation = getattr(collection, "validation", None)
        return getattr(collection_validation, "result_consistency", None)

    @staticmethod
    def _continuity_threshold_reason(
        continuity: dict[str, float | int],
        continuity_cfg: ValidationContinuityConfig | None,
    ) -> str | None:
        if continuity_cfg is None or continuity_cfg.min_score is None:
            return None
        threshold = float(continuity_cfg.min_score)
        continuity_score = float(continuity.get("score", 0.0))
        if continuity_score < threshold:
            return (
                "min_continuity_score_not_met("
                f"required={threshold}, available={continuity_score})"
            )
        return None

    @staticmethod
    def _min_data_points_reason(
        raw_df: pd.DataFrame,
        min_data_points_cfg: int | None,
    ) -> str | None:
        if min_data_points_cfg is None:
            return None
        required = int(min_data_points_cfg)
        available = len(raw_df)
        if available < required:
            return f"min_data_points_not_met(required={required}, available={available})"
        return None

    @staticmethod
    def _missing_bar_pct_reason(
        continuity: dict[str, float | int],
        continuity_cfg: ValidationContinuityConfig | None,
    ) -> str | None:
        if continuity_cfg is None or continuity_cfg.max_missing_bar_pct is None:
            return None
        threshold = float(continuity_cfg.max_missing_bar_pct)
        expected_bars = int(continuity.get("expected_bars", 0))
        missing_bars = int(continuity.get("missing_bars", 0))
        missing_bar_pct = (
            0.0 if expected_bars <= 0 else (float(missing_bars) / float(expected_bars)) * 100.0
        )
        if missing_bar_pct > threshold:
            return (
                "max_missing_bar_pct_exceeded("
                f"max_allowed={threshold}, available={missing_bar_pct})"
            )
        return None

    @staticmethod
    def _resolve_close_column(raw_df: pd.DataFrame) -> str | None:
        if "Close" in raw_df.columns:
            return "Close"
        if "close" in raw_df.columns:
            return "close"
        return None

    @classmethod
    def _max_kurtosis_reason(
        cls,
        raw_df: pd.DataFrame,
        kurtosis_cfg: float | None,
    ) -> str | None:
        if kurtosis_cfg is None:
            return None
        close_col = cls._resolve_close_column(raw_df)
        if close_col is None:
            return None
        threshold = float(kurtosis_cfg)
        returns = raw_df[close_col].astype(float).pct_change().dropna()
        if returns.empty:
            return None
        sample_kurtosis = returns.kurt()
        if pd.notna(sample_kurtosis) and float(sample_kurtosis) > threshold:
            return (
                "max_kurtosis_exceeded("
                f"max_allowed={threshold}, available={float(sample_kurtosis)})"
            )
        return None

    @staticmethod
    def _collect_reliability_reasons(
        *,
        raw_df: pd.DataFrame,
        continuity: dict[str, float | int],
        min_data_points_cfg: int | None,
        continuity_cfg: ValidationContinuityConfig | None,
        kurtosis_cfg: float | None,
        outlier_detection: ValidationOutlierDetectionConfig | None,
    ) -> list[str]:
        reasons: list[str] = []
        reason_checks = (
            BacktestRunner._continuity_threshold_reason(continuity, continuity_cfg),
            BacktestRunner._min_data_points_reason(raw_df, min_data_points_cfg),
            BacktestRunner._missing_bar_pct_reason(continuity, continuity_cfg),
            BacktestRunner._max_kurtosis_reason(raw_df, kurtosis_cfg),
            BacktestRunner._outlier_pct_reason(
                raw_df=raw_df,
                outlier_detection=outlier_detection,
            ),
        )
        for reason in reason_checks:
            if reason is not None:
                reasons.append(reason)
        return reasons

    @classmethod
    def _outlier_pct_reason(
        cls,
        *,
        raw_df: pd.DataFrame,
        outlier_detection: ValidationOutlierDetectionConfig | None,
    ) -> str | None:
        if outlier_detection is None:
            return None
        close_col = cls._resolve_close_column(raw_df)
        if close_col is None:
            return None
        returns = raw_df[close_col].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return None
        method = outlier_detection.method
        threshold = outlier_detection.zscore_threshold
        outlier_mask, issue = cls._compute_outlier_mask(returns=returns, method=method, threshold=threshold)
        if issue is not None:
            return f"outlier_check_indeterminate(method={method}, reason={issue})"
        if outlier_mask is None:
            return None
        outlier_pct = float(outlier_mask.mean() * 100.0)
        allowed = outlier_detection.max_outlier_pct
        if allowed is None:
            return None
        if outlier_pct > allowed:
            return (
                "max_outlier_pct_exceeded("
                f"method={method}, threshold={threshold}, max_allowed={allowed}, available={outlier_pct})"
            )
        return None

    @staticmethod
    def _compute_outlier_mask(
        *,
        returns: pd.Series,
        method: str,
        threshold: float,
    ) -> tuple[np.ndarray | None, str | None]:
        values = returns.to_numpy(dtype=float)
        if values.size == 0:
            return None, "empty_returns"
        if method == "zscore":
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1))
            if not np.isfinite(std_val) or std_val <= 0:
                return None, "std_zero"
            return np.abs((values - mean_val) / std_val) > threshold, None
        if method != "modified_zscore":
            return None, f"unsupported_method:{method}"
        median_val = float(np.median(values))
        mad = float(np.median(np.abs(values - median_val)))
        if mad <= 0:
            return None, "mad_zero"
        modified_z = 0.6745 * (values - median_val) / mad
        return np.abs(modified_z) > threshold, None

    @staticmethod
    def _data_validation_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "data_validation")

    @staticmethod
    def _data_validation_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "skip_job",
            ["walk_forward_data_validation_not_implemented"],
            "data_validation",
        )

    def _execution_context_prepare(
        self, state: JobState, validated_data: ValidatedData
    ) -> tuple[GateDecision, ExecutionPreparedData | None]:
        context = ValidationContext(
            stage="data_preparation",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
            validated_data=validated_data,
        )
        common_decision, prepared = self._execution_context_prepare_common(context)
        mode_decision = (
            self._execution_context_prepare_backtest(context)
            if context.mode == "backtest"
            else self._execution_context_prepare_walk_forward(context)
        )
        decision = self._compose_gate_decisions("data_preparation", common_decision, mode_decision)
        return decision, prepared

    def _execution_context_prepare_common(
        self, context: ValidationContext
    ) -> tuple[GateDecision, ExecutionPreparedData | None]:
        # Prepare engine-ready frame and execution metadata for strategy stages.
        try:
            validated_data = context.validated_data
            if validated_data is None:
                return GateDecision(False, "skip_job", ["missing_validated_data"], "data_preparation"), None
            _, _, _, _, data_col_enum = self._ensure_pybroker()
            data_frame, dates = self._prepare_pybroker_frame(
                validated_data.raw_df, context.job.symbol, data_col_enum
            )
            fractional = self._fractional_enabled(context.job.collection, context.job.symbol)
            bars_per_year = self._bars_per_year(context.job.timeframe)
            close_col = self._resolve_close_column(validated_data.raw_df)
            if close_col is None:
                raise ValueError("missing close column")
            price = validated_data.raw_df[close_col].astype(float)
            fingerprint = (
                f"{len(validated_data.raw_df)}:{validated_data.raw_df.index[-1].isoformat()}:{float(price.iloc[-1])}"
            )
            fees, slippage = self._fees_slippage_for(context.job.collection)
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

    @staticmethod
    def _execution_context_prepare_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "data_preparation")

    @staticmethod
    def _execution_context_prepare_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "skip_job",
            ["walk_forward_data_preparation_not_implemented"],
            "data_preparation",
        )

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
        state: JobState,
        validated_data: ValidatedData,
        plan: StrategyPlan,
    ) -> GateDecision:
        context = ValidationContext(
            stage="strategy_optimization",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
            validated_data=validated_data,
            plan=plan,
        )
        common_decision = self._strategy_validate_plan_common(context)
        mode_decision = (
            self._strategy_validate_plan_backtest(context)
            if context.mode == "backtest"
            else self._strategy_validate_plan_walk_forward(context)
        )
        return self._compose_gate_decisions("strategy_optimization", common_decision, mode_decision)

    @staticmethod
    def _mark_plan_optimization_skipped(
        plan: StrategyPlan,
        skip_reasons: list[str],
        *,
        min_bars_required: int | None = None,
        bars_available: int | None = None,
    ) -> None:
        plan.skip_optimization = True
        plan.optimization_skip_reasons = list(skip_reasons)
        plan.optimization_skip_reason = "; ".join(skip_reasons)
        if not isinstance(plan.optimization_details, dict):
            plan.optimization_details = {}
        plan.optimization_details["skipped"] = True
        plan.optimization_details["reason"] = skip_reasons[0]
        plan.optimization_details["reasons"] = skip_reasons
        if min_bars_required is not None:
            plan.optimization_details["min_bars_required"] = min_bars_required
        if bars_available is not None:
            plan.optimization_details["bars_available"] = bars_available

    @staticmethod
    def _reset_plan_optimization_state(plan: StrategyPlan) -> None:
        plan.skip_optimization = False
        plan.optimization_skip_reasons = []
        plan.optimization_skip_reason = None
        plan.optimization_details = None

    @staticmethod
    def _insufficient_bars_for_optimization(
        plan: StrategyPlan,
        bars_available: int,
        min_bars_cfg: int,
        dof_multiplier: int,
    ) -> tuple[bool, int]:
        n_params = len(plan.search_space)
        min_bars_required = max(min_bars_cfg, dof_multiplier * n_params)
        insufficient = bool(plan.search_space) and bars_available < min_bars_required
        return insufficient, min_bars_required

    def _strategy_validate_plan_common(self, context: ValidationContext) -> GateDecision:
        # `validation.optimization` decides strategy-level action when search is infeasible.
        plan = context.plan
        validated_data = context.validated_data
        if plan is None or validated_data is None:
            return GateDecision(False, "skip_job", ["missing_strategy_plan_context"], "strategy_optimization")
        # If an upstream gate already decided to skip optimization for this plan,
        # keep that decision and avoid running strategy-level policy checks.
        if plan.optimization_skip_reasons:
            skip_reasons = list(plan.optimization_skip_reasons)
            self._mark_plan_optimization_skipped(plan, skip_reasons)
            return GateDecision(
                passed=True,
                action="baseline_only",
                reasons=skip_reasons,
                stage="strategy_optimization",
            )
        policy = self._load_optimization_policy(context.job.collection)
        if policy is None:
            self._reset_plan_optimization_state(plan)
            return GateDecision(
                passed=True,
                action="continue",
                reasons=[],
                stage="strategy_optimization",
            )

        optimization_on_fail, min_bars_cfg, dof_multiplier = policy
        bars_available = len(validated_data.raw_df)
        insufficient_bars, min_bars_required = self._insufficient_bars_for_optimization(
            plan,
            bars_available,
            min_bars_cfg,
            dof_multiplier,
        )
        if insufficient_bars:
            skip_reasons = ["insufficient_bars_for_optimization"]
            self._mark_plan_optimization_skipped(
                plan,
                skip_reasons,
                min_bars_required=min_bars_required,
                bars_available=bars_available,
            )
            return GateDecision(
                passed=(optimization_on_fail == "baseline_only"),
                action=optimization_on_fail,
                reasons=skip_reasons,
                stage="strategy_optimization",
            )

        self._reset_plan_optimization_state(plan)
        return GateDecision(
            passed=True,
            action="continue",
            reasons=[],
            stage="strategy_optimization",
        )

    @staticmethod
    def _strategy_validate_plan_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "strategy_optimization")

    @staticmethod
    def _strategy_validate_plan_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "skip_job",
            ["walk_forward_strategy_plan_validation_not_implemented"],
            "strategy_optimization",
        )

    def _strategy_evaluation(
        self,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
        params: dict[str, Any],
    ) -> float:
        """Evaluate one parameter set, preferring mode-aware cache when available."""
        full_params = {**plan.fixed_params, **params}
        call_params = full_params.copy()
        try:
            entries, exits = plan.strategy.generate_signals(validated_data.raw_df, call_params)
        except Exception as exc:
            self._failure_record(
                self._strategy_failure_payload(
                    state=state,
                    stage="generate_signals",
                    error=str(exc),
                    strategy=plan.strategy.name,
                    params=full_params,
                )
            )
            return float("nan")
        entries = entries.reindex(validated_data.raw_df.index, fill_value=False)
        exits = exits.reindex(validated_data.raw_df.index, fill_value=False)

        request = self._build_evaluation_request(plan, state, prepared, full_params)

        cached = self.evaluation_cache.get(
            collection=request.collection,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            params=request.params,
            metric_name=request.metric_name,
            data_fingerprint=request.data_fingerprint,
            fees=request.fees,
            slippage=request.slippage,
            evaluation_mode=self.mode_config.mode,
            mode_config_hash=self.mode_config_hash,
            validation_config_hash=state.validation_config_hash,
        )
        if cached is not None:
            return self._apply_cached_evaluation(plan, validated_data, request, cached, full_params)

        self.metrics["result_cache_misses"] += 1
        outcome = self._get_evaluator().evaluate(
            request,
            prepared.data_frame,
            prepared.dates,
            entries,
            exits,
            prepared.fractional,
        )
        return self._apply_fresh_evaluation(
            plan=plan,
            state=state,
            validated_data=validated_data,
            request=request,
            outcome=outcome,
            full_params=full_params,
        )

    def _build_evaluation_request(
        self,
        plan: StrategyPlan,
        state: JobState,
        prepared: ExecutionPreparedData,
        full_params: dict[str, Any],
    ) -> EvaluationRequest:
        result_consistency_policy = self._load_result_consistency_policy(state.job.collection)
        outlier_policy = (
            result_consistency_policy.outlier_dependency
            if result_consistency_policy is not None
            else None
        )
        execution_price_policy = (
            result_consistency_policy.execution_price_variance
            if result_consistency_policy is not None
            else None
        )
        return EvaluationRequest(
            collection=state.job.collection.name,
            symbol=state.job.symbol,
            timeframe=state.job.timeframe,
            source=state.job.source,
            strategy=plan.strategy.name,
            params=full_params,
            metric_name=self.cfg.metric,
            data_fingerprint=prepared.fingerprint,
            fees=prepared.fees,
            slippage=prepared.slippage,
            bars_per_year=prepared.bars_per_year,
            mode_config=self.mode_config,
            result_consistency_outlier_dependency_slices=(
                outlier_policy.slices if outlier_policy is not None else None
            ),
            result_consistency_outlier_dependency_profit_share_threshold=(
                outlier_policy.profit_share_threshold if outlier_policy is not None else None
            ),
            result_consistency_execution_price_tolerance_bps=(
                execution_price_policy.price_tolerance_bps
                if execution_price_policy is not None
                else None
            ),
        )

    def _apply_cached_evaluation(
        self,
        plan: StrategyPlan,
        validated_data: ValidatedData,
        request: EvaluationRequest,
        cached: dict[str, Any],
        full_params: dict[str, Any],
    ) -> float:
        self.metrics["result_cache_hits"] += 1
        plan.evaluations += 1
        metric_val = float(cached["metric_value"])
        stats = self._enrich_evaluation_stats(cached["stats"], plan, validated_data)
        self._cache_set(
            collection=request.collection,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            params=request.params,
            metric_name=request.metric_name,
            metric_value=metric_val,
            stats=stats,
            data_fingerprint=request.data_fingerprint,
            fees=request.fees,
            slippage=request.slippage,
            run_id=self.run_id,
            evaluation_mode=self.mode_config.mode,
            mode_config_hash=self.mode_config_hash,
        )
        self._update_best_result(plan, metric_val, full_params, stats)
        return metric_val

    def _apply_fresh_evaluation(
        self,
        *,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        request: EvaluationRequest,
        outcome: EvaluationOutcome,
        full_params: dict[str, Any],
    ) -> float:
        raw_stats = dict(outcome.stats)
        stats = self._enrich_evaluation_stats(raw_stats, plan, validated_data)
        plan.evaluations += 1
        self._track_fresh_evaluation_metrics(outcome)
        metric_val = float(outcome.metric_value)
        if outcome.valid or outcome.metric_computed:
            self._evaluation_cache_set(
                collection=request.collection,
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                params=request.params,
                metric_name=request.metric_name,
                metric_value=metric_val,
                stats=raw_stats,
                data_fingerprint=request.data_fingerprint,
                fees=request.fees,
                slippage=request.slippage,
                evaluation_mode=self.mode_config.mode,
                mode_config_hash=self.mode_config_hash,
                validation_config_hash=state.validation_config_hash,
            )
            self._cache_set(
                collection=request.collection,
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                params=request.params,
                metric_name=request.metric_name,
                metric_value=metric_val,
                stats=stats,
                data_fingerprint=request.data_fingerprint,
                fees=request.fees,
                slippage=request.slippage,
                run_id=self.run_id,
                evaluation_mode=self.mode_config.mode,
                mode_config_hash=self.mode_config_hash,
            )
        if not outcome.valid:
            return float("-inf")
        self._update_best_result(plan, metric_val, full_params, stats)
        return metric_val

    def _track_fresh_evaluation_metrics(self, outcome: EvaluationOutcome) -> None:
        # Evaluator owns execution-state flags; runner only aggregates.
        if outcome.simulation_executed:
            self.metrics["fresh_simulation_runs"] += 1
        if outcome.metric_computed:
            self.metrics["fresh_metric_evals"] += 1
            # Backward-compatible alias for existing dashboards/tests.
            self.metrics["param_evals"] = self.metrics["fresh_metric_evals"]

    @staticmethod
    def _update_best_result(
        plan: StrategyPlan,
        metric_val: float,
        full_params: dict[str, Any],
        stats: dict[str, Any],
    ) -> None:
        if metric_val > plan.best_val:
            plan.best_val = metric_val
            plan.best_params = full_params.copy()
            plan.best_stats = stats

    def _strategy_run(
        self,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
    ) -> StrategyEvalOutcome | None:
        """Run optimization/baseline evaluation and return the best strategy outcome."""
        try:
            if not plan.search_space or plan.skip_optimization:
                self._strategy_evaluation(plan, state, validated_data, prepared, {})
            else:
                search_method = self._resolve_search_method(plan.search_method)
                if search_method == "optuna":
                    self._run_optuna_strategy_search(plan, state, validated_data, prepared)
                else:
                    for params in self._grid(plan.search_space):
                        self._strategy_evaluation(plan, state, validated_data, prepared, params)
            return self._build_strategy_eval_outcome(plan, state)
        except Exception as exc:
            self._failure_record(
                self._strategy_failure_payload(
                    state=state,
                    stage="strategy_optimization",
                    error=str(exc),
                    strategy=plan.strategy.name,
                )
            )
            return None

    @staticmethod
    def _resolve_search_method(search_method: str) -> str:
        if search_method != "optuna":
            return search_method
        try:
            import optuna  # noqa: F401
        except ImportError:
            return "grid"
        return "optuna"

    @staticmethod
    def _total_search_combinations(search_space: dict[str, list[Any]]) -> int:
        total_combos = 1
        for options in search_space.values():
            total_combos *= max(1, len(options))
        return max(1, total_combos)

    def _run_optuna_strategy_search(
        self,
        plan: StrategyPlan,
        state: JobState,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
    ) -> None:
        import optuna

        space_items = list(plan.search_space.items())

        def objective(trial):
            var_params = {
                name: trial.suggest_categorical(name, options) for name, options in space_items
            }
            result = self._strategy_evaluation(plan, state, validated_data, prepared, var_params)
            return result if np.isfinite(result) else float("-inf")

        n_trials = min(plan.trials_target, self._total_search_combinations(plan.search_space))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

    @staticmethod
    def _build_strategy_eval_outcome(plan: StrategyPlan, state: JobState) -> StrategyEvalOutcome:
        has_valid_candidate = (
            plan.best_params is not None and isinstance(plan.best_stats, dict) and bool(plan.best_stats)
        )
        return StrategyEvalOutcome(
            best_val=float(plan.best_val),
            best_params=plan.best_params,
            best_stats=plan.best_stats,
            has_valid_candidate=has_valid_candidate,
            evaluations=plan.evaluations,
            skipped_reason=plan.optimization_skip_reason,
            strategy=plan.strategy.name,
            job=state.job,
        )

    def _strategy_validate_results(self, state: JobState, outcome: StrategyEvalOutcome) -> GateDecision:
        context = ValidationContext(
            stage="strategy_validation",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
            outcome=outcome,
        )
        common_decision = self._strategy_validate_results_common(context)
        mode_decision = (
            self._strategy_validate_results_backtest(context)
            if context.mode == "backtest"
            else self._strategy_validate_results_walk_forward(context)
        )
        return self._compose_gate_decisions("strategy_validation", common_decision, mode_decision)

    def _outlier_dependency_reason(
        self,
        stats: dict[str, Any],
        policy: ResultConsistencyOutlierDependencyConfig,
    ) -> str | None:
        trade_meta = stats.get("trade_meta")
        if not isinstance(trade_meta, dict):
            return None
        outlier_meta = trade_meta.get("outlier_dependency")
        if not isinstance(outlier_meta, dict):
            return None
        if not bool(outlier_meta.get("is_complete", False)):
            return None
        dominant_trade_share_raw = outlier_meta.get("dominant_trade_share_for_profit_share")
        if dominant_trade_share_raw is None:
            return None
        dominant_trade_share = float(dominant_trade_share_raw)
        if dominant_trade_share >= policy.trade_share_threshold:
            return None
        dominant_trade_count = outlier_meta.get("dominant_trade_count_for_profit_share")
        slice_concentration = outlier_meta.get("max_slice_profit_share")

        reason = (
            "outlier_dependency_exceeded("
            f"dominant_trade_share={dominant_trade_share:.4f}, "
            f"dominant_trade_count={dominant_trade_count}, "
            f"profit_share_threshold={policy.profit_share_threshold}, "
            f"trade_share_threshold={policy.trade_share_threshold}, "
            f"slices={policy.slices}"
        )
        if isinstance(slice_concentration, (float, int)):
            reason += f", max_slice_profit_share={float(slice_concentration):.4f}"
        reason += ")"
        return reason

    @staticmethod
    def _execution_price_variance_reason(
        stats: dict[str, Any],
        policy: ResultConsistencyExecutionPriceVarianceConfig,
    ) -> str | None:
        trade_meta = stats.get("trade_meta")
        if not isinstance(trade_meta, dict):
            return None
        execution_meta = trade_meta.get("execution_price_variance")
        if not isinstance(execution_meta, dict):
            return None
        # Missing metadata is explicitly non-blocking for this gate.
        if not bool(execution_meta.get("is_complete", False)):
            return None
        violations_raw = execution_meta.get("violations")
        if violations_raw is None:
            return None
        violations = int(violations_raw)
        if violations <= 0:
            return None
        checked_fills = execution_meta.get("checked_fills")
        violation_ratio = execution_meta.get("violation_ratio")
        return (
            "execution_price_variance_exceeded("
            f"violations={violations}, "
            f"checked_fills={checked_fills}, "
            f"violation_ratio={violation_ratio}, "
            f"price_tolerance_bps={policy.price_tolerance_bps}"
            ")"
        )

    def _strategy_validate_results_common(self, context: ValidationContext) -> GateDecision:
        # Keep this gate lightweight for now; stricter schema checks can be added later.
        outcome = context.outcome
        if outcome is None:
            return GateDecision(False, "reject_result", ["missing_strategy_outcome"], "strategy_validation")
        if not outcome.has_valid_candidate:
            return GateDecision(False, "reject_result", ["no_valid_candidate"], "strategy_validation")

        reasons = self._collect_strategy_validation_reasons(context, outcome)
        if reasons:
            decision = GateDecision(False, "reject_result", reasons, "strategy_validation")
        else:
            decision = GateDecision(True, "continue", [], "strategy_validation")
        return decision

    def _collect_strategy_validation_reasons(
        self,
        context: ValidationContext,
        outcome: StrategyEvalOutcome,
    ) -> list[str]:
        reasons: list[str] = []
        if not np.isfinite(outcome.best_val):
            reasons.append("best_metric_not_finite")
        policy = self._load_result_consistency_policy(context.job.collection)
        if policy is None or not isinstance(outcome.best_stats, dict):
            return reasons
        outlier_policy = policy.outlier_dependency
        if outlier_policy is not None:
            reason = self._outlier_dependency_reason(outcome.best_stats, outlier_policy)
            if reason is not None:
                reasons.append(reason)
        execution_price_policy = policy.execution_price_variance
        if execution_price_policy is not None:
            reason = self._execution_price_variance_reason(
                outcome.best_stats,
                execution_price_policy,
            )
            if reason is not None:
                reasons.append(reason)
        return reasons

    @staticmethod
    def _strategy_validate_results_backtest(_context: ValidationContext) -> GateDecision:
        return GateDecision(True, "continue", [], "strategy_validation")

    @staticmethod
    def _strategy_validate_results_walk_forward(_context: ValidationContext) -> GateDecision:
        return GateDecision(
            False,
            "reject_result",
            ["walk_forward_result_validation_not_implemented"],
            "strategy_validation",
        )

    def _build_result_record(self, best: BestResult, job: JobContext, prepared: ExecutionPreparedData) -> ResultRecord:
        return ResultRecord(
            run_id=self.run_id,
            evaluation_mode=self.mode_config.mode,
            collection=best.collection,
            symbol=best.symbol,
            timeframe=best.timeframe,
            source=job.source,
            strategy=best.strategy,
            params=dict(best.params),
            metric_name=best.metric_name,
            metric_value=float(best.metric_value),
            stats=dict(best.stats),
            data_fingerprint=prepared.fingerprint,
            fees=prepared.fees,
            slippage=prepared.slippage,
            mode_config_hash=self.mode_config_hash,
        )

    def list_result_rows_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.result_store.list_by_run(run_id)
        return normalized_rows_to_legacy_rows(rows)

    def run_all(self, only_cached: bool = False) -> list[BestResult]:
        """Execute the full backtest pipeline across all jobs and strategies."""
        best_results: list[BestResult] = []
        # Initialize per-run counters and transient state.
        self.metrics = {
            "result_cache_hits": 0,
            "result_cache_misses": 0,
            "fresh_simulation_runs": 0,
            "fresh_metric_evals": 0,
            "param_evals": 0,
            "symbols_tested": 0,
            "strategies_used": set(),
        }
        self.failures = []
        self._cache_write_failures = 0
        self._result_store_write_failures = 0
        self._evaluation_cache_write_failures = 0
        self._evaluator = None
        self._strategy_overrides = (
            {s.name: s.params for s in self.cfg.strategies} if self.cfg.strategies else {}
        )
        validation_metadata = self._build_validation_metadata()
        self.validation_metadata = validation_metadata
        self.active_validation_gates = list(validation_metadata.get("active_gates", []))
        self.inactive_validation_gates = list(validation_metadata.get("inactive_gates", []))
        self._result_store_upsert_run_metadata(
            validation_profile=validation_metadata.get("profile", {}),
            active_gates=self.active_validation_gates,
            inactive_gates=self.inactive_validation_gates,
        )

        jobs = self._create_job_list()
        # Cache collection gate decisions so each collection is validated once per run.
        validated_collections: dict[int, GateDecision] = {}
        validated_collection_sources: dict[int, DataSource] = {}
        blocked_collections: set[int] = set()

        for job in jobs:
            state = JobState(job=job)
            collection_key = self._collection_gate_key(job.collection)
            if collection_key in blocked_collections:
                continue
            collection_decision = validated_collections.get(collection_key)
            if collection_decision is None:
                collection_decision, source = self._collection_validation(state)
                collection_decision = self._handle_gate_decision(
                    state,
                    collection_decision,
                    blocked_collections=blocked_collections,
                )
                validated_collections[collection_key] = collection_decision
                if source is not None:
                    validated_collection_sources[collection_key] = source
            else:
                # Apply cached gate state without re-emitting logs/failure side effects.
                self._apply_gate_to_state(state, collection_decision)
            if not collection_decision.passed:
                continue

            data_fetch_decision, fetched_data = self._data_fetch(
                state,
                only_cached=only_cached,
                source=validated_collection_sources.get(collection_key),
            )
            data_fetch_decision = self._handle_gate_decision(
                state,
                data_fetch_decision,
                blocked_collections=blocked_collections,
            )
            if not data_fetch_decision.passed or fetched_data is None:
                continue

            data_decision, validated_data = self._data_validation(state, fetched_data)
            data_decision = self._handle_gate_decision(
                state,
                data_decision,
                blocked_collections=blocked_collections,
            )
            if not data_decision.passed or validated_data is None:
                continue

            prep_decision, prepared = self._execution_context_prepare(state, validated_data)
            prep_decision = self._handle_gate_decision(
                state,
                prep_decision,
                blocked_collections=blocked_collections,
            )
            if not prep_decision.passed or prepared is None:
                continue

            # This hash must be computed at runtime from the effective collection-level
            # validation profile (global + per-collection overrides) because it keys
            # evaluation-cache correctness per job, not just raw loaded config.
            state.validation_config_hash = self._hash_validation_profile(
                self._build_job_validation_profile(state.job.collection)
            )
            for strat_name in self.external_index.keys():
                self.metrics["symbols_tested"] += 1
                # Strategy stage: create plan -> validate plan -> run -> validate results.
                plan = self._strategy_create_plan(state, strat_name)
                self._apply_policy_constraints_to_plan(state, validated_data, plan)
                plan_decision = self._strategy_validate_plan(state, validated_data, plan)
                plan_decision = self._handle_gate_decision(
                    state,
                    plan_decision,
                    context_extra={
                        "strategy": plan.strategy.name,
                        "search_method": plan.search_method,
                    },
                    blocked_collections=blocked_collections,
                )
                if not plan_decision.passed:
                    if plan_decision.action in {"skip_job", "skip_collection"}:
                        break
                    continue
                self.metrics["strategies_used"].add(plan.strategy.name)
                outcome = self._strategy_run(plan, state, validated_data, prepared)
                if outcome is None:
                    continue
                validation_decision = self._handle_gate_decision(
                    state,
                    self._strategy_validate_results(state, outcome),
                    context_extra={"strategy": outcome.strategy},
                    blocked_collections=blocked_collections,
                )
                if not validation_decision.passed:
                    if validation_decision.action in {"skip_job", "skip_collection"}:
                        break
                    continue

                best = BestResult(
                    collection=state.job.collection.name,
                    symbol=state.job.symbol,
                    timeframe=state.job.timeframe,
                    strategy=plan.strategy.name,
                    params=outcome.best_params or {},
                    metric_name=self.cfg.metric,
                    metric_value=float(outcome.best_val),
                    stats=outcome.best_stats or {},
                )
                best_results.append(best)
                self._result_store_insert(self._build_result_record(best, state.job, prepared))

        if isinstance(self.metrics.get("strategies_used"), set):
            self.metrics["strategies_count"] = len(self.metrics["strategies_used"])  # type: ignore
            self.metrics.pop("strategies_used", None)
        return best_results
