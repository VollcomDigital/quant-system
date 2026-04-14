from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
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
    ResultConsistencyConfig,
    ResultConsistencyExecutionPriceVarianceConfig,
    ResultConsistencyTransactionCostBreakevenConfig,
    ResultConsistencyTransactionCostRobustnessConfig,
    ResultConsistencyOutlierDependencyConfig,
    STATIONARITY_MIN_POINTS_DEFAULT,
    ValidationContinuityConfig,
    ValidationDataQualityConfig,
    ValidationLookaheadShuffleTestConfig,
    ValidationOHLCIntegrityConfig,
    ValidationOutlierDetectionConfig,
    ValidationStationarityConfig,
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
    EvaluationOutcome,
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
    reliability_on_fail: str | None
    reliability_reasons: list[str]
    canonicalization: dict[str, int]


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


@dataclass
class LookaheadShuffleRunContext:
    context: ValidationContext
    plan: StrategyPlan
    policy: ValidationLookaheadShuffleTestConfig
    raw_df: pd.DataFrame
    close_col: str
    data_col_enum: Any
    fractional: bool
    bars_per_year: int
    fees: float
    slippage: float
    rng: np.random.Generator
    effective_params: dict[str, Any]
    max_failed_permutations: int | None
    derived_seed: int


@dataclass
class TransactionCostRobustnessRunContext:
    context: ValidationContext
    plan: StrategyPlan
    policy: ResultConsistencyTransactionCostRobustnessConfig
    prepared: ExecutionPreparedData
    full_params: dict[str, Any]
    baseline_metric: float | None
    baseline_profit: float | None
    aligned_signals: tuple[pd.Series, pd.Series] | None = None


class BacktestRunner:
    _CRYPTO_SOURCE_NAMES = {"binance", "bybit", "ccxt", "coinbase", "kraken", "okx", "kucoin"}
    _TRANSACTION_COST_ROBUSTNESS_DROP_EPSILON = 1e-9
    _VALIDATION_GATE_IDS = (
        "data_quality.min_required_bars",
        "data_quality.min_data_points",
        "data_quality.continuity.min_score",
        "data_quality.continuity.max_missing_bar_pct",
        "data_quality.ohlc_integrity",
        "data_quality.kurtosis",
        "data_quality.outlier_detection",
        "data_quality.stationarity",
        "optimization.feasibility",
        "result_consistency.min_metric",
        "result_consistency.min_trades",
        "result_consistency.outlier_dependency",
        "result_consistency.execution_price_variance",
        "result_consistency.lookahead_shuffle_test",
        "result_consistency.transaction_cost_robustness",
    )

    _STATIONARITY_TRUE = "true"
    _STATIONARITY_FALSE = "false"

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
        self._runtime_signal_error_counts: dict[tuple[str, str, str, str], int] = {}
        self._runtime_signal_error_capped: set[tuple[str, str, str, str]] = set()
        self._strategy_fingerprint_cache: dict[type[BaseStrategy], str] = {}
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
    def _serialize_calendar_profile(calendar: Any) -> dict[str, Any] | None:
        if calendar is None:
            return None
        return {
            "kind": getattr(calendar, "kind", None),
            "exchange": getattr(calendar, "exchange", None),
            "timezone": getattr(calendar, "timezone", None),
        }

    @staticmethod
    def _serialize_continuity_profile(continuity: Any) -> dict[str, Any] | None:
        if continuity is None:
            return None
        return {
            "min_score": getattr(continuity, "min_score", None),
            "max_missing_bar_pct": getattr(continuity, "max_missing_bar_pct", None),
        }

    @staticmethod
    def _serialize_outlier_detection_profile(outlier_detection: Any) -> dict[str, Any] | None:
        if outlier_detection is None:
            return None
        return {
            "max_outlier_pct": getattr(outlier_detection, "max_outlier_pct", None),
            "method": getattr(outlier_detection, "method", None),
            "zscore_threshold": getattr(outlier_detection, "zscore_threshold", None),
        }

    @staticmethod
    def _serialize_stationarity_regime_shift_profile(regime_shift: Any) -> dict[str, Any] | None:
        if regime_shift is None:
            return None
        return {
            "window": getattr(regime_shift, "window", None),
            "mean_shift_max": getattr(regime_shift, "mean_shift_max", None),
            "vol_ratio_max": getattr(regime_shift, "vol_ratio_max", None),
        }

    @classmethod
    def _serialize_stationarity_profile(cls, stationarity: Any) -> dict[str, Any] | None:
        if stationarity is None:
            return None
        return {
            "adf_pvalue_max": getattr(stationarity, "adf_pvalue_max", None),
            "kpss_pvalue_min": getattr(stationarity, "kpss_pvalue_min", None),
            "min_points": getattr(stationarity, "min_points", None),
            "regime_shift": cls._serialize_stationarity_regime_shift_profile(
                getattr(stationarity, "regime_shift", None)
            ),
        }

    @staticmethod
    def _serialize_lookahead_shuffle_test_profile(
        lookahead_shuffle_test: Any,
    ) -> dict[str, Any] | None:
        if lookahead_shuffle_test is None:
            return None
        return {
            "permutations": getattr(lookahead_shuffle_test, "permutations", None),
            "pvalue_max": getattr(lookahead_shuffle_test, "pvalue_max", None),
            "seed": getattr(lookahead_shuffle_test, "seed", None),
            "max_failed_permutations": getattr(
                lookahead_shuffle_test, "max_failed_permutations", None
            ),
        }

    @staticmethod
    def _serialize_transaction_cost_breakeven_profile(
        breakeven: Any,
    ) -> dict[str, Any] | None:
        if breakeven is None:
            return None
        return {
            "enabled": getattr(breakeven, "enabled", None),
            "min_multiplier": getattr(breakeven, "min_multiplier", None),
            "max_multiplier": getattr(breakeven, "max_multiplier", None),
            "max_iterations": getattr(breakeven, "max_iterations", None),
            "tolerance": getattr(breakeven, "tolerance", None),
        }

    @staticmethod
    def _serialize_transaction_cost_robustness_profile(
        transaction_cost_robustness: Any,
    ) -> dict[str, Any] | None:
        if transaction_cost_robustness is None:
            return None
        return {
            "mode": getattr(transaction_cost_robustness, "mode", None),
            "stress_multipliers": getattr(transaction_cost_robustness, "stress_multipliers", None),
            "max_metric_drop_pct": getattr(transaction_cost_robustness, "max_metric_drop_pct", None),
            "breakeven": BacktestRunner._serialize_transaction_cost_breakeven_profile(
                getattr(transaction_cost_robustness, "breakeven", None)
            ),
        }

    @staticmethod
    def _serialize_data_quality_profile(data_quality: Any) -> dict[str, Any] | None:
        if data_quality is None:
            return None
        return {
            "on_fail": getattr(data_quality, "on_fail", None),
            "min_data_points": getattr(data_quality, "min_data_points", None),
            "is_verified": getattr(data_quality, "is_verified", None),
            "calendar": BacktestRunner._serialize_calendar_profile(
                getattr(data_quality, "calendar", None)
            ),
            "continuity": BacktestRunner._serialize_continuity_profile(
                getattr(data_quality, "continuity", None)
            ),
            "ohlc_integrity": {
                "max_invalid_bar_pct": getattr(
                    getattr(data_quality, "ohlc_integrity", None),
                    "max_invalid_bar_pct",
                    None,
                ),
                "allow_negative_price": getattr(
                    getattr(data_quality, "ohlc_integrity", None),
                    "allow_negative_price",
                    None,
                ),
                "allow_negative_volume": getattr(
                    getattr(data_quality, "ohlc_integrity", None),
                    "allow_negative_volume",
                    None,
                ),
            }
            if getattr(data_quality, "ohlc_integrity", None) is not None
            else None,
            "kurtosis": getattr(data_quality, "kurtosis", None),
            "outlier_detection": BacktestRunner._serialize_outlier_detection_profile(
                getattr(data_quality, "outlier_detection", None)
            ),
            "stationarity": BacktestRunner._serialize_stationarity_profile(
                getattr(data_quality, "stationarity", None)
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
            "runtime_error_max_per_tuple": getattr(optimization, "runtime_error_max_per_tuple", None),
        }

    @staticmethod
    def _serialize_result_consistency_profile(result_consistency: Any) -> dict[str, Any] | None:
        if result_consistency is None:
            return None
        outlier_dependency = getattr(result_consistency, "outlier_dependency", None)
        execution_price_variance = getattr(result_consistency, "execution_price_variance", None)
        return {
            "min_metric": getattr(result_consistency, "min_metric", None),
            "min_trades": getattr(result_consistency, "min_trades", None),
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
            "lookahead_shuffle_test": BacktestRunner._serialize_lookahead_shuffle_test_profile(
                getattr(result_consistency, "lookahead_shuffle_test", None)
            ),
            "transaction_cost_robustness": BacktestRunner._serialize_transaction_cost_robustness_profile(
                getattr(result_consistency, "transaction_cost_robustness", None)
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
        if getattr(data_quality, "ohlc_integrity", None) is not None:
            active.add("data_quality.ohlc_integrity")
        if getattr(data_quality, "kurtosis", None) is not None:
            active.add("data_quality.kurtosis")
        if getattr(data_quality, "outlier_detection", None) is not None:
            active.add("data_quality.outlier_detection")
        if getattr(data_quality, "stationarity", None) is not None:
            active.add("data_quality.stationarity")
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
        if getattr(result_consistency, "min_metric", None) is not None:
            active.add("result_consistency.min_metric")
        if getattr(result_consistency, "min_trades", None) is not None:
            active.add("result_consistency.min_trades")
        if getattr(result_consistency, "outlier_dependency", None) is not None:
            active.add("result_consistency.outlier_dependency")
        if getattr(result_consistency, "execution_price_variance", None) is not None:
            active.add("result_consistency.execution_price_variance")
        if getattr(result_consistency, "lookahead_shuffle_test", None) is not None:
            active.add("result_consistency.lookahead_shuffle_test")
        if getattr(result_consistency, "transaction_cost_robustness", None) is not None:
            active.add("result_consistency.transaction_cost_robustness")
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
        duplicate_bars: int = 0,
    ) -> dict[str, float | int]:
        idx = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values()
        if idx.has_duplicates:
            raise ValueError("duplicate_index_for_continuity: canonicalize_before_scoring")
        duplicate_bars = max(0, int(duplicate_bars))
        unique_bars = int(len(idx))
        actual_bars = unique_bars + duplicate_bars

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
        reliability["canonicalization"] = dict(validated_data.canonicalization)
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

    @staticmethod
    def _should_log_gate_decision(decision: GateDecision) -> bool:
        return (not decision.passed) or decision.action != "continue"

    @staticmethod
    def _should_record_gate_failure(record_failure: bool, decision: GateDecision) -> bool:
        return record_failure and decision.action in {"skip_job", "skip_collection", "reject_result"}

    def _gate_context(self, state: JobState, context_extra: dict[str, Any] | None) -> dict[str, Any]:
        context = self._job_log_context(state.job)
        if context_extra:
            context |= context_extra
        return context

    def _record_gate_failure(
        self,
        state: JobState,
        decision: GateDecision,
        context_extra: dict[str, Any] | None,
    ) -> None:
        failure: dict[str, Any] = {
            **self._job_log_context(state.job),
            "stage": decision.stage,
            "error": "; ".join(decision.reasons) if decision.reasons else "gate_failed",
        }
        if context_extra and "strategy" in context_extra and "strategy" not in failure:
            failure["strategy"] = context_extra["strategy"]
        self._failure_record(failure)

    def _update_blocked_collections(
        self,
        state: JobState,
        decision: GateDecision,
        blocked_collections: set[int] | None,
    ) -> None:
        if decision.action == "skip_collection" and blocked_collections is not None:
            blocked_collections.add(self._collection_gate_key(state.job.collection))

    def _handle_gate_decision(
        self,
        state: JobState,
        decision: GateDecision,
        context_extra: dict[str, Any] | None = None,
        record_failure: bool = True,
        blocked_collections: set[int] | None = None,
    ) -> GateDecision:
        """Apply, log, and optionally record a non-continue gate decision."""
        context = self._gate_context(state, context_extra)
        self._apply_gate_to_state(state, decision)
        self._update_blocked_collections(state, decision, blocked_collections)
        if self._should_log_gate_decision(decision):
            self._gate_log(decision.stage, decision, context)
            if self._should_record_gate_failure(record_failure, decision):
                self._record_gate_failure(state, decision, context_extra)
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
        plan: StrategyPlan,
    ) -> None:
        if not state.policy_skip_optimization or not plan.search_space:
            return
        self._plan_add_skip_reason(plan, "reliability_threshold_skip_optimization")

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
        (
            reliability_on_fail,
            min_data_points_cfg,
            continuity_cfg,
            kurtosis_cfg,
            ohlc_integrity_cfg,
            outlier_detection,
            stationarity_cfg,
            is_verified,
            calendar_kind,
            calendar_exchange,
            calendar_timezone,
        ) = self._load_data_quality_policy(context.job.collection)
        try:
            canonical_raw_df, canonicalization_meta = self._canonicalize_validation_frame(
                fetched_data.raw_df,
                calendar_timezone=calendar_timezone,
            )
        except ValueError as exc:
            return GateDecision(False, "skip_job", [str(exc)], "data_validation"), None
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
                    canonical_raw_df,
                    context.job.timeframe,
                    calendar_kind=default_calendar_kind,
                    exchange_calendar=default_calendar_exchange,
                    duplicate_bars=canonicalization_meta.get("duplicate_bars_removed", 0),
                )
            except ValueError:
                # No policy is configured, so continuity diagnostics are best-effort only.
                continuity = {}
            validated_data = ValidatedData(
                raw_df=canonical_raw_df,
                continuity=continuity,
                reliability_on_fail=None,
                reliability_reasons=[],
                canonicalization=canonicalization_meta,
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
                canonical_raw_df,
                context.job.timeframe,
                calendar_kind=continuity_calendar_kind,
                exchange_calendar=continuity_calendar_exchange,
                duplicate_bars=canonicalization_meta.get("duplicate_bars_removed", 0),
            )
        except ValueError as exc:
            return GateDecision(False, "skip_job", [str(exc)], "data_validation"), None
        reliability_reasons = self._collect_reliability_reasons(
            raw_df=canonical_raw_df,
            continuity=continuity,
            min_data_points_cfg=min_data_points_cfg,
            continuity_cfg=continuity_cfg,
            kurtosis_cfg=kurtosis_cfg,
            ohlc_integrity_cfg=ohlc_integrity_cfg,
            outlier_detection=outlier_detection,
            stationarity_cfg=stationarity_cfg,
            is_verified=is_verified,
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
            raw_df=canonical_raw_df,
            continuity=continuity,
            reliability_on_fail=reliability_on_fail,
            reliability_reasons=list(reliability_reasons),
            canonicalization=canonicalization_meta,
        )
        return decision, validated_data

    def _load_data_quality_policy(
        self, collection: CollectionConfig
    ) -> tuple[
        str | None,
        int | None,
        ValidationContinuityConfig | None,
        float | None,
        ValidationOHLCIntegrityConfig | None,
        ValidationOutlierDetectionConfig | None,
        ValidationStationarityConfig | None,
        bool | None,
        str | None,
        str | None,
        str | None,
    ]:
        collection_validation = getattr(collection, "validation", None)
        resolved_dq: ValidationDataQualityConfig | None = (
            getattr(collection_validation, "data_quality", None) if collection_validation else None
        )
        if resolved_dq is None:
            # Sentinel for "no data-quality policy configured for this collection".
            return None, None, None, None, None, None, None, None, None, None, None
        on_fail = resolved_dq.on_fail
        min_data_points_cfg = resolved_dq.min_data_points
        continuity_cfg = resolved_dq.continuity
        kurtosis_cfg = resolved_dq.kurtosis
        ohlc_integrity_cfg = getattr(resolved_dq, "ohlc_integrity", None)
        outlier_detection = resolved_dq.outlier_detection
        stationarity_cfg = getattr(resolved_dq, "stationarity", None)
        is_verified = getattr(resolved_dq, "is_verified", None)
        calendar_kind: str | None = None
        calendar_exchange: str | None = None
        calendar_timezone: str | None = None
        calendar_cfg = getattr(resolved_dq, "calendar", None)
        if calendar_cfg is not None:
            calendar_kind, calendar_exchange, calendar_timezone = self._resolve_calendar_policy(
                calendar_cfg, collection.source
            )
        return (
            on_fail,
            min_data_points_cfg,
            continuity_cfg,
            kurtosis_cfg,
            ohlc_integrity_cfg,
            outlier_detection,
            stationarity_cfg,
            is_verified,
            calendar_kind,
            calendar_exchange,
            calendar_timezone,
        )

    def _load_lookahead_shuffle_test_policy(
        self, collection: CollectionConfig
    ) -> ValidationLookaheadShuffleTestConfig | None:
        collection_validation = getattr(collection, "validation", None)
        resolved_rc: ResultConsistencyConfig | None = (
            getattr(collection_validation, "result_consistency", None) if collection_validation else None
        )
        if resolved_rc is not None:
            return getattr(resolved_rc, "lookahead_shuffle_test", None)
        return None

    def _load_transaction_cost_robustness_policy(
        self, collection: CollectionConfig
    ) -> ResultConsistencyTransactionCostRobustnessConfig | None:
        collection_validation = getattr(collection, "validation", None)
        resolved_rc: ResultConsistencyConfig | None = (
            getattr(collection_validation, "result_consistency", None) if collection_validation else None
        )
        if resolved_rc is not None:
            return getattr(resolved_rc, "transaction_cost_robustness", None)
        return None

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

    def _load_optimization_policy(self, collection: CollectionConfig) -> tuple[str, int, int, int] | None:
        # `validation.optimization` is optional per collection; when omitted no feasibility gate is applied.
        collection_validation = getattr(collection, "validation", None)
        policy = getattr(collection_validation, "optimization", None)
        if policy is None:
            return None
        runtime_error_max_per_tuple_raw = getattr(policy, "runtime_error_max_per_tuple", None)
        runtime_error_max_per_tuple = (
            1 if runtime_error_max_per_tuple_raw is None else int(runtime_error_max_per_tuple_raw)
        )
        return policy.on_fail, policy.min_bars, policy.dof_multiplier, runtime_error_max_per_tuple

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
    def _collection_verification_reason(is_verified: bool | None) -> str | None:
        if is_verified is False:
            return "collection_not_verified"
        return None

    @staticmethod
    def _resolve_close_column(raw_df: pd.DataFrame) -> str | None:
        if "Close" in raw_df.columns:
            return "Close"
        if "close" in raw_df.columns:
            return "close"
        return None

    @staticmethod
    def _canonicalize_price_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
        normalized = raw_df.copy()
        required = {
            "close": "Close",
        }
        optional = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        existing_keys = {str(column).strip().lower() for column in normalized.columns}
        missing = [name for name in required if name not in existing_keys]
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"missing_price_columns({missing_cols})")
        canonical_names = {**required, **optional}
        rename_map: dict[Any, str] = {}
        for original in normalized.columns:
            source = str(original).strip().lower()
            target = canonical_names.get(source)
            if target is not None:
                rename_map[original] = target
        normalized = normalized.rename(columns=rename_map)
        if normalized.columns.has_duplicates:
            normalized = normalized.loc[:, ~normalized.columns.duplicated(keep="last")]
        if "Volume" not in normalized.columns:
            normalized["Volume"] = 0.0
        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized

    @classmethod
    def _canonicalize_datetime_index(
        cls,
        raw_df: pd.DataFrame,
        calendar_timezone: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        normalized = raw_df.copy()
        original_rows = int(len(normalized))
        idx = pd.to_datetime(normalized.index, errors="coerce", utc=True)
        valid_mask = pd.notna(idx)
        invalid_timestamp_rows = int((~valid_mask).sum())
        if not bool(valid_mask.all()):
            normalized = normalized.loc[valid_mask]
            idx = idx[valid_mask]
        if len(normalized) == 0:
            raise ValueError("empty_dataframe_after_timestamp_normalization")
        dt_idx_utc = pd.DatetimeIndex(idx)
        dt_idx = (
            dt_idx_utc.tz_convert(None)
            if calendar_timezone is None
            else cls._normalize_for_calendar_timezone(dt_idx_utc, calendar_timezone)
        )
        normalized.index = dt_idx
        if not normalized.index.is_monotonic_increasing:
            normalized = normalized.sort_index()
        rows_after_timestamp = int(len(normalized))
        duplicate_bars_removed = 0
        if normalized.index.has_duplicates:
            duplicate_bars_removed = int(normalized.index.duplicated(keep="last").sum())
            normalized = normalized[~normalized.index.duplicated(keep="last")]
        if len(normalized) == 0:
            raise ValueError("empty_dataframe_after_deduplication")
        diagnostics = {
            "input_rows": original_rows,
            "invalid_timestamp_rows_removed": invalid_timestamp_rows,
            "duplicate_bars_removed": duplicate_bars_removed,
            "output_rows": int(len(normalized)),
            "timestamp_normalized_rows": rows_after_timestamp,
        }
        return normalized, diagnostics

    @classmethod
    def _canonicalize_validation_frame(
        cls,
        raw_df: pd.DataFrame,
        calendar_timezone: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        canonical = cls._canonicalize_price_columns(raw_df)
        return cls._canonicalize_datetime_index(canonical, calendar_timezone=calendar_timezone)

    @staticmethod
    def _ohlc_integrity_reason(
        raw_df: pd.DataFrame,
        ohlc_integrity_cfg: ValidationOHLCIntegrityConfig | None,
    ) -> str | None:
        if ohlc_integrity_cfg is None:
            return None
        if raw_df.empty:
            return "ohlc_integrity_indeterminate(reason=empty_dataframe)"
        required_columns = ("Open", "High", "Low", "Close", "Volume")
        missing_columns = [name.lower() for name in required_columns if name not in raw_df.columns]
        if missing_columns:
            missing = ",".join(missing_columns)
            return f"ohlc_integrity_indeterminate(reason=missing_price_columns({missing}))"
        open_values = raw_df["Open"].to_numpy(dtype=float)
        high_values = raw_df["High"].to_numpy(dtype=float)
        low_values = raw_df["Low"].to_numpy(dtype=float)
        close_values = raw_df["Close"].to_numpy(dtype=float)
        volume_values = raw_df["Volume"].to_numpy(dtype=float)
        non_finite = ~(
            np.isfinite(open_values)
            & np.isfinite(high_values)
            & np.isfinite(low_values)
            & np.isfinite(close_values)
            & np.isfinite(volume_values)
        )
        high_below_low = high_values < low_values
        high_below_body = high_values < np.maximum(open_values, close_values)
        low_above_body = low_values > np.minimum(open_values, close_values)
        negative_price = (
            np.zeros(len(raw_df), dtype=bool)
            if bool(ohlc_integrity_cfg.allow_negative_price)
            else (open_values < 0) | (high_values < 0) | (low_values < 0) | (close_values < 0)
        )
        negative_volume = (
            np.zeros(len(raw_df), dtype=bool)
            if bool(ohlc_integrity_cfg.allow_negative_volume)
            else (volume_values < 0)
        )
        invalid_mask = non_finite | high_below_low | high_below_body | low_above_body | negative_price | negative_volume
        invalid_bars = int(invalid_mask.sum())
        if invalid_bars <= 0:
            return None
        total = int(len(raw_df))
        invalid_pct = (float(invalid_bars) / float(total)) * 100.0
        threshold = float(ohlc_integrity_cfg.max_invalid_bar_pct or 0.0)
        if invalid_pct <= threshold:
            return None
        return (
            "ohlc_integrity_invalid_bar_pct_exceeded("
            f"max_allowed={threshold}, "
            f"available={invalid_pct}, "
            f"invalid_bars={invalid_bars}, "
            f"total_bars={total}, "
            f"non_finite={int(non_finite.sum())}, "
            f"high_below_low={int(high_below_low.sum())}, "
            f"high_below_body={int(high_below_body.sum())}, "
            f"low_above_body={int(low_above_body.sum())}, "
            f"negative_price={int(negative_price.sum())}, "
            f"negative_volume={int(negative_volume.sum())}"
            ")"
        )

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

    @classmethod
    def _stationarity_close_returns(
        cls,
        raw_df: pd.DataFrame,
    ) -> tuple[pd.Series | None, str | None]:
        close_col = cls._resolve_close_column(raw_df)
        if close_col is None:
            return None, "missing_close_column"
        close = raw_df[close_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(close) < 2:
            return None, "insufficient_close_points"
        if (close <= 0).any():
            returns = close.pct_change()
        else:
            returns = np.log(close).diff()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return None, "insufficient_return_points"
        return returns.astype(float), None

    @staticmethod
    def _stationarity_adfuller() -> Any | None:
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency unavailable
            return None
        return adfuller

    @staticmethod
    def _stationarity_kpss() -> Any | None:
        try:
            from statsmodels.tsa.stattools import kpss  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency unavailable
            return None
        return kpss

    @classmethod
    def _stationarity_adf_assessment(
        cls,
        raw_df: pd.DataFrame,
        stationarity_cfg: ValidationStationarityConfig | None,
        *,
        returns: pd.Series | None = None,
        returns_issue: str | None = None,
    ) -> tuple[bool | None, str | None, float | None]:
        if stationarity_cfg is None:
            return None, None, None
        if returns is None and returns_issue is None:
            returns, returns_issue = cls._stationarity_close_returns(raw_df)
        if returns_issue is not None:
            return None, f"stationarity_adf_indeterminate(reason={returns_issue})", None
        if returns is None:
            return None, "stationarity_adf_indeterminate(reason=missing_returns)", None
        available = len(returns)
        required = cls._stationarity_min_points(stationarity_cfg)
        if available < required:
            return (
                None,
                "stationarity_min_points_not_met("
                f"required={required}, available={available})",
                None,
            )
        adfuller = cls._stationarity_adfuller()
        if adfuller is None:
            return None, "stationarity_indeterminate(reason=statsmodels_missing)", None
        values = returns.to_numpy(dtype=float)
        if values.size < 4:
            return None, "stationarity_adf_indeterminate(reason=insufficient_points_for_adf)", None
        try:
            pvalue = float(adfuller(values, autolag="AIC")[1])
        except Exception as exc:
            get_logger().debug("stationarity adfuller failed", exc_info=exc)
            return None, "stationarity_adf_indeterminate(reason=adfuller_failed)", None
        if not np.isfinite(pvalue):
            return None, "stationarity_adf_indeterminate(reason=adfuller_non_finite)", None
        threshold = float(stationarity_cfg.adf_pvalue_max)
        if pvalue > threshold:
            return (
                False,
                "stationarity_adf_pvalue_exceeded("
                f"max_allowed={threshold}, available={pvalue})",
                pvalue,
            )
        return True, None, pvalue

    @classmethod
    def _stationarity_kpss_assessment(
        cls,
        raw_df: pd.DataFrame,
        stationarity_cfg: ValidationStationarityConfig | None,
        *,
        returns: pd.Series | None = None,
        returns_issue: str | None = None,
    ) -> tuple[bool | None, str | None, float | None]:
        if stationarity_cfg is None:
            return None, None, None
        threshold = getattr(stationarity_cfg, "kpss_pvalue_min", None)
        if threshold is None:
            return None, None, None
        if returns is None and returns_issue is None:
            returns, returns_issue = cls._stationarity_close_returns(raw_df)
        if returns_issue is not None:
            return None, f"stationarity_kpss_indeterminate(reason={returns_issue})", None
        if returns is None:
            return None, "stationarity_kpss_indeterminate(reason=missing_returns)", None
        available = len(returns)
        required = cls._stationarity_min_points(stationarity_cfg)
        if available < required:
            return (
                None,
                "stationarity_min_points_not_met("
                f"required={required}, available={available})",
                None,
            )
        kpss_fn = cls._stationarity_kpss()
        if kpss_fn is None:
            return None, "stationarity_indeterminate(reason=statsmodels_missing)", None
        values = returns.to_numpy(dtype=float)
        if values.size < 4:
            return None, "stationarity_kpss_indeterminate(reason=insufficient_points_for_kpss)", None
        try:
            pvalue = float(kpss_fn(values, nlags="auto")[1])
        except Exception as exc:
            get_logger().debug("stationarity kpss failed", exc_info=exc)
            return None, "stationarity_kpss_indeterminate(reason=kpss_failed)", None
        if not np.isfinite(pvalue):
            return None, "stationarity_kpss_indeterminate(reason=kpss_non_finite)", None
        threshold_val = float(threshold)
        if pvalue < threshold_val:
            return (
                False,
                "stationarity_kpss_pvalue_below("
                f"min_allowed={threshold_val}, available={pvalue})",
                pvalue,
            )
        return True, None, pvalue

    @staticmethod
    def _stationarity_min_points(stationarity_cfg: ValidationStationarityConfig) -> int:
        min_points = stationarity_cfg.min_points
        return STATIONARITY_MIN_POINTS_DEFAULT if min_points is None else int(min_points)

    @classmethod
    def _stationarity_regime_shift_reason(
        cls,
        raw_df: pd.DataFrame,
        stationarity_cfg: ValidationStationarityConfig | None,
        *,
        returns: pd.Series | None = None,
        returns_issue: str | None = None,
    ) -> list[str]:
        if stationarity_cfg is None or stationarity_cfg.regime_shift is None:
            return []
        if returns is None and returns_issue is None:
            returns, returns_issue = cls._stationarity_close_returns(raw_df)
        if returns_issue is not None:
            return [f"stationarity_regime_shift_indeterminate(reason={returns_issue})"]
        if returns is None:
            return ["stationarity_regime_shift_indeterminate(reason=missing_returns)"]
        regime = stationarity_cfg.regime_shift
        window = int(regime.window)
        required = max(cls._stationarity_min_points(stationarity_cfg), window * 2)
        available = len(returns)
        if available < required:
            return [
                "stationarity_regime_shift_not_enough_points("
                f"required={required}, available={available})"
            ]
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std(ddof=1)
        prior_mean = rolling_mean.shift(window)
        prior_std = rolling_std.shift(window)
        eps = 1e-12
        rolling_mean_values = rolling_mean.to_numpy(dtype=float)
        rolling_std_values = rolling_std.to_numpy(dtype=float)
        prior_mean_values = prior_mean.to_numpy(dtype=float)
        prior_std_values = prior_std.to_numpy(dtype=float)
        scale = np.maximum(np.maximum(rolling_std_values, prior_std_values), eps)
        mean_shift = np.abs(rolling_mean_values - prior_mean_values) / scale
        vol_ratio = np.maximum(
            rolling_std_values / np.maximum(prior_std_values, eps),
            prior_std_values / np.maximum(rolling_std_values, eps),
        )
        valid_mask = np.isfinite(mean_shift) & np.isfinite(vol_ratio)
        if not valid_mask.any():
            return ["stationarity_regime_shift_indeterminate(reason=insufficient_window_history)"]
        max_mean_shift = float(np.nanmax(mean_shift[valid_mask]))
        max_vol_ratio = float(np.nanmax(vol_ratio[valid_mask]))
        reasons: list[str] = []
        if max_mean_shift > regime.mean_shift_max:
            reasons.append(
                "stationarity_regime_shift_mean_shift_exceeded("
                f"max_allowed={regime.mean_shift_max}, available={max_mean_shift})"
            )
        if max_vol_ratio > regime.vol_ratio_max:
            reasons.append(
                "stationarity_regime_shift_vol_ratio_exceeded("
                f"max_allowed={regime.vol_ratio_max}, available={max_vol_ratio})"
            )
        return reasons

    @classmethod
    def _collect_reliability_reasons(
        cls,
        *,
        raw_df: pd.DataFrame,
        continuity: dict[str, float | int],
        min_data_points_cfg: int | None,
        continuity_cfg: ValidationContinuityConfig | None,
        kurtosis_cfg: float | None,
        ohlc_integrity_cfg: ValidationOHLCIntegrityConfig | None,
        outlier_detection: ValidationOutlierDetectionConfig | None,
        stationarity_cfg: ValidationStationarityConfig | None,
        is_verified: bool | None,
    ) -> list[str]:
        reasons: list[str] = []
        reason_checks = (
            cls._continuity_threshold_reason(continuity, continuity_cfg),
            cls._min_data_points_reason(raw_df, min_data_points_cfg),
            cls._missing_bar_pct_reason(continuity, continuity_cfg),
            cls._max_kurtosis_reason(raw_df, kurtosis_cfg),
            cls._ohlc_integrity_reason(raw_df, ohlc_integrity_cfg),
            cls._collection_verification_reason(is_verified),
            cls._outlier_pct_reason(
                raw_df=raw_df,
                outlier_detection=outlier_detection,
            ),
        )
        for reason in reason_checks:
            if reason is not None:
                reasons.append(reason)
        reasons.extend(cls._stationarity_reasons(raw_df, stationarity_cfg))
        return reasons

    @classmethod
    def _stationarity_reasons(
        cls,
        raw_df: pd.DataFrame,
        stationarity_cfg: ValidationStationarityConfig | None,
    ) -> list[str]:
        if stationarity_cfg is None:
            return []
        returns, returns_issue = cls._stationarity_close_returns(raw_df)
        reasons: list[str] = []
        adf_stationary, adf_reason, adf_pvalue = cls._stationarity_adf_assessment(
            raw_df,
            stationarity_cfg,
            returns=returns,
            returns_issue=returns_issue,
        )
        if adf_reason is not None:
            reasons.append(adf_reason)
        kpss_stationary, kpss_reason, kpss_pvalue = cls._stationarity_kpss_assessment(
            raw_df,
            stationarity_cfg,
            returns=returns,
            returns_issue=returns_issue,
        )
        if kpss_reason is not None and kpss_reason not in reasons:
            reasons.append(kpss_reason)
        if (
            adf_stationary is not None
            and kpss_stationary is not None
            and adf_stationary != kpss_stationary
        ):
            adf_status = cls._STATIONARITY_TRUE if adf_stationary else cls._STATIONARITY_FALSE
            kpss_status = cls._STATIONARITY_TRUE if kpss_stationary else cls._STATIONARITY_FALSE
            reasons.append(
                "stationarity_test_conflict("
                f"adf_stationary={adf_status}, "
                f"kpss_stationary={kpss_status}, "
                f"adf_pvalue={adf_pvalue}, "
                f"kpss_pvalue={kpss_pvalue})"
            )
        reasons.extend(
            cls._stationarity_regime_shift_reason(
                raw_df,
                stationarity_cfg,
                returns=returns,
                returns_issue=returns_issue,
            )
        )
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

    @staticmethod
    def _lookahead_shuffle_seed(
        base_seed: int,
        collection: str,
        symbol: str,
        timeframe: str,
        strategy: str,
    ) -> int:
        payload = f"{base_seed}:{collection}:{symbol}:{timeframe}:{strategy}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _generate_aligned_signals(
        self,
        strategy: BaseStrategy,
        raw_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        plan: StrategyPlan | None = None,
        state: JobState | None = None,
        track_runtime_errors: bool = False,
    ) -> tuple[pd.Series, pd.Series]:
        try:
            call_params = params.copy()
            entries, exits = strategy.generate_signals(raw_df, call_params)
        except Exception as exc:
            if track_runtime_errors and plan is not None and state is not None:
                self._record_runtime_signal_failure(
                    plan=plan,
                    state=state,
                    full_params=params,
                    exc=exc,
                )
            raise
        entries = entries.reindex(raw_df.index, fill_value=False)
        exits = exits.reindex(raw_df.index, fill_value=False)
        return entries, exits

    def _evaluate_strategy_outcome(
        self,
        request: EvaluationRequest,
        prepared: ExecutionPreparedData,
        entries: pd.Series,
        exits: pd.Series,
    ) -> EvaluationOutcome:
        return self._get_evaluator().evaluate(
            request,
            prepared.data_frame,
            prepared.dates,
            entries,
            exits,
            prepared.fractional,
        )

    @staticmethod
    def _lookahead_shuffle_indeterminate_reason(reason: str) -> str:
        return f"lookahead_shuffle_test_indeterminate(reason={reason})"

    def _lookahead_shuffle_indeterminate(
        self,
        reason: str,
        *,
        policy: ValidationLookaheadShuffleTestConfig | None = None,
        seed: int | None = None,
        failed_permutations: int | None = None,
        max_failed_permutations: int | None = None,
        include_max_failed_permutations: bool = False,
        reason_detail: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "is_complete": False,
            "reason": reason,
            "metric_name": self.cfg.metric,
        }
        if reason_detail is not None:
            diagnostics["reason_detail"] = reason_detail
        if policy is not None:
            diagnostics["permutations"] = policy.permutations
        if seed is not None:
            diagnostics["seed"] = seed
        if failed_permutations is not None:
            diagnostics["failed_permutations"] = failed_permutations
        if include_max_failed_permutations or max_failed_permutations is not None:
            diagnostics["max_failed_permutations"] = max_failed_permutations
        return self._lookahead_shuffle_indeterminate_reason(reason), diagnostics

    @staticmethod
    def _attach_post_run_meta(
        outcome: StrategyEvalOutcome,
        key: str,
        meta: dict[str, Any] | None,
    ) -> None:
        if meta is None or not isinstance(outcome.best_stats, dict):
            return
        best_stats = dict(outcome.best_stats)
        existing_post_run_meta = best_stats.get("post_run_meta")
        post_run_meta = (
            dict(existing_post_run_meta) if isinstance(existing_post_run_meta, dict) else {}
        )
        post_run_meta[key] = dict(meta)
        best_stats["post_run_meta"] = post_run_meta
        outcome.best_stats = best_stats

    def _lookahead_shuffle_execution_context(
        self,
        context: ValidationContext,
        derived_seed: int,
    ) -> tuple[Any, bool, int, float, float, np.random.Generator]:
        _, _, _, _, data_col_enum = self._ensure_pybroker()
        fractional = self._fractional_enabled(context.job.collection, context.job.symbol)
        bars_per_year = self._bars_per_year(context.job.timeframe)
        fees, slippage = self._fees_slippage_for(context.job.collection)
        rng = np.random.default_rng(derived_seed)
        return data_col_enum, fractional, bars_per_year, fees, slippage, rng

    def _lookahead_shuffle_prepared_data(
        self,
        shuffled_raw: pd.DataFrame,
        close_col: str,
        context: ValidationContext,
        data_col_enum: Any,
        fractional: bool,
        bars_per_year: int,
        fees: float,
        slippage: float,
    ) -> ExecutionPreparedData:
        data_frame, dates = self._prepare_pybroker_frame(
            shuffled_raw,
            context.job.symbol,
            data_col_enum,
        )
        return ExecutionPreparedData(
            data_frame=data_frame,
            dates=dates,
            fees=fees,
            slippage=slippage,
            fractional=fractional,
            bars_per_year=bars_per_year,
            fingerprint=(
                f"{len(shuffled_raw)}:{shuffled_raw.index[-1].isoformat()}:"
                f"{float(shuffled_raw[close_col].astype(float).iloc[-1])}"
            ),
        )

    def _run_lookahead_shuffle_permutations(
        self,
        run_ctx: LookaheadShuffleRunContext,
    ) -> tuple[list[float], int, tuple[str, dict[str, Any]] | None]:
        metric_values: list[float] = []
        failed_permutations = 0
        shuffle_plan = StrategyPlan(
            strategy=run_ctx.plan.strategy,
            fixed_params=run_ctx.plan.fixed_params.copy(),
            search_space={},
            search_method=run_ctx.plan.search_method,
            trials_target=run_ctx.plan.trials_target,
        )
        for _ in range(run_ctx.policy.permutations):
            permutation = run_ctx.rng.permutation(len(run_ctx.raw_df))
            shuffled_raw = run_ctx.raw_df.iloc[permutation].copy()
            shuffled_raw.index = run_ctx.raw_df.index
            prepared = self._lookahead_shuffle_prepared_data(
                shuffled_raw=shuffled_raw,
                close_col=run_ctx.close_col,
                context=run_ctx.context,
                data_col_enum=run_ctx.data_col_enum,
                fractional=run_ctx.fractional,
                bars_per_year=run_ctx.bars_per_year,
                fees=run_ctx.fees,
                slippage=run_ctx.slippage,
            )
            full_params = {**run_ctx.plan.fixed_params, **run_ctx.effective_params}
            try:
                entries, exits = self._generate_aligned_signals(
                    shuffle_plan.strategy,
                    shuffled_raw,
                    full_params,
                    plan=None,
                    state=None,
                    track_runtime_errors=False,
                )
                request = self._build_evaluation_request(
                    shuffle_plan,
                    run_ctx.context.state,
                    prepared,
                    full_params,
                    cacheable=False,
                )
                outcome = self._evaluate_strategy_outcome(request, prepared, entries, exits)
            except Exception as exc:
                failed_permutations += 1
                if (
                    run_ctx.max_failed_permutations is not None
                    and failed_permutations > run_ctx.max_failed_permutations
                ):
                    return metric_values, failed_permutations, self._lookahead_shuffle_indeterminate(
                        "too_many_failed_permutations",
                        policy=run_ctx.policy,
                        seed=run_ctx.derived_seed,
                        failed_permutations=failed_permutations,
                        max_failed_permutations=run_ctx.max_failed_permutations,
                        include_max_failed_permutations=True,
                        reason_detail=str(exc),
                    )
                continue
            if outcome.metric_computed and np.isfinite(outcome.metric_value):
                metric_values.append(float(outcome.metric_value))
        return metric_values, failed_permutations, None

    def _lookahead_shuffle_test_result(
        self,
        context: ValidationContext,
        plan: StrategyPlan,
        policy: ValidationLookaheadShuffleTestConfig | None,
        observed_metric: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if policy is None or context.validated_data is None:
            return None, None

        raw_df = context.validated_data.raw_df
        if raw_df.empty:
            return self._lookahead_shuffle_indeterminate("empty_dataframe")

        derived_seed = self._lookahead_shuffle_seed(
            policy.seed,
            context.job.collection.name,
            context.job.symbol,
            context.job.timeframe,
            plan.strategy.name,
        )
        close_col = self._resolve_close_column(raw_df)
        if close_col is None:
            return self._lookahead_shuffle_indeterminate("missing_close_column")
        try:
            data_col_enum, fractional, bars_per_year, fees, slippage, rng = (
                self._lookahead_shuffle_execution_context(context, derived_seed)
            )
            effective_params = params or {}
            max_failed_permutations = getattr(policy, "max_failed_permutations", None)
            max_failed_permutations = (
                int(max_failed_permutations) if max_failed_permutations is not None else None
            )
            metric_values, failed_permutations, early_result = self._run_lookahead_shuffle_permutations(
                LookaheadShuffleRunContext(
                    context=context,
                    plan=plan,
                    policy=policy,
                    raw_df=raw_df,
                    close_col=close_col,
                    data_col_enum=data_col_enum,
                    fractional=fractional,
                    bars_per_year=bars_per_year,
                    fees=fees,
                    slippage=slippage,
                    rng=rng,
                    effective_params=effective_params,
                    max_failed_permutations=max_failed_permutations,
                    derived_seed=derived_seed,
                )
            )
            if early_result is not None:
                return early_result

            if not metric_values:
                return self._lookahead_shuffle_indeterminate(
                    "no_finite_metrics",
                    policy=policy,
                    seed=derived_seed,
                    failed_permutations=failed_permutations,
                    max_failed_permutations=max_failed_permutations,
                    include_max_failed_permutations=True,
                )

            metric_array = np.asarray(metric_values, dtype=float)
            median_metric = float(np.median(metric_array))
            diagnostics = {
                "is_complete": True,
                "permutations": policy.permutations,
                "seed": derived_seed,
                "metric_name": self.cfg.metric,
                "pvalue_max": getattr(policy, "pvalue_max", None),
                "finite_permutations": int(metric_array.size),
                "failed_permutations": failed_permutations,
                "max_failed_permutations": max_failed_permutations,
                "median_shuffled_metric": median_metric,
                "min_shuffled_metric": float(np.min(metric_array)),
                "max_shuffled_metric": float(np.max(metric_array)),
            }
            pvalue_max = getattr(policy, "pvalue_max", None)
            if pvalue_max is not None and observed_metric is not None and np.isfinite(observed_metric):
                observed_metric_val = float(observed_metric)
                exceed_count = int(np.sum(metric_array >= observed_metric_val))
                pvalue = float((exceed_count + 1) / (metric_array.size + 1))
                diagnostics["observed_metric"] = observed_metric_val
                diagnostics["shuffle_pvalue"] = pvalue
                if pvalue > float(pvalue_max):
                    reason = (
                        "lookahead_shuffle_test_pvalue_exceeded("
                        f"metric={self.cfg.metric}, "
                        f"pvalue_max={pvalue_max}, "
                        f"available={pvalue}, "
                        f"observed_metric={observed_metric_val}, "
                        f"permutations={policy.permutations}, "
                        f"seed={derived_seed})"
                    )
                    return reason, diagnostics
            return None, diagnostics
        except Exception as exc:
            return self._lookahead_shuffle_indeterminate(
                str(exc),
                policy=policy,
                seed=derived_seed,
            )

    @staticmethod
    def _safe_float_stat(stats: dict[str, Any], key: str) -> float | None:
        value = stats.get(key)
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _transaction_cost_robustness_indeterminate(
        self,
        reason: str,
        *,
        policy: ResultConsistencyTransactionCostRobustnessConfig | None = None,
        stress_scenarios: list[dict[str, Any]] | None = None,
        breakeven: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "is_complete": False,
            "status": "indeterminate",
            "reason": reason,
            "metric_name": self.cfg.metric,
        }
        if policy is not None:
            diagnostics["mode"] = getattr(policy, "mode", None)
            diagnostics["stress_multipliers"] = list(getattr(policy, "stress_multipliers", []) or [])
            diagnostics["max_metric_drop_pct"] = getattr(policy, "max_metric_drop_pct", None)
        if stress_scenarios is not None:
            diagnostics["stress_scenarios"] = stress_scenarios
        if breakeven is not None:
            diagnostics["breakeven"] = breakeven
        return f"transaction_cost_robustness_indeterminate(reason={reason})", diagnostics

    def _transaction_cost_robustness_scenario(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
        multiplier: float,
    ) -> dict[str, Any]:
        prepared = run_ctx.prepared
        stressed_fees = float(prepared.fees) * float(multiplier)
        stressed_slippage = float(prepared.slippage) * float(multiplier)
        raw_df = run_ctx.context.validated_data.raw_df if run_ctx.context.validated_data else None
        if raw_df is None:
            return {
                "is_complete": False,
                "status": "indeterminate",
                "reason": "missing_validated_data",
                "metric_name": self.cfg.metric,
                "multiplier": float(multiplier),
                "fees": stressed_fees,
                "slippage": stressed_slippage,
            }
        try:
            if run_ctx.aligned_signals is None:
                run_ctx.aligned_signals = self._generate_aligned_signals(
                    run_ctx.plan.strategy,
                    raw_df,
                    run_ctx.full_params,
                    plan=run_ctx.plan,
                    state=run_ctx.context.state,
                    track_runtime_errors=False,
                )
            entries, exits = run_ctx.aligned_signals
            request = self._build_evaluation_request(
                run_ctx.plan,
                run_ctx.context.state,
                prepared,
                run_ctx.full_params,
                cacheable=False,
                fees=stressed_fees,
                slippage=stressed_slippage,
            )
            outcome = self._evaluate_strategy_outcome(request, prepared, entries, exits)
        except Exception as exc:
            return {
                "is_complete": False,
                "status": "indeterminate",
                "reason": str(exc),
                "metric_name": self.cfg.metric,
                "multiplier": float(multiplier),
                "fees": stressed_fees,
                "slippage": stressed_slippage,
            }

        metric_value = self._safe_float_stat(
            {"metric_value": outcome.metric_value},
            "metric_value",
        )
        profit = self._safe_float_stat(outcome.stats, "profit") if isinstance(outcome.stats, dict) else None
        metric_drop_pct = self._transaction_cost_metric_drop_pct(
            run_ctx.baseline_metric,
            metric_value,
        )
        is_complete = bool(outcome.valid and outcome.metric_computed and metric_value is not None and profit is not None)
        scenario: dict[str, Any] = {
            "is_complete": is_complete,
            "status": "complete" if is_complete else "indeterminate",
            "metric_name": self.cfg.metric,
            "multiplier": float(multiplier),
            "fees": stressed_fees,
            "slippage": stressed_slippage,
            "baseline_metric": run_ctx.baseline_metric,
            "baseline_profit": run_ctx.baseline_profit,
            "metric_value": metric_value,
            "profit": profit,
            "metric_drop_pct": metric_drop_pct,
            "metric_drop_exceeded": self._transaction_cost_drop_exceeds_threshold(
                metric_drop_pct,
                run_ctx.policy.max_metric_drop_pct,
            ),
            "profit_negative": profit is not None and profit < 0.0,
        }
        if not is_complete:
            scenario["reason"] = "missing_metric_or_profit"
        return scenario

    @staticmethod
    def _transaction_cost_metric_drop_pct(
        baseline_metric: float | None,
        stressed_metric: float | None,
    ) -> float | None:
        if baseline_metric is None or stressed_metric is None:
            return None
        if not np.isfinite(baseline_metric) or not np.isfinite(stressed_metric):
            return None
        if baseline_metric <= 0.0:
            return None
        return max(0.0, (baseline_metric - stressed_metric) / baseline_metric)

    def _transaction_cost_drop_exceeds_threshold(
        self,
        metric_drop_pct: float | None,
        threshold: float | None,
    ) -> bool:
        if metric_drop_pct is None or threshold is None:
            return False
        return float(metric_drop_pct) > float(threshold) + self._TRANSACTION_COST_ROBUSTNESS_DROP_EPSILON

    @staticmethod
    def _transaction_cost_drop_exceeds_threshold_strict(
        metric_drop_pct: float | None,
        threshold: float | None,
    ) -> bool:
        if metric_drop_pct is None or threshold is None:
            return False
        return float(metric_drop_pct) > float(threshold)

    def _transaction_cost_breakeven_base_meta(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
    ) -> dict[str, Any]:
        breakeven_cfg = run_ctx.policy.breakeven
        return {
            "enabled": bool(getattr(breakeven_cfg, "enabled", False)),
            "min_multiplier": getattr(breakeven_cfg, "min_multiplier", None),
            "max_multiplier": getattr(breakeven_cfg, "max_multiplier", None),
            "max_iterations": getattr(breakeven_cfg, "max_iterations", None),
            "tolerance": getattr(breakeven_cfg, "tolerance", None),
            "metric_name": self.cfg.metric,
            "threshold": run_ctx.policy.max_metric_drop_pct,
        }

    @staticmethod
    def _transaction_cost_breakeven_indeterminate(
        base_meta: dict[str, Any],
        reason: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            **base_meta,
            "status": "indeterminate",
            "reason": reason,
            **extra,
        }

    @staticmethod
    def _transaction_cost_breakeven_boundary_incomplete(
        min_result: dict[str, Any],
        max_result: dict[str, Any],
    ) -> bool:
        return not bool(min_result.get("is_complete")) or not bool(max_result.get("is_complete"))

    def _transaction_cost_breakeven_boundary_status(
        self,
        base_meta: dict[str, Any],
        *,
        min_multiplier: float,
        max_multiplier: float,
        threshold: float,
        min_result: dict[str, Any],
        max_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        min_drop = min_result.get("metric_drop_pct")
        max_drop = max_result.get("metric_drop_pct")
        if self._transaction_cost_breakeven_boundary_incomplete(min_result, max_result):
            return self._transaction_cost_breakeven_indeterminate(
                base_meta,
                "incomplete_boundary_evaluations",
                boundary_results=[min_result, max_result],
            )
        if min_drop is None or max_drop is None:
            return self._transaction_cost_breakeven_indeterminate(
                base_meta,
                "missing_boundary_metric_drop",
                boundary_results=[min_result, max_result],
            )
        if self._transaction_cost_drop_exceeds_threshold(float(min_drop), threshold):
            return {
                **base_meta,
                "status": "below_range",
                "estimated_multiplier": min_multiplier,
                "metric_drop_pct": float(min_drop),
                "boundary_results": [min_result, max_result],
            }
        if not self._transaction_cost_drop_exceeds_threshold(float(max_drop), threshold):
            return {
                **base_meta,
                "status": "above_range",
                "estimated_multiplier": max_multiplier,
                "metric_drop_pct": float(max_drop),
                "boundary_results": [min_result, max_result],
            }
        return None

    def _transaction_cost_breakeven_binary_search(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
        *,
        min_multiplier: float,
        max_multiplier: float,
        threshold: float,
        max_iterations: int,
        tolerance: float,
        min_result: dict[str, Any],
        max_result: dict[str, Any],
        base_meta: dict[str, Any],
    ) -> dict[str, Any]:
        low_multiplier = min_multiplier
        high_multiplier = max_multiplier
        low_drop = float(min_result.get("metric_drop_pct"))
        high_drop = float(max_result.get("metric_drop_pct"))
        latest_result: dict[str, Any] | None = None
        iterations = 0
        while iterations < max_iterations and high_multiplier - low_multiplier > tolerance:
            mid_multiplier = (low_multiplier + high_multiplier) / 2.0
            mid_result = self._transaction_cost_robustness_scenario(run_ctx, mid_multiplier)
            if not bool(mid_result.get("is_complete")):
                return self._transaction_cost_breakeven_indeterminate(
                    base_meta,
                    str(mid_result.get("reason", "incomplete_midpoint_evaluation")),
                    boundary_results=[min_result, max_result],
                    iterations=iterations + 1,
                )
            mid_drop = mid_result.get("metric_drop_pct")
            if mid_drop is None:
                return self._transaction_cost_breakeven_indeterminate(
                    base_meta,
                    "missing_midpoint_metric_drop",
                    boundary_results=[min_result, max_result],
                    iterations=iterations + 1,
                )
            latest_result = mid_result
            iterations += 1
            if self._transaction_cost_drop_exceeds_threshold_strict(float(mid_drop), threshold):
                high_multiplier = mid_multiplier
                high_drop = float(mid_drop)
            else:
                low_multiplier = mid_multiplier
                low_drop = float(mid_drop)
        estimated_multiplier = (low_multiplier + high_multiplier) / 2.0
        estimated_drop = (
            high_drop
            if self._transaction_cost_drop_exceeds_threshold_strict(high_drop, threshold)
            else low_drop
        )
        return {
            **base_meta,
            "status": "found",
            "estimated_multiplier": estimated_multiplier,
            "metric_drop_pct": estimated_drop,
            "lower_multiplier": low_multiplier,
            "upper_multiplier": high_multiplier,
            "iterations": iterations,
            "latest_result": latest_result,
            "boundary_results": [min_result, max_result],
        }

    def _transaction_cost_breakeven_result(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
    ) -> dict[str, Any] | None:
        breakeven_cfg = run_ctx.policy.breakeven
        if breakeven_cfg is None:
            return None
        base_meta = self._transaction_cost_breakeven_base_meta(run_ctx)
        if not base_meta["enabled"]:
            return self._transaction_cost_breakeven_indeterminate(base_meta, "disabled")
        if (
            run_ctx.baseline_metric is None
            or run_ctx.baseline_metric <= 0.0
            or not np.isfinite(run_ctx.baseline_metric)
        ):
            return self._transaction_cost_breakeven_indeterminate(base_meta, "invalid_baseline_metric")

        min_multiplier = float(getattr(breakeven_cfg, "min_multiplier"))
        max_multiplier = float(getattr(breakeven_cfg, "max_multiplier"))
        max_iterations = int(getattr(breakeven_cfg, "max_iterations"))
        tolerance = float(getattr(breakeven_cfg, "tolerance"))
        threshold = float(run_ctx.policy.max_metric_drop_pct)

        min_result = self._transaction_cost_robustness_scenario(run_ctx, min_multiplier)
        max_result = self._transaction_cost_robustness_scenario(run_ctx, max_multiplier)
        boundary_status = self._transaction_cost_breakeven_boundary_status(
            base_meta,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            threshold=threshold,
            min_result=min_result,
            max_result=max_result,
        )
        if boundary_status is not None:
            return boundary_status
        return self._transaction_cost_breakeven_binary_search(
            run_ctx,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            threshold=threshold,
            max_iterations=max_iterations,
            tolerance=tolerance,
            min_result=min_result,
            max_result=max_result,
            base_meta=base_meta,
        )

    def _transaction_cost_robustness_stress_scenarios(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
    ) -> list[dict[str, Any]]:
        return [
            self._transaction_cost_robustness_scenario(run_ctx, float(multiplier))
            for multiplier in (run_ctx.policy.stress_multipliers or [])
        ]

    def _transaction_cost_robustness_meta(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
        stress_scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        is_complete = all(bool(scenario.get("is_complete")) for scenario in stress_scenarios)
        return {
            "is_complete": is_complete,
            "status": "complete" if is_complete else "indeterminate",
            "mode": run_ctx.policy.mode,
            "metric_name": self.cfg.metric,
            "baseline_metric": run_ctx.baseline_metric,
            "baseline_profit": run_ctx.baseline_profit,
            "max_metric_drop_pct": run_ctx.policy.max_metric_drop_pct,
            "stress_multipliers": [scenario.get("multiplier") for scenario in stress_scenarios],
            "stress_scenarios": stress_scenarios,
        }

    def _transaction_cost_robustness_smallest_drop_breach_reason(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
        meta: dict[str, Any],
        stress_scenarios: list[dict[str, Any]],
    ) -> str | None:
        if not stress_scenarios:
            return None
        smallest = stress_scenarios[0]
        smallest_drop = smallest.get("metric_drop_pct")
        if not smallest.get("is_complete") or smallest_drop is None:
            return None
        meta["smallest_multiplier_metric_drop_pct"] = float(smallest_drop)
        if not self._transaction_cost_drop_exceeds_threshold(
            float(smallest_drop),
            run_ctx.policy.max_metric_drop_pct,
        ):
            return None
        return (
            "transaction_cost_robustness_metric_drop_exceeded("
            f"multiplier={smallest.get('multiplier')}, "
            f"drop_pct={float(smallest_drop)}, "
            f"threshold={run_ctx.policy.max_metric_drop_pct})"
        )

    @staticmethod
    def _transaction_cost_robustness_negative_profit_breach_reasons(
        stress_scenarios: list[dict[str, Any]],
    ) -> list[str]:
        reasons: list[str] = []
        for scenario in stress_scenarios:
            if scenario.get("is_complete") and scenario.get("profit_negative"):
                reasons.append(
                    "transaction_cost_robustness_negative_profit("
                    f"multiplier={scenario.get('multiplier')}, "
                    f"profit={scenario.get('profit')})"
                )
        return reasons

    def _transaction_cost_robustness_attach_breakeven(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
        meta: dict[str, Any],
        breach_reasons: list[str],
    ) -> None:
        breakeven_meta = self._transaction_cost_breakeven_result(run_ctx)
        if breakeven_meta is None:
            return
        meta["breakeven"] = breakeven_meta
        is_enabled = bool(breakeven_meta.get("enabled"))
        is_indeterminate = breakeven_meta.get("status") == "indeterminate"
        if not (is_enabled and is_indeterminate):
            return
        meta["is_complete"] = False
        if run_ctx.policy.mode == "enforce":
            breach_reasons.append(
                "transaction_cost_robustness_indeterminate("
                f"reason={breakeven_meta.get('reason')})"
            )

    def _transaction_cost_robustness_result(
        self,
        run_ctx: TransactionCostRobustnessRunContext,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if run_ctx.context.validated_data is None:
            return self._transaction_cost_robustness_indeterminate(
                "missing_validated_data",
                policy=run_ctx.policy,
            )
        if run_ctx.context.prepared_data is None:
            return self._transaction_cost_robustness_indeterminate(
                "missing_prepared_data",
                policy=run_ctx.policy,
            )
        stress_scenarios = self._transaction_cost_robustness_stress_scenarios(run_ctx)
        if not stress_scenarios:
            return self._transaction_cost_robustness_indeterminate(
                "missing_stress_multipliers",
                policy=run_ctx.policy,
            )
        meta = self._transaction_cost_robustness_meta(run_ctx, stress_scenarios)
        breach_reasons: list[str] = []
        smallest_breach = self._transaction_cost_robustness_smallest_drop_breach_reason(
            run_ctx,
            meta,
            stress_scenarios,
        )
        if smallest_breach is not None:
            breach_reasons.append(smallest_breach)
        breach_reasons.extend(
            self._transaction_cost_robustness_negative_profit_breach_reasons(stress_scenarios)
        )
        self._transaction_cost_robustness_attach_breakeven(run_ctx, meta, breach_reasons)
        meta["status"] = "complete" if meta["is_complete"] else "indeterminate"
        if not meta["is_complete"] and run_ctx.policy.mode == "enforce":
            if not breach_reasons:
                breach_reasons.append(
                    "transaction_cost_robustness_indeterminate(reason=incomplete_scenario_evaluation)"
                )
            meta["breach_reasons"] = list(breach_reasons)
            return breach_reasons[0], meta
        meta["breach_reasons"] = list(breach_reasons)
        if breach_reasons and run_ctx.policy.mode == "enforce":
            return breach_reasons[0], meta
        return None, meta

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

        optimization_on_fail, min_bars_cfg, dof_multiplier, _ = policy

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
        request = self._build_evaluation_request(plan, state, prepared, full_params)
        cached = None
        if request.cacheable:
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
                strategy_fingerprint=request.strategy_fingerprint,
            )
        if cached is not None:
            return self._apply_cached_evaluation(plan, validated_data, request, cached, full_params)
        self.metrics["result_cache_misses"] += 1
        try:
            entries, exits = self._generate_aligned_signals(
                plan.strategy,
                validated_data.raw_df,
                full_params,
                plan=plan,
                state=state,
                track_runtime_errors=True,
            )
        except Exception:
            return float("nan")
        outcome = self._evaluate_strategy_outcome(request, prepared, entries, exits)
        return self._apply_fresh_evaluation(
            plan=plan,
            state=state,
            validated_data=validated_data,
            request=request,
            outcome=outcome,
            full_params=full_params,
        )

    def _strategy_fingerprint(self, strategy: BaseStrategy) -> str:
        strategy_cls = type(strategy)
        cached = self._strategy_fingerprint_cache.get(strategy_cls)
        if cached is not None:
            return cached
        payload: dict[str, Any] = {
            "module": getattr(strategy_cls, "__module__", ""),
            "qualname": getattr(strategy_cls, "__qualname__", strategy_cls.__name__),
        }
        try:
            payload["source"] = inspect.getsource(strategy_cls)
        except (OSError, TypeError):
            try:
                payload["source"] = inspect.getsource(strategy_cls.generate_signals)
            except (OSError, TypeError, AttributeError):
                payload["source"] = repr(strategy_cls)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._strategy_fingerprint_cache[strategy_cls] = digest
        return digest

    def _build_evaluation_request(
        self,
        plan: StrategyPlan,
        state: JobState,
        prepared: ExecutionPreparedData,
        full_params: dict[str, Any],
        *,
        cacheable: bool = True,
        fees: float | None = None,
        slippage: float | None = None,
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
            fees=prepared.fees if fees is None else float(fees),
            slippage=prepared.slippage if slippage is None else float(slippage),
            bars_per_year=prepared.bars_per_year,
            mode_config=self.mode_config,
            cacheable=cacheable,
            strategy_fingerprint=self._strategy_fingerprint(plan.strategy),
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
        metric_val = float(cached["metric_value"])
        plan.evaluations += 1
        if not np.isfinite(metric_val):
            return float("-inf")
        self.metrics["result_cache_hits"] += 1
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
        cached_metric_val = metric_val if outcome.valid else float("-inf")
        if request.cacheable and (outcome.valid or outcome.metric_computed):
            self._evaluation_cache_set(
                collection=request.collection,
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                params=request.params,
                metric_name=request.metric_name,
                metric_value=cached_metric_val,
                stats=raw_stats,
                data_fingerprint=request.data_fingerprint,
                fees=request.fees,
                slippage=request.slippage,
                evaluation_mode=self.mode_config.mode,
                mode_config_hash=self.mode_config_hash,
                validation_config_hash=state.validation_config_hash,
                strategy_fingerprint=request.strategy_fingerprint,
            )
            self._cache_set(
                collection=request.collection,
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                params=request.params,
                metric_name=request.metric_name,
                metric_value=cached_metric_val,
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
            if self._is_runtime_error_tuple_capped(plan, state):
                self._plan_add_skip_reason(plan, "runtime_error_threshold_exceeded")
                return self._build_strategy_eval_outcome(plan, state)
            if not plan.search_space or plan.skip_optimization:
                self._strategy_evaluation(plan, state, validated_data, prepared, {})
            else:
                search_method = self._resolve_search_method(plan.search_method)
                if search_method == "optuna":
                    self._run_optuna_strategy_search(plan, state, validated_data, prepared)
                else:
                    for params in self._grid(plan.search_space):
                        if self._is_runtime_error_tuple_capped(plan, state):
                            break
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
            if self._is_runtime_error_tuple_capped(plan, state):
                return float("-inf")
            var_params = {
                name: trial.suggest_categorical(name, options) for name, options in space_items
            }
            result = self._strategy_evaluation(plan, state, validated_data, prepared, var_params)
            return result if np.isfinite(result) else float("-inf")

        def _stop_on_runtime_error_threshold(study, _trial):
            if self._is_runtime_error_tuple_capped(plan, state):
                study.stop()

        n_trials = min(plan.trials_target, self._total_search_combinations(plan.search_space))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[_stop_on_runtime_error_threshold])

    @staticmethod
    def _runtime_error_tuple_key(plan: StrategyPlan, state: JobState) -> tuple[str, str, str, str]:
        return (
            state.job.collection.name,
            plan.strategy.name,
            state.job.symbol,
            state.job.timeframe,
        )

    def _runtime_signal_error_limit(self, collection: CollectionConfig) -> int | None:
        policy = self._load_optimization_policy(collection)
        if policy is None:
            return None
        return int(policy[3])

    def _is_runtime_error_tuple_capped(self, plan: StrategyPlan, state: JobState) -> bool:
        return self._runtime_error_tuple_key(plan, state) in self._runtime_signal_error_capped

    def _record_runtime_signal_failure(
        self,
        *,
        plan: StrategyPlan,
        state: JobState,
        full_params: dict[str, Any],
        exc: Exception,
    ) -> None:
        self._failure_record(
            self._strategy_failure_payload(
                state=state,
                stage="generate_signals",
                error=str(exc),
                strategy=plan.strategy.name,
                params=full_params,
            )
        )
        key = self._runtime_error_tuple_key(plan, state)
        count = self._runtime_signal_error_counts.get(key, 0) + 1
        self._runtime_signal_error_counts[key] = count
        threshold = self._runtime_signal_error_limit(state.job.collection)
        if threshold is None:
            return
        if count < threshold or key in self._runtime_signal_error_capped:
            return
        self._runtime_signal_error_capped.add(key)
        reason = "runtime_error_threshold_exceeded"
        self._plan_add_skip_reason(plan, reason)
        if not isinstance(plan.optimization_details, dict):
            plan.optimization_details = {}
        plan.optimization_details["runtime_error_threshold"] = {
            "strategy": plan.strategy.name,
            "symbol": state.job.symbol,
            "timeframe": state.job.timeframe,
            "count": count,
            "max_per_tuple": threshold,
        }
        self._failure_record(
            {
                **self._job_log_context(state.job),
                "strategy": plan.strategy.name,
                "stage": "strategy_optimization",
                "error": (
                    f"{reason}(count={count}, max_per_tuple={threshold})"
                ),
            }
        )

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

    def _strategy_validate_results(
        self,
        state: JobState,
        outcome: StrategyEvalOutcome,
        plan: StrategyPlan,
        validated_data: ValidatedData,
        prepared: ExecutionPreparedData,
    ) -> GateDecision:
        context = ValidationContext(
            stage="strategy_validation",
            state=state,
            mode=self.mode_config.mode,
            job=state.job,
            validated_data=validated_data,
            prepared_data=prepared,
            plan=plan,
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
        precheck_decision = self._strategy_validation_precheck(context)
        if precheck_decision is not None:
            return precheck_decision
        outcome = context.outcome
        plan = context.plan

        reasons = self._collect_strategy_validation_reasons(context, outcome)
        if reasons:
            # Fail fast on cheap result-consistency gates before expensive shuffle checks.
            return self._strategy_validation_reject_or_continue(reasons)
        self._run_lookahead_shuffle_validation(context, plan, outcome, reasons)
        self._run_transaction_cost_robustness_validation(context, plan, outcome, reasons)
        return self._strategy_validation_reject_or_continue(reasons)

    @staticmethod
    def _strategy_validation_precheck(context: ValidationContext) -> GateDecision | None:
        outcome = context.outcome
        if outcome is None:
            return GateDecision(False, "reject_result", ["missing_strategy_outcome"], "strategy_validation")
        if context.plan is None or context.validated_data is None:
            return GateDecision(False, "reject_result", ["missing_strategy_plan_context"], "strategy_validation")
        if not outcome.has_valid_candidate:
            return GateDecision(False, "reject_result", ["no_valid_candidate"], "strategy_validation")
        return None

    def _run_lookahead_shuffle_validation(
        self,
        context: ValidationContext,
        plan: StrategyPlan,
        outcome: StrategyEvalOutcome,
        reasons: list[str],
    ) -> None:
        lookahead_policy = self._load_lookahead_shuffle_test_policy(context.job.collection)
        if lookahead_policy is None:
            return
        lookahead_reason, lookahead_meta = self._lookahead_shuffle_test_result(
            context,
            plan,
            lookahead_policy,
            observed_metric=float(outcome.best_val) if np.isfinite(outcome.best_val) else None,
            params=outcome.best_params if isinstance(outcome.best_params, dict) else None,
        )
        self._attach_post_run_meta(outcome, "lookahead_shuffle_test", lookahead_meta)
        if lookahead_reason is not None:
            reasons.append(lookahead_reason)

    def _run_transaction_cost_robustness_validation(
        self,
        context: ValidationContext,
        plan: StrategyPlan,
        outcome: StrategyEvalOutcome,
        reasons: list[str],
    ) -> None:
        policy = self._load_transaction_cost_robustness_policy(context.job.collection)
        if policy is None or context.prepared_data is None or context.validated_data is None:
            return
        if not isinstance(outcome.best_params, dict):
            return
        baseline_profit = (
            self._safe_float_stat(outcome.best_stats, "profit")
            if isinstance(outcome.best_stats, dict)
            else None
        )
        run_ctx = TransactionCostRobustnessRunContext(
            context=context,
            plan=plan,
            policy=policy,
            prepared=context.prepared_data,
            full_params={**plan.fixed_params, **outcome.best_params},
            baseline_metric=float(outcome.best_val) if np.isfinite(outcome.best_val) else None,
            baseline_profit=baseline_profit,
        )
        transaction_cost_reason, transaction_cost_meta = self._transaction_cost_robustness_result(
            run_ctx
        )
        self._attach_post_run_meta(
            outcome,
            "transaction_cost_robustness",
            transaction_cost_meta,
        )
        if policy.mode == "enforce" and transaction_cost_reason is not None:
            reasons.append(transaction_cost_reason)

    @staticmethod
    def _strategy_validation_reject_or_continue(reasons: list[str]) -> GateDecision:
        if reasons:
            return GateDecision(False, "reject_result", reasons, "strategy_validation")
        return GateDecision(True, "continue", [], "strategy_validation")

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
        min_metric_reason = self._min_metric_reason(outcome.best_val, policy.min_metric)
        if min_metric_reason is not None:
            reasons.append(min_metric_reason)
        min_trades_reason = self._min_trades_reason(outcome.best_stats, policy.min_trades)
        if min_trades_reason is not None:
            reasons.append(min_trades_reason)
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
    def _min_metric_reason(
        metric_value: float,
        min_metric: float | None,
    ) -> str | None:
        if min_metric is None:
            return None
        if not np.isfinite(metric_value):
            return None
        observed = float(metric_value)
        required = float(min_metric)
        if observed < required:
            return f"min_metric_not_met(required={required}, available={observed})"
        return None

    @staticmethod
    def _min_trades_reason(
        stats: dict[str, Any],
        min_trades: int | None,
    ) -> str | None:
        if min_trades is None:
            return None
        trades_raw = stats.get("trades")
        if trades_raw is None:
            return None
        try:
            available = int(trades_raw)
        except (TypeError, ValueError):
            return None
        required = int(min_trades)
        if available < required:
            return f"min_trades_not_met(required={required}, available={available})"
        return None

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
        self._runtime_signal_error_counts = {}
        self._runtime_signal_error_capped = set()
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
        validation_hash_by_collection: dict[int, str] = {}
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

            # Compute once per collection: hash keys evaluation-cache correctness
            # from the effective collection-level validation profile.
            validation_hash = validation_hash_by_collection.get(collection_key)
            if validation_hash is None:
                validation_hash = self._hash_validation_profile(
                    self._build_job_validation_profile(state.job.collection)
                )
                validation_hash_by_collection[collection_key] = validation_hash
            state.validation_config_hash = validation_hash
            for strat_name in self.external_index.keys():
                self.metrics["symbols_tested"] += 1
                # Strategy stage: create plan -> validate plan -> run -> validate results.
                plan = self._strategy_create_plan(state, strat_name)
                self._apply_policy_constraints_to_plan(state, plan)
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
                raw_validation_decision = self._strategy_validate_results(
                    state, outcome, plan, validated_data, prepared
                )
                validation_context_extra: dict[str, Any] = {"strategy": outcome.strategy}
                handled_validation_decision = self._handle_gate_decision(
                    state,
                    raw_validation_decision,
                    context_extra=validation_context_extra,
                    blocked_collections=blocked_collections,
                )
                if not handled_validation_decision.passed:
                    if handled_validation_decision.action in {"skip_job", "skip_collection"}:
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
