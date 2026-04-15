from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml


@dataclass
class StrategyConfig:
    name: str
    module: str | None  # optional if scanning external subclasses
    cls: str | None
    params: dict[str, list[Any]]


@dataclass
class CollectionConfig:
    name: str
    source: str  # yfinance, ccxt, custom
    symbols: list[str]
    reference_source: str | None = None
    exchange: str | None = None  # for ccxt
    currency: str | None = None
    quote: str | None = None  # for ccxt symbols e.g., USDT
    fees: float | None = None
    slippage: float | None = None
    validation: "ValidationConfig | None" = None


@dataclass
class SlackNotificationConfig:
    webhook_url: str
    metric: str = "sharpe"
    threshold: float | None = None
    channel: str | None = None
    username: str | None = None


@dataclass
class NotificationsConfig:
    slack: SlackNotificationConfig | None = None


@dataclass
class ValidationCalendarConfig:
    kind: str | None = None
    exchange: str | None = None
    timezone: str | None = None


@dataclass
class ValidationDataQualityConfig:
    min_data_points: int | None = None
    calendar: ValidationCalendarConfig | None = None
    continuity: "ValidationContinuityConfig | None" = None
    kurtosis: float | None = None
    ohlc_integrity: "ValidationOHLCIntegrityConfig | None" = None
    outlier_detection: "ValidationOutlierDetectionConfig | None" = None
    stationarity: "ValidationStationarityConfig | None" = None
    is_verified: bool | None = None
    on_fail: str | None = None


@dataclass
class ValidationContinuityConfig:
    min_score: float | None = None
    max_missing_bar_pct: float | None = None


@dataclass
class ValidationOHLCIntegrityConfig:
    max_invalid_bar_pct: float | None = None
    allow_negative_price: bool | None = None
    allow_negative_volume: bool | None = None


@dataclass
class ValidationOutlierDetectionConfig:
    max_outlier_pct: float
    method: str
    zscore_threshold: float


@dataclass
class ValidationStationarityRegimeShiftConfig:
    window: int
    mean_shift_max: float
    vol_ratio_max: float


@dataclass
class ValidationStationarityConfig:
    adf_pvalue_max: float
    kpss_pvalue_min: float | None = None
    min_points: int | None = None
    regime_shift: ValidationStationarityRegimeShiftConfig | None = None


@dataclass
class ValidationLookaheadShuffleTestConfig:
    permutations: int | None = None
    pvalue_max: float | None = None
    seed: int | None = None
    max_failed_permutations: int | None = None


@dataclass
class ValidationConfig:
    data_quality: ValidationDataQualityConfig | None = None
    optimization: "OptimizationPolicyConfig | None" = None
    result_consistency: "ResultConsistencyConfig | None" = None


@dataclass
class OptimizationPolicyConfig:
    on_fail: str
    min_bars: int
    dof_multiplier: int
    runtime_error_max_per_tuple: int | None = None


@dataclass
class ResultConsistencyOutlierDependencyConfig:
    slices: int
    profit_share_threshold: float
    trade_share_threshold: float


@dataclass
class ResultConsistencyExecutionPriceVarianceConfig:
    price_tolerance_bps: float


@dataclass
class ResultConsistencyTransactionCostBreakevenConfig:
    enabled: bool | None = None
    min_multiplier: float | None = None
    max_multiplier: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass
class ResultConsistencyTransactionCostRobustnessConfig:
    mode: Literal["analytics", "enforce"] | None = None
    stress_multipliers: list[float] | None = None
    max_metric_drop_pct: float | None = None
    breakeven: ResultConsistencyTransactionCostBreakevenConfig | None = None


@dataclass
class ResultConsistencyDataIntegrityAuditConfig:
    min_overlap_ratio: float | None = None
    max_median_ohlc_diff_bps: float | None = None
    max_p95_ohlc_diff_bps: float | None = None


@dataclass
class ResultConsistencyConfig:
    min_metric: float | None = None
    min_trades: int | None = None
    outlier_dependency: ResultConsistencyOutlierDependencyConfig | None = None
    execution_price_variance: ResultConsistencyExecutionPriceVarianceConfig | None = None
    lookahead_shuffle_test: ValidationLookaheadShuffleTestConfig | None = None
    transaction_cost_robustness: ResultConsistencyTransactionCostRobustnessConfig | None = None
    data_integrity_audit: ResultConsistencyDataIntegrityAuditConfig | None = None


@dataclass
class Config:
    collections: list[CollectionConfig]
    timeframes: list[str]
    metric: str  # sharpe | sortino | profit
    strategies: list[StrategyConfig]
    engine: str = "pybroker"  # pybroker engine
    param_search: str = "grid"  # grid | optuna
    param_trials: int = 25
    max_workers: int = 1
    asset_workers: int = 1
    param_workers: int = 1
    max_fetch_concurrency: int = 2
    fees: float = 0.0
    slippage: float = 0.0
    risk_free_rate: float = 0.0
    cache_dir: str = ".cache/data"
    evaluation_mode: str = "backtest"
    notifications: NotificationsConfig | None = None
    validation: ValidationConfig | None = None


CALENDAR_KIND_DEFAULT = "auto"
STATIONARITY_MIN_POINTS_DEFAULT = 30
STATIONARITY_MIN_POINTS_MIN = 20
STATIONARITY_REGIME_SHIFT_WINDOW_MIN = 10
STATIONARITY_REGIME_SHIFT_MEAN_SHIFT_MIN = 0.0
STATIONARITY_REGIME_SHIFT_VOL_RATIO_MIN = 1.0
LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_DEFAULT = 100
LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_MIN = 100
LOOKAHEAD_SHUFFLE_TEST_SEED_DEFAULT = 1337
LOOKAHEAD_SHUFFLE_TEST_SEED_MIN = 0
LOOKAHEAD_SHUFFLE_TEST_FAILED_PERMUTATIONS_MIN = 0
LOOKAHEAD_SHUFFLE_TEST_CONFIG_PREFIX = "validation.result_consistency.lookahead_shuffle_test"
DATA_INTEGRITY_AUDIT_CONFIG_PREFIX = "validation.result_consistency.data_integrity_audit"
DATA_INTEGRITY_AUDIT_MIN_OVERLAP_RATIO_DEFAULT = 0.99
DATA_INTEGRITY_AUDIT_MAX_MEDIAN_OHLC_DIFF_BPS_DEFAULT = 5.0
DATA_INTEGRITY_AUDIT_MAX_P95_OHLC_DIFF_BPS_DEFAULT = 20.0
TRANSACTION_COST_ROBUSTNESS_MODE_ANALYTICS = "analytics"
TRANSACTION_COST_ROBUSTNESS_MODE_ENFORCE = "enforce"
TRANSACTION_COST_ROBUSTNESS_MODES = {
    TRANSACTION_COST_ROBUSTNESS_MODE_ANALYTICS,
    TRANSACTION_COST_ROBUSTNESS_MODE_ENFORCE,
}
TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN = 1.0
TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN = 0.0
TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX = 1.0
TRANSACTION_COST_ROBUSTNESS_MAX_ITERATIONS_MIN = 1
TRANSACTION_COST_ROBUSTNESS_TOLERANCE_MIN = 0.0
VALIDATION_PROBABILITY_MIN = 0.0
VALIDATION_PROBABILITY_MAX = 1.0
VALIDATION_PERCENT_MIN = 0.0
VALIDATION_PERCENT_MAX = 100.0
VALIDATION_NON_NEGATIVE_INT_MIN = 0
VALIDATION_NON_NEGATIVE_FLOAT_MIN = 0.0
OPTIMIZATION_RUNTIME_ERROR_MAX_PER_TUPLE_MIN = 1
RESULT_CONSISTENCY_OUTLIER_DEPENDENCY_SLICES_MIN = 2
RESULT_CONSISTENCY_MIN_TRADES_MIN = 1
OUTLIER_DETECTION_ZSCORE_THRESHOLD_MIN_EXCLUSIVE = 0.0
OHLC_INTEGRITY_INVALID_BAR_PCT_DEFAULT = 0.0


def _merged_field(base: Any, override: Any, field: str) -> Any:
    override_value = getattr(override, field, None)
    if override_value is not None:
        return override_value
    return getattr(base, field, None)


def _require_normalized(value: Any, prefix: str) -> Any:
    if value is None:
        raise RuntimeError(f"Internal error: normalization returned None for {prefix}")
    return value


def _apply_calendar_defaults(cfg: ValidationCalendarConfig) -> ValidationCalendarConfig:
    kind = cfg.kind if cfg.kind is not None else CALENDAR_KIND_DEFAULT
    return ValidationCalendarConfig(
        kind=kind,
        exchange=cfg.exchange,
        timezone=cfg.timezone,
    )


def _normalize_calendar_config(
    cfg: ValidationCalendarConfig | None,
    prefix: str,
) -> ValidationCalendarConfig | None:
    if cfg is None:
        return None
    kind_raw = getattr(cfg, "kind", None)
    kind = str(kind_raw).strip().lower() if kind_raw is not None else None
    allowed_kinds = {"auto", "crypto_24_7", "weekday", "exchange"}
    if kind is not None and kind not in allowed_kinds:
        raise ValueError(
            f"Invalid `{prefix}.kind`: expected one of {sorted(allowed_kinds)}, got '{kind}'"
        )
    exchange_raw = getattr(cfg, "exchange", None)
    exchange = str(exchange_raw).strip() if exchange_raw is not None else None
    timezone = _parse_utc_timezone(getattr(cfg, "timezone", None), f"{prefix}.timezone")
    return ValidationCalendarConfig(
        kind=kind,
        exchange=exchange,
        timezone=timezone,
    )


def _normalize_optimization_config(
    cfg: OptimizationPolicyConfig | None,
    prefix: str,
) -> OptimizationPolicyConfig | None:
    if cfg is None:
        return None
    on_fail = _parse_on_fail(getattr(cfg, "on_fail", None), f"{prefix}.on_fail", {"baseline_only", "skip_job"})
    if on_fail is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): on_fail")
    min_bars = getattr(cfg, "min_bars", None)
    if min_bars is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): min_bars")
    min_bars = _coerce_int(min_bars, f"{prefix}.min_bars")
    if min_bars < VALIDATION_NON_NEGATIVE_INT_MIN:
        raise ValueError(f"`{prefix}.min_bars` must be >= {VALIDATION_NON_NEGATIVE_INT_MIN}")
    dof_multiplier = getattr(cfg, "dof_multiplier", None)
    if dof_multiplier is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): dof_multiplier")
    dof_multiplier = _coerce_int(dof_multiplier, f"{prefix}.dof_multiplier")
    if dof_multiplier < VALIDATION_NON_NEGATIVE_INT_MIN:
        raise ValueError(f"`{prefix}.dof_multiplier` must be >= {VALIDATION_NON_NEGATIVE_INT_MIN}")
    runtime_error_max_per_tuple = getattr(cfg, "runtime_error_max_per_tuple", None)
    if runtime_error_max_per_tuple is not None:
        runtime_error_max_per_tuple = _coerce_int(
            runtime_error_max_per_tuple,
            f"{prefix}.runtime_error_max_per_tuple",
        )
        if runtime_error_max_per_tuple < OPTIMIZATION_RUNTIME_ERROR_MAX_PER_TUPLE_MIN:
            raise ValueError(
                f"`{prefix}.runtime_error_max_per_tuple` must be >= {OPTIMIZATION_RUNTIME_ERROR_MAX_PER_TUPLE_MIN}"
            )
    return OptimizationPolicyConfig(
        on_fail=on_fail,
        min_bars=min_bars,
        dof_multiplier=dof_multiplier,
        runtime_error_max_per_tuple=runtime_error_max_per_tuple,
    )


def _apply_optimization_defaults(cfg: OptimizationPolicyConfig) -> OptimizationPolicyConfig:
    runtime_error_max_per_tuple = getattr(cfg, "runtime_error_max_per_tuple", None)
    if runtime_error_max_per_tuple is None:
        runtime_error_max_per_tuple = OPTIMIZATION_RUNTIME_ERROR_MAX_PER_TUPLE_MIN
    return OptimizationPolicyConfig(
        on_fail=cfg.on_fail,
        min_bars=cfg.min_bars,
        dof_multiplier=cfg.dof_multiplier,
        runtime_error_max_per_tuple=_coerce_int(
            runtime_error_max_per_tuple,
            "validation.optimization.runtime_error_max_per_tuple",
        ),
    )


def _normalize_data_quality_config(
    cfg: ValidationDataQualityConfig | None,
    prefix: str,
) -> ValidationDataQualityConfig | None:
    if cfg is None:
        return None
    on_fail = _parse_on_fail(
        getattr(cfg, "on_fail", None),
        f"{prefix}.on_fail",
        {"skip_optimization", "skip_job", "skip_collection"},
    )
    if on_fail is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): on_fail")
    min_data_points = getattr(cfg, "min_data_points", None)
    if min_data_points is not None:
        min_data_points = _coerce_int(min_data_points, f"{prefix}.min_data_points")
        if min_data_points < VALIDATION_NON_NEGATIVE_INT_MIN:
            raise ValueError(f"`{prefix}.min_data_points` must be >= {VALIDATION_NON_NEGATIVE_INT_MIN}")
    kurtosis = getattr(cfg, "kurtosis", None)
    if kurtosis is not None:
        kurtosis = _coerce_float(kurtosis, f"{prefix}.kurtosis")
        if kurtosis < VALIDATION_NON_NEGATIVE_FLOAT_MIN:
            raise ValueError(f"`{prefix}.kurtosis` must be >= {VALIDATION_NON_NEGATIVE_FLOAT_MIN}")
    continuity = _normalize_continuity_config(
        getattr(cfg, "continuity", None),
        f"{prefix}.continuity",
    )
    calendar = _normalize_calendar_config(getattr(cfg, "calendar", None), f"{prefix}.calendar")
    return ValidationDataQualityConfig(
        min_data_points=min_data_points,
        calendar=calendar,
        continuity=continuity,
        kurtosis=kurtosis,
        ohlc_integrity=_normalize_ohlc_integrity_config(
            getattr(cfg, "ohlc_integrity", None),
            f"{prefix}.ohlc_integrity",
        ),
        outlier_detection=_normalize_outlier_detection_config(
            getattr(cfg, "outlier_detection", None),
            f"{prefix}.outlier_detection",
        ),
        stationarity=_normalize_stationarity_config(
            getattr(cfg, "stationarity", None),
            f"{prefix}.stationarity",
        ),
        is_verified=getattr(cfg, "is_verified", None),
        on_fail=on_fail,
    )


def _apply_data_quality_defaults(cfg: ValidationDataQualityConfig) -> ValidationDataQualityConfig:
    return ValidationDataQualityConfig(
        min_data_points=cfg.min_data_points,
        calendar=_apply_calendar_defaults(cfg.calendar) if cfg.calendar is not None else None,
        continuity=_apply_continuity_defaults(cfg.continuity) if cfg.continuity is not None else None,
        kurtosis=cfg.kurtosis,
        ohlc_integrity=(
            _apply_ohlc_integrity_defaults(cfg.ohlc_integrity)
            if cfg.ohlc_integrity is not None
            else None
        ),
        outlier_detection=(
            _apply_outlier_detection_defaults(cfg.outlier_detection)
            if cfg.outlier_detection is not None
            else None
        ),
        stationarity=(
            _apply_stationarity_defaults(cfg.stationarity) if cfg.stationarity is not None else None
        ),
        is_verified=cfg.is_verified,
        on_fail=cfg.on_fail,
    )


def _normalize_continuity_config(
    cfg: ValidationContinuityConfig | None,
    prefix: str,
) -> ValidationContinuityConfig | None:
    if cfg is None:
        return None
    min_score = getattr(cfg, "min_score", None)
    if min_score is not None:
        min_score = _coerce_float(min_score, f"{prefix}.min_score")
        if min_score < VALIDATION_PROBABILITY_MIN or min_score > VALIDATION_PROBABILITY_MAX:
            raise ValueError(
                f"`{prefix}.min_score` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
            )
    max_missing_bar_pct = getattr(cfg, "max_missing_bar_pct", None)
    if max_missing_bar_pct is not None:
        max_missing_bar_pct = _coerce_float(max_missing_bar_pct, f"{prefix}.max_missing_bar_pct")
        if max_missing_bar_pct < VALIDATION_PERCENT_MIN or max_missing_bar_pct > VALIDATION_PERCENT_MAX:
            raise ValueError(
                f"`{prefix}.max_missing_bar_pct` must be between {VALIDATION_PERCENT_MIN} and {VALIDATION_PERCENT_MAX}"
            )
    return ValidationContinuityConfig(
        min_score=min_score,
        max_missing_bar_pct=max_missing_bar_pct,
    )


def _apply_continuity_defaults(cfg: ValidationContinuityConfig) -> ValidationContinuityConfig:
    return ValidationContinuityConfig(
        min_score=cfg.min_score,
        max_missing_bar_pct=cfg.max_missing_bar_pct,
    )


def _normalize_ohlc_integrity_config(
    cfg: ValidationOHLCIntegrityConfig | None,
    prefix: str,
) -> ValidationOHLCIntegrityConfig | None:
    if cfg is None:
        return None
    max_invalid_bar_pct = getattr(cfg, "max_invalid_bar_pct", None)
    if max_invalid_bar_pct is not None:
        max_invalid_bar_pct = _coerce_float(max_invalid_bar_pct, f"{prefix}.max_invalid_bar_pct")
        if max_invalid_bar_pct < VALIDATION_PERCENT_MIN or max_invalid_bar_pct > VALIDATION_PERCENT_MAX:
            raise ValueError(
                f"`{prefix}.max_invalid_bar_pct` must be between {VALIDATION_PERCENT_MIN} and {VALIDATION_PERCENT_MAX}"
            )
    allow_negative_price = getattr(cfg, "allow_negative_price", None)
    if allow_negative_price is not None and not isinstance(allow_negative_price, bool):
        raise ValueError(f"`{prefix}.allow_negative_price` must be a boolean or null")
    allow_negative_volume = getattr(cfg, "allow_negative_volume", None)
    if allow_negative_volume is not None and not isinstance(allow_negative_volume, bool):
        raise ValueError(f"`{prefix}.allow_negative_volume` must be a boolean or null")
    return ValidationOHLCIntegrityConfig(
        max_invalid_bar_pct=max_invalid_bar_pct,
        allow_negative_price=allow_negative_price,
        allow_negative_volume=allow_negative_volume,
    )


def _apply_ohlc_integrity_defaults(
    cfg: ValidationOHLCIntegrityConfig,
) -> ValidationOHLCIntegrityConfig:
    max_invalid_bar_pct = (
        float(cfg.max_invalid_bar_pct)
        if cfg.max_invalid_bar_pct is not None
        else OHLC_INTEGRITY_INVALID_BAR_PCT_DEFAULT
    )
    allow_negative_price = cfg.allow_negative_price if cfg.allow_negative_price is not None else False
    allow_negative_volume = cfg.allow_negative_volume if cfg.allow_negative_volume is not None else False
    return ValidationOHLCIntegrityConfig(
        max_invalid_bar_pct=max_invalid_bar_pct,
        allow_negative_price=allow_negative_price,
        allow_negative_volume=allow_negative_volume,
    )


def _normalize_outlier_detection_config(
    cfg: ValidationOutlierDetectionConfig | None,
    prefix: str,
) -> ValidationOutlierDetectionConfig | None:
    if cfg is None:
        return None
    max_outlier_pct = getattr(cfg, "max_outlier_pct", None)
    if max_outlier_pct is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): max_outlier_pct")
    max_outlier_pct = _coerce_float(max_outlier_pct, f"{prefix}.max_outlier_pct")
    if max_outlier_pct < VALIDATION_PERCENT_MIN or max_outlier_pct > VALIDATION_PERCENT_MAX:
        raise ValueError(
            f"`{prefix}.max_outlier_pct` must be between {VALIDATION_PERCENT_MIN} and {VALIDATION_PERCENT_MAX}"
        )
    method_raw = getattr(cfg, "method", None)
    method = str(method_raw).strip().lower() if method_raw is not None else None
    if method is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): method")
    if method not in {"zscore", "modified_zscore"}:
        raise ValueError(
            f"Invalid `{prefix}.method`: expected one of ['modified_zscore', 'zscore']"
        )
    zscore_threshold = getattr(cfg, "zscore_threshold", None)
    if zscore_threshold is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): zscore_threshold")
    zscore_threshold = _coerce_float(zscore_threshold, f"{prefix}.zscore_threshold")
    if zscore_threshold <= OUTLIER_DETECTION_ZSCORE_THRESHOLD_MIN_EXCLUSIVE:
        raise ValueError(
            f"`{prefix}.zscore_threshold` must be > {OUTLIER_DETECTION_ZSCORE_THRESHOLD_MIN_EXCLUSIVE}"
        )
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=max_outlier_pct,
        method=method,
        zscore_threshold=zscore_threshold,
    )


def _apply_outlier_detection_defaults(
    cfg: ValidationOutlierDetectionConfig,
) -> ValidationOutlierDetectionConfig:
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=cfg.max_outlier_pct,
        method=cfg.method,
        zscore_threshold=cfg.zscore_threshold,
    )


def _normalize_result_consistency_outlier_dependency_config(
    cfg: ResultConsistencyOutlierDependencyConfig | None,
    prefix: str,
) -> ResultConsistencyOutlierDependencyConfig | None:
    if cfg is None:
        return None
    slices = getattr(cfg, "slices", None)
    if slices is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): slices")
    slices = _coerce_int(slices, f"{prefix}.slices")
    if slices < RESULT_CONSISTENCY_OUTLIER_DEPENDENCY_SLICES_MIN:
        raise ValueError(f"`{prefix}.slices` must be >= {RESULT_CONSISTENCY_OUTLIER_DEPENDENCY_SLICES_MIN}")
    profit_share_threshold = getattr(cfg, "profit_share_threshold", None)
    if profit_share_threshold is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): profit_share_threshold")
    profit_share_threshold = _coerce_float(profit_share_threshold, f"{prefix}.profit_share_threshold")
    if (
        profit_share_threshold < VALIDATION_PROBABILITY_MIN
        or profit_share_threshold > VALIDATION_PROBABILITY_MAX
    ):
        raise ValueError(
            f"`{prefix}.profit_share_threshold` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
        )
    trade_share_threshold = getattr(cfg, "trade_share_threshold", None)
    if trade_share_threshold is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): trade_share_threshold")
    trade_share_threshold = _coerce_float(trade_share_threshold, f"{prefix}.trade_share_threshold")
    if (
        trade_share_threshold < VALIDATION_PROBABILITY_MIN
        or trade_share_threshold > VALIDATION_PROBABILITY_MAX
    ):
        raise ValueError(
            f"`{prefix}.trade_share_threshold` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
        )
    return ResultConsistencyOutlierDependencyConfig(
        slices=slices,
        profit_share_threshold=profit_share_threshold,
        trade_share_threshold=trade_share_threshold,
    )


def _apply_result_consistency_outlier_dependency_defaults(
    cfg: ResultConsistencyOutlierDependencyConfig,
) -> ResultConsistencyOutlierDependencyConfig:
    return ResultConsistencyOutlierDependencyConfig(
        slices=cfg.slices,
        profit_share_threshold=cfg.profit_share_threshold,
        trade_share_threshold=cfg.trade_share_threshold,
    )


def _normalize_result_consistency_execution_price_variance_config(
    cfg: ResultConsistencyExecutionPriceVarianceConfig | None,
    prefix: str,
) -> ResultConsistencyExecutionPriceVarianceConfig | None:
    if cfg is None:
        return None
    price_tolerance_bps = getattr(cfg, "price_tolerance_bps", None)
    if price_tolerance_bps is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): price_tolerance_bps")
    price_tolerance_bps = _coerce_float(price_tolerance_bps, f"{prefix}.price_tolerance_bps")
    if price_tolerance_bps < VALIDATION_NON_NEGATIVE_FLOAT_MIN:
        raise ValueError(f"`{prefix}.price_tolerance_bps` must be >= {VALIDATION_NON_NEGATIVE_FLOAT_MIN}")
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=price_tolerance_bps,
    )


def _apply_result_consistency_execution_price_variance_defaults(
    cfg: ResultConsistencyExecutionPriceVarianceConfig,
) -> ResultConsistencyExecutionPriceVarianceConfig:
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=cfg.price_tolerance_bps,
    )


def _normalize_result_consistency_data_integrity_audit_config(
    cfg: ResultConsistencyDataIntegrityAuditConfig | None,
    prefix: str,
) -> ResultConsistencyDataIntegrityAuditConfig | None:
    if cfg is None:
        return None
    min_overlap_ratio_raw = getattr(cfg, "min_overlap_ratio", None)
    min_overlap_ratio = (
        _coerce_float(min_overlap_ratio_raw, f"{prefix}.min_overlap_ratio")
        if min_overlap_ratio_raw is not None
        else None
    )
    if min_overlap_ratio is not None and not (
        VALIDATION_PROBABILITY_MIN <= min_overlap_ratio <= VALIDATION_PROBABILITY_MAX
    ):
        raise ValueError(
            f"`{prefix}.min_overlap_ratio` must be between {VALIDATION_PROBABILITY_MIN} and "
            f"{VALIDATION_PROBABILITY_MAX}"
        )
    max_median_ohlc_diff_bps_raw = getattr(cfg, "max_median_ohlc_diff_bps", None)
    max_median_ohlc_diff_bps = (
        _coerce_float(max_median_ohlc_diff_bps_raw, f"{prefix}.max_median_ohlc_diff_bps")
        if max_median_ohlc_diff_bps_raw is not None
        else None
    )
    if (
        max_median_ohlc_diff_bps is not None
        and max_median_ohlc_diff_bps < VALIDATION_NON_NEGATIVE_FLOAT_MIN
    ):
        raise ValueError(
            f"`{prefix}.max_median_ohlc_diff_bps` must be >= {VALIDATION_NON_NEGATIVE_FLOAT_MIN}"
        )
    max_p95_ohlc_diff_bps_raw = getattr(cfg, "max_p95_ohlc_diff_bps", None)
    max_p95_ohlc_diff_bps = (
        _coerce_float(max_p95_ohlc_diff_bps_raw, f"{prefix}.max_p95_ohlc_diff_bps")
        if max_p95_ohlc_diff_bps_raw is not None
        else None
    )
    if max_p95_ohlc_diff_bps is not None and max_p95_ohlc_diff_bps < VALIDATION_NON_NEGATIVE_FLOAT_MIN:
        raise ValueError(
            f"`{prefix}.max_p95_ohlc_diff_bps` must be >= {VALIDATION_NON_NEGATIVE_FLOAT_MIN}"
        )
    if (
        max_median_ohlc_diff_bps is not None
        and max_p95_ohlc_diff_bps is not None
        and max_p95_ohlc_diff_bps < max_median_ohlc_diff_bps
    ):
        raise ValueError(
            f"`{prefix}.max_p95_ohlc_diff_bps` must be >= `{prefix}.max_median_ohlc_diff_bps`"
        )
    return ResultConsistencyDataIntegrityAuditConfig(
        min_overlap_ratio=min_overlap_ratio,
        max_median_ohlc_diff_bps=max_median_ohlc_diff_bps,
        max_p95_ohlc_diff_bps=max_p95_ohlc_diff_bps,
    )


def _apply_result_consistency_data_integrity_audit_defaults(
    cfg: ResultConsistencyDataIntegrityAuditConfig,
) -> ResultConsistencyDataIntegrityAuditConfig:
    min_overlap_ratio = (
        cfg.min_overlap_ratio
        if cfg.min_overlap_ratio is not None
        else DATA_INTEGRITY_AUDIT_MIN_OVERLAP_RATIO_DEFAULT
    )
    max_median_ohlc_diff_bps = (
        cfg.max_median_ohlc_diff_bps
        if cfg.max_median_ohlc_diff_bps is not None
        else DATA_INTEGRITY_AUDIT_MAX_MEDIAN_OHLC_DIFF_BPS_DEFAULT
    )
    max_p95_ohlc_diff_bps = (
        cfg.max_p95_ohlc_diff_bps
        if cfg.max_p95_ohlc_diff_bps is not None
        else DATA_INTEGRITY_AUDIT_MAX_P95_OHLC_DIFF_BPS_DEFAULT
    )
    if max_p95_ohlc_diff_bps < max_median_ohlc_diff_bps:
        raise ValueError(
            f"`{DATA_INTEGRITY_AUDIT_CONFIG_PREFIX}.max_p95_ohlc_diff_bps` must be >= "
            f"`{DATA_INTEGRITY_AUDIT_CONFIG_PREFIX}.max_median_ohlc_diff_bps`"
        )
    return ResultConsistencyDataIntegrityAuditConfig(
        min_overlap_ratio=min_overlap_ratio,
        max_median_ohlc_diff_bps=max_median_ohlc_diff_bps,
        max_p95_ohlc_diff_bps=max_p95_ohlc_diff_bps,
    )


def _default_data_integrity_audit_config() -> ResultConsistencyDataIntegrityAuditConfig:
    return _apply_result_consistency_data_integrity_audit_defaults(
        ResultConsistencyDataIntegrityAuditConfig()
    )


def _normalize_transaction_cost_breakeven_config(
    cfg: ResultConsistencyTransactionCostBreakevenConfig | None,
    prefix: str,
) -> ResultConsistencyTransactionCostBreakevenConfig | None:
    if cfg is None:
        return None
    enabled = _normalize_transaction_cost_breakeven_enabled(getattr(cfg, "enabled", None), prefix)
    min_multiplier = _normalize_transaction_cost_breakeven_multiplier(
        getattr(cfg, "min_multiplier", None),
        f"{prefix}.min_multiplier",
    )
    max_multiplier = _normalize_transaction_cost_breakeven_multiplier(
        getattr(cfg, "max_multiplier", None),
        f"{prefix}.max_multiplier",
    )
    max_iterations = _normalize_transaction_cost_breakeven_iterations(
        getattr(cfg, "max_iterations", None),
        prefix,
    )
    tolerance = _normalize_transaction_cost_breakeven_tolerance(
        getattr(cfg, "tolerance", None),
        prefix,
    )
    _validate_transaction_cost_breakeven_multiplier_range(
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        prefix=prefix,
    )
    return ResultConsistencyTransactionCostBreakevenConfig(
        enabled=enabled,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _normalize_transaction_cost_breakeven_enabled(
    enabled_raw: Any,
    prefix: str,
) -> bool | None:
    if enabled_raw is not None and not isinstance(enabled_raw, bool):
        raise ValueError(f"Invalid `{prefix}.enabled`: expected a boolean")
    return enabled_raw


def _normalize_transaction_cost_breakeven_multiplier(
    multiplier_raw: Any,
    field_path: str,
) -> float | None:
    if multiplier_raw is None:
        return None
    multiplier = _coerce_float(multiplier_raw, field_path)
    if multiplier < TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN:
        raise ValueError(
            f"`{field_path}` must be >= {TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN}"
        )
    return multiplier


def _normalize_transaction_cost_breakeven_iterations(
    max_iterations_raw: Any,
    prefix: str,
) -> int | None:
    if max_iterations_raw is None:
        return None
    max_iterations = _coerce_int(max_iterations_raw, f"{prefix}.max_iterations")
    if max_iterations < TRANSACTION_COST_ROBUSTNESS_MAX_ITERATIONS_MIN:
        raise ValueError(
            f"`{prefix}.max_iterations` must be >= {TRANSACTION_COST_ROBUSTNESS_MAX_ITERATIONS_MIN}"
        )
    return max_iterations


def _normalize_transaction_cost_breakeven_tolerance(
    tolerance_raw: Any,
    prefix: str,
) -> float | None:
    if tolerance_raw is None:
        return None
    tolerance = _coerce_float(tolerance_raw, f"{prefix}.tolerance")
    if tolerance <= TRANSACTION_COST_ROBUSTNESS_TOLERANCE_MIN:
        raise ValueError(
            f"`{prefix}.tolerance` must be > {TRANSACTION_COST_ROBUSTNESS_TOLERANCE_MIN}"
        )
    return tolerance


def _validate_transaction_cost_breakeven_multiplier_range(
    *,
    min_multiplier: float | None,
    max_multiplier: float | None,
    prefix: str,
) -> None:
    if min_multiplier is None or max_multiplier is None:
        return
    if max_multiplier < min_multiplier:
        raise ValueError(f"`{prefix}.max_multiplier` must be >= `{prefix}.min_multiplier`")


def _normalize_transaction_cost_mode(mode_raw: Any, prefix: str) -> str | None:
    mode = str(mode_raw).strip().lower() if mode_raw is not None else None
    if mode is not None and mode not in TRANSACTION_COST_ROBUSTNESS_MODES:
        raise ValueError(
            f"Invalid `{prefix}.mode`: expected one of {sorted(TRANSACTION_COST_ROBUSTNESS_MODES)}"
        )
    return mode


def _normalize_transaction_cost_stress_multipliers(
    stress_multipliers_raw: Any,
    prefix: str,
) -> list[float] | None:
    if stress_multipliers_raw is None:
        return None
    if not isinstance(stress_multipliers_raw, list):
        raise ValueError(f"Invalid `{prefix}.stress_multipliers`: expected a list")
    normalized_multipliers: list[float] = []
    previous: float | None = None
    for idx, value in enumerate(stress_multipliers_raw):
        multiplier = _coerce_float(value, f"{prefix}.stress_multipliers[{idx}]")
        if multiplier < TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN:
            raise ValueError(
                f"`{prefix}.stress_multipliers[{idx}]` must be >= {TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN}"
            )
        if previous is not None and multiplier <= previous:
            raise ValueError(f"`{prefix}.stress_multipliers` must be sorted in ascending order")
        normalized_multipliers.append(multiplier)
        previous = multiplier
    return normalized_multipliers


def _normalize_transaction_cost_max_metric_drop_pct(
    max_metric_drop_pct_raw: Any,
    prefix: str,
) -> float | None:
    if max_metric_drop_pct_raw is None:
        return None
    max_metric_drop_pct = _coerce_float(max_metric_drop_pct_raw, f"{prefix}.max_metric_drop_pct")
    if (
        max_metric_drop_pct < TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN
        or max_metric_drop_pct > TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX
    ):
        raise ValueError(
            f"`{prefix}.max_metric_drop_pct` must be between "
            f"{TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN} and "
            f"{TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX}"
        )
    return max_metric_drop_pct


def _apply_transaction_cost_breakeven_defaults(
    cfg: ResultConsistencyTransactionCostBreakevenConfig,
) -> ResultConsistencyTransactionCostBreakevenConfig:
    enabled = cfg.enabled
    if enabled is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "missing required field(s): enabled"
        )
    if cfg.min_multiplier is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "missing required field(s): min_multiplier"
        )
    if cfg.max_multiplier is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "missing required field(s): max_multiplier"
        )
    if cfg.max_iterations is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "missing required field(s): max_iterations"
        )
    if cfg.tolerance is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "missing required field(s): tolerance"
        )
    if cfg.min_multiplier < TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            f"min_multiplier must be >= {TRANSACTION_COST_ROBUSTNESS_MIN_MULTIPLIER_MIN}"
        )
    if cfg.max_multiplier < cfg.min_multiplier:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            "max_multiplier must be >= min_multiplier"
        )
    if cfg.max_iterations < TRANSACTION_COST_ROBUSTNESS_MAX_ITERATIONS_MIN:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            f"max_iterations must be >= {TRANSACTION_COST_ROBUSTNESS_MAX_ITERATIONS_MIN}"
        )
    if cfg.tolerance <= TRANSACTION_COST_ROBUSTNESS_TOLERANCE_MIN:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness.breakeven`: "
            f"tolerance must be > {TRANSACTION_COST_ROBUSTNESS_TOLERANCE_MIN}"
        )
    return ResultConsistencyTransactionCostBreakevenConfig(
        enabled=enabled,
        min_multiplier=_coerce_float(
            cfg.min_multiplier,
            "validation.result_consistency.transaction_cost_robustness.breakeven.min_multiplier",
        ),
        max_multiplier=_coerce_float(
            cfg.max_multiplier,
            "validation.result_consistency.transaction_cost_robustness.breakeven.max_multiplier",
        ),
        max_iterations=_coerce_int(
            cfg.max_iterations,
            "validation.result_consistency.transaction_cost_robustness.breakeven.max_iterations",
        ),
        tolerance=_coerce_float(
            cfg.tolerance,
            "validation.result_consistency.transaction_cost_robustness.breakeven.tolerance",
        ),
    )


def _normalize_transaction_cost_robustness_config(
    cfg: ResultConsistencyTransactionCostRobustnessConfig | None,
    prefix: str,
) -> ResultConsistencyTransactionCostRobustnessConfig | None:
    if cfg is None:
        return None
    mode = _normalize_transaction_cost_mode(getattr(cfg, "mode", None), prefix)
    stress_multipliers = _normalize_transaction_cost_stress_multipliers(
        getattr(cfg, "stress_multipliers", None),
        prefix,
    )
    max_metric_drop_pct = _normalize_transaction_cost_max_metric_drop_pct(
        getattr(cfg, "max_metric_drop_pct", None),
        prefix,
    )
    breakeven = _normalize_transaction_cost_breakeven_config(
        getattr(cfg, "breakeven", None),
        f"{prefix}.breakeven",
    )
    return ResultConsistencyTransactionCostRobustnessConfig(
        mode=mode,
        stress_multipliers=stress_multipliers,
        max_metric_drop_pct=max_metric_drop_pct,
        breakeven=breakeven,
    )


def _apply_transaction_cost_robustness_defaults(
    cfg: ResultConsistencyTransactionCostRobustnessConfig,
) -> ResultConsistencyTransactionCostRobustnessConfig:
    mode = cfg.mode
    if mode is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            "missing required field(s): mode"
        )
    if mode not in TRANSACTION_COST_ROBUSTNESS_MODES:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            f"expected mode to be one of {sorted(TRANSACTION_COST_ROBUSTNESS_MODES)}"
        )
    stress_multipliers = cfg.stress_multipliers
    if stress_multipliers is None or not stress_multipliers:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            "missing required field(s): stress_multipliers"
        )
    if cfg.max_metric_drop_pct is None:
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            "missing required field(s): max_metric_drop_pct"
        )
    if not math.isfinite(cfg.max_metric_drop_pct):
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            "`validation.result_consistency.transaction_cost_robustness.max_metric_drop_pct` must be finite"
        )
    if cfg.max_metric_drop_pct < TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN or (
        cfg.max_metric_drop_pct > TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX
    ):
        raise ValueError(
            "Invalid `validation.result_consistency.transaction_cost_robustness`: "
            f"max_metric_drop_pct must be between "
            f"{TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN} and "
            f"{TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX}"
        )
    breakeven = (
        _apply_transaction_cost_breakeven_defaults(cfg.breakeven)
        if cfg.breakeven is not None
        else None
    )
    return ResultConsistencyTransactionCostRobustnessConfig(
        mode=mode,
        stress_multipliers=[
            _coerce_float(
                value,
                f"validation.result_consistency.transaction_cost_robustness.stress_multipliers[{idx}]",
            )
            for idx, value in enumerate(stress_multipliers)
        ],
        max_metric_drop_pct=_coerce_float(
            cfg.max_metric_drop_pct,
            "validation.result_consistency.transaction_cost_robustness.max_metric_drop_pct",
        ),
        breakeven=breakeven,
    )


def _normalize_result_consistency_config(
    cfg: ResultConsistencyConfig | None,
    prefix: str,
) -> ResultConsistencyConfig | None:
    if cfg is None:
        return None
    outlier_dependency = _normalize_result_consistency_outlier_dependency_config(
        getattr(cfg, "outlier_dependency", None),
        f"{prefix}.outlier_dependency",
    )
    execution_price_variance = _normalize_result_consistency_execution_price_variance_config(
        getattr(cfg, "execution_price_variance", None),
        f"{prefix}.execution_price_variance",
    )
    lookahead_shuffle_test = _normalize_lookahead_shuffle_test_config(
        getattr(cfg, "lookahead_shuffle_test", None),
        f"{prefix}.lookahead_shuffle_test",
    )
    data_integrity_audit = _normalize_result_consistency_data_integrity_audit_config(
        getattr(cfg, "data_integrity_audit", None),
        f"{prefix}.data_integrity_audit",
    )
    transaction_cost_robustness = _normalize_transaction_cost_robustness_config(
        getattr(cfg, "transaction_cost_robustness", None),
        f"{prefix}.transaction_cost_robustness",
    )
    if (
        outlier_dependency is None
        and execution_price_variance is None
        and lookahead_shuffle_test is None
        and data_integrity_audit is None
        and transaction_cost_robustness is None
    ):
        raise ValueError(
            f"Invalid `{prefix}`: expected at least one configured module "
            "(`outlier_dependency`, `execution_price_variance`, `lookahead_shuffle_test`, "
            "`data_integrity_audit`, or `transaction_cost_robustness`)"
        )
    min_metric_raw = getattr(cfg, "min_metric", None)
    min_metric = _coerce_float(min_metric_raw, f"{prefix}.min_metric") if min_metric_raw is not None else None
    min_trades_raw = getattr(cfg, "min_trades", None)
    min_trades = _coerce_int(min_trades_raw, f"{prefix}.min_trades") if min_trades_raw is not None else None
    if min_trades is not None and min_trades < RESULT_CONSISTENCY_MIN_TRADES_MIN:
        raise ValueError(f"`{prefix}.min_trades` must be >= {RESULT_CONSISTENCY_MIN_TRADES_MIN}")
    return ResultConsistencyConfig(
        min_metric=min_metric,
        min_trades=min_trades,
        outlier_dependency=outlier_dependency,
        execution_price_variance=execution_price_variance,
        lookahead_shuffle_test=lookahead_shuffle_test,
        data_integrity_audit=data_integrity_audit,
        transaction_cost_robustness=transaction_cost_robustness,
    )


def _apply_result_consistency_defaults(cfg: ResultConsistencyConfig) -> ResultConsistencyConfig:
    return ResultConsistencyConfig(
        min_metric=cfg.min_metric,
        min_trades=cfg.min_trades,
        outlier_dependency=(
            _apply_result_consistency_outlier_dependency_defaults(cfg.outlier_dependency)
            if cfg.outlier_dependency is not None
            else None
        ),
        execution_price_variance=(
            _apply_result_consistency_execution_price_variance_defaults(cfg.execution_price_variance)
            if cfg.execution_price_variance is not None
            else None
        ),
        lookahead_shuffle_test=(
            _apply_lookahead_shuffle_test_defaults(
                cfg.lookahead_shuffle_test,
                LOOKAHEAD_SHUFFLE_TEST_CONFIG_PREFIX,
            )
            if cfg.lookahead_shuffle_test is not None
            else None
        ),
        data_integrity_audit=(
            _apply_result_consistency_data_integrity_audit_defaults(cfg.data_integrity_audit)
            if cfg.data_integrity_audit is not None
            else None
        ),
        transaction_cost_robustness=(
            _apply_transaction_cost_robustness_defaults(cfg.transaction_cost_robustness)
            if cfg.transaction_cost_robustness is not None
            else None
        ),
    )


def _merge_data_quality_config(
    base: ValidationDataQualityConfig | None,
    override: ValidationDataQualityConfig | None,
) -> ValidationDataQualityConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_data_quality_config(
        ValidationDataQualityConfig(
            min_data_points=_merged_field(base, override, "min_data_points"),
            calendar=_merge_calendar_config(
                getattr(base, "calendar", None),
                getattr(override, "calendar", None),
            ),
            continuity=_merge_continuity_config(
                getattr(base, "continuity", None),
                getattr(override, "continuity", None),
            ),
            kurtosis=_merged_field(base, override, "kurtosis"),
            ohlc_integrity=_merge_ohlc_integrity_config(
                getattr(base, "ohlc_integrity", None),
                getattr(override, "ohlc_integrity", None),
            ),
            outlier_detection=_merge_outlier_detection_config(
                getattr(base, "outlier_detection", None),
                getattr(override, "outlier_detection", None),
            ),
            stationarity=_merge_stationarity_config(
                getattr(base, "stationarity", None),
                getattr(override, "stationarity", None),
            ),
            is_verified=_merged_field(base, override, "is_verified"),
            on_fail=_merged_field(base, override, "on_fail"),
        ),
        "validation.data_quality",
    )
    return _apply_data_quality_defaults(
        _require_normalized(normalized, "validation.data_quality")
    )


def _merge_optimization_config(
    base: OptimizationPolicyConfig | None,
    override: OptimizationPolicyConfig | None,
) -> OptimizationPolicyConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_optimization_config(
        OptimizationPolicyConfig(
            on_fail=_merged_field(base, override, "on_fail"),
            min_bars=_merged_field(base, override, "min_bars"),
            dof_multiplier=_merged_field(base, override, "dof_multiplier"),
            runtime_error_max_per_tuple=_merged_field(base, override, "runtime_error_max_per_tuple"),
        ),
        "validation.optimization",
    )
    return _apply_optimization_defaults(
        _require_normalized(normalized, "validation.optimization")
    )


def _merge_result_consistency_config(
    base: ResultConsistencyConfig | None,
    override: ResultConsistencyConfig | None,
) -> ResultConsistencyConfig | None:
    if base is None and override is None:
        return None
    merged = ResultConsistencyConfig(
        min_metric=_merged_field(base, override, "min_metric"),
        min_trades=_merged_field(base, override, "min_trades"),
        outlier_dependency=_merge_result_consistency_outlier_dependency_config(
            getattr(base, "outlier_dependency", None),
            getattr(override, "outlier_dependency", None),
        ),
        execution_price_variance=_merge_result_consistency_execution_price_variance_config(
            getattr(base, "execution_price_variance", None),
            getattr(override, "execution_price_variance", None),
        ),
        lookahead_shuffle_test=_merge_lookahead_shuffle_test_config(
            getattr(base, "lookahead_shuffle_test", None),
            getattr(override, "lookahead_shuffle_test", None),
        ),
        data_integrity_audit=_merge_result_consistency_data_integrity_audit_config(
            getattr(base, "data_integrity_audit", None),
            getattr(override, "data_integrity_audit", None),
        ),
        transaction_cost_robustness=_merge_transaction_cost_robustness_config(
            getattr(base, "transaction_cost_robustness", None),
            getattr(override, "transaction_cost_robustness", None),
        ),
    )
    if (
        merged.outlier_dependency is None
        and merged.execution_price_variance is None
        and merged.lookahead_shuffle_test is None
        and merged.data_integrity_audit is None
        and merged.transaction_cost_robustness is None
    ):
        return None
    normalized = _normalize_result_consistency_config(merged, "validation.result_consistency")
    return _apply_result_consistency_defaults(
        _require_normalized(normalized, "validation.result_consistency")
    )


def _merge_result_consistency_outlier_dependency_config(
    base: ResultConsistencyOutlierDependencyConfig | None,
    override: ResultConsistencyOutlierDependencyConfig | None,
) -> ResultConsistencyOutlierDependencyConfig | None:
    if base is None and override is None:
        return None
    return ResultConsistencyOutlierDependencyConfig(
        slices=_merged_field(base, override, "slices"),
        profit_share_threshold=_merged_field(base, override, "profit_share_threshold"),
        trade_share_threshold=_merged_field(base, override, "trade_share_threshold"),
    )


def _merge_result_consistency_execution_price_variance_config(
    base: ResultConsistencyExecutionPriceVarianceConfig | None,
    override: ResultConsistencyExecutionPriceVarianceConfig | None,
) -> ResultConsistencyExecutionPriceVarianceConfig | None:
    if base is None and override is None:
        return None
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=_merged_field(base, override, "price_tolerance_bps"),
    )


def _merge_result_consistency_data_integrity_audit_config(
    base: ResultConsistencyDataIntegrityAuditConfig | None,
    override: ResultConsistencyDataIntegrityAuditConfig | None,
) -> ResultConsistencyDataIntegrityAuditConfig | None:
    if base is None and override is None:
        return None
    return ResultConsistencyDataIntegrityAuditConfig(
        min_overlap_ratio=_merged_field(base, override, "min_overlap_ratio"),
        max_median_ohlc_diff_bps=_merged_field(base, override, "max_median_ohlc_diff_bps"),
        max_p95_ohlc_diff_bps=_merged_field(base, override, "max_p95_ohlc_diff_bps"),
    )


def _merge_transaction_cost_breakeven_config(
    base: ResultConsistencyTransactionCostBreakevenConfig | None,
    override: ResultConsistencyTransactionCostBreakevenConfig | None,
) -> ResultConsistencyTransactionCostBreakevenConfig | None:
    if base is None and override is None:
        return None
    return ResultConsistencyTransactionCostBreakevenConfig(
        enabled=_merged_field(base, override, "enabled"),
        min_multiplier=_merged_field(base, override, "min_multiplier"),
        max_multiplier=_merged_field(base, override, "max_multiplier"),
        max_iterations=_merged_field(base, override, "max_iterations"),
        tolerance=_merged_field(base, override, "tolerance"),
    )


def _merge_transaction_cost_robustness_config(
    base: ResultConsistencyTransactionCostRobustnessConfig | None,
    override: ResultConsistencyTransactionCostRobustnessConfig | None,
) -> ResultConsistencyTransactionCostRobustnessConfig | None:
    if base is None and override is None:
        return None
    merged = ResultConsistencyTransactionCostRobustnessConfig(
        mode=_merged_field(base, override, "mode"),
        stress_multipliers=_merged_field(base, override, "stress_multipliers"),
        max_metric_drop_pct=_merged_field(base, override, "max_metric_drop_pct"),
        breakeven=_merge_transaction_cost_breakeven_config(
            getattr(base, "breakeven", None),
            getattr(override, "breakeven", None),
        ),
    )
    if (
        merged.mode is None
        and merged.stress_multipliers is None
        and merged.max_metric_drop_pct is None
        and merged.breakeven is None
    ):
        return None
    normalized = _normalize_transaction_cost_robustness_config(
        merged,
        "validation.result_consistency.transaction_cost_robustness",
    )
    return _apply_transaction_cost_robustness_defaults(
        _require_normalized(
            normalized,
            "validation.result_consistency.transaction_cost_robustness",
        )
    )


def _merge_continuity_config(
    base: ValidationContinuityConfig | None,
    override: ValidationContinuityConfig | None,
) -> ValidationContinuityConfig | None:
    if base is None and override is None:
        return None
    return ValidationContinuityConfig(
        min_score=_merged_field(base, override, "min_score"),
        max_missing_bar_pct=_merged_field(base, override, "max_missing_bar_pct"),
    )


def _merge_ohlc_integrity_config(
    base: ValidationOHLCIntegrityConfig | None,
    override: ValidationOHLCIntegrityConfig | None,
) -> ValidationOHLCIntegrityConfig | None:
    if base is None and override is None:
        return None
    return ValidationOHLCIntegrityConfig(
        max_invalid_bar_pct=_merged_field(base, override, "max_invalid_bar_pct"),
        allow_negative_price=_merged_field(base, override, "allow_negative_price"),
        allow_negative_volume=_merged_field(base, override, "allow_negative_volume"),
    )


def _merge_calendar_config(
    base: ValidationCalendarConfig | None,
    override: ValidationCalendarConfig | None,
) -> ValidationCalendarConfig | None:
    if base is None and override is None:
        return None
    return ValidationCalendarConfig(
        kind=_merged_field(base, override, "kind"),
        exchange=_merged_field(base, override, "exchange"),
        timezone=_merged_field(base, override, "timezone"),
    )


def _merge_outlier_detection_config(
    base: ValidationOutlierDetectionConfig | None,
    override: ValidationOutlierDetectionConfig | None,
) -> ValidationOutlierDetectionConfig | None:
    if base is None and override is None:
        return None
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=_merged_field(base, override, "max_outlier_pct"),
        method=_merged_field(base, override, "method"),
        zscore_threshold=_merged_field(base, override, "zscore_threshold"),
    )


def _normalize_stationarity_regime_shift_config(
    cfg: ValidationStationarityRegimeShiftConfig | None,
    prefix: str,
) -> ValidationStationarityRegimeShiftConfig | None:
    if cfg is None:
        return None
    window = getattr(cfg, "window", None)
    if window is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): window")
    window = _coerce_int(window, f"{prefix}.window")
    if window < STATIONARITY_REGIME_SHIFT_WINDOW_MIN:
        raise ValueError(f"`{prefix}.window` must be >= {STATIONARITY_REGIME_SHIFT_WINDOW_MIN}")
    mean_shift_max = getattr(cfg, "mean_shift_max", None)
    if mean_shift_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): mean_shift_max")
    mean_shift_max = _coerce_float(mean_shift_max, f"{prefix}.mean_shift_max")
    if mean_shift_max < STATIONARITY_REGIME_SHIFT_MEAN_SHIFT_MIN:
        raise ValueError(
            f"`{prefix}.mean_shift_max` must be >= {STATIONARITY_REGIME_SHIFT_MEAN_SHIFT_MIN}"
        )
    vol_ratio_max = getattr(cfg, "vol_ratio_max", None)
    if vol_ratio_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): vol_ratio_max")
    vol_ratio_max = _coerce_float(vol_ratio_max, f"{prefix}.vol_ratio_max")
    if vol_ratio_max < STATIONARITY_REGIME_SHIFT_VOL_RATIO_MIN:
        raise ValueError(f"`{prefix}.vol_ratio_max` must be >= {STATIONARITY_REGIME_SHIFT_VOL_RATIO_MIN}")
    return ValidationStationarityRegimeShiftConfig(
        window=window,
        mean_shift_max=float(mean_shift_max),
        vol_ratio_max=float(vol_ratio_max),
    )


def _normalize_stationarity_config(
    cfg: ValidationStationarityConfig | None,
    prefix: str,
) -> ValidationStationarityConfig | None:
    if cfg is None:
        return None
    adf_pvalue_max = getattr(cfg, "adf_pvalue_max", None)
    if adf_pvalue_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): adf_pvalue_max")
    adf_pvalue_max = _coerce_float(adf_pvalue_max, f"{prefix}.adf_pvalue_max")
    if adf_pvalue_max < VALIDATION_PROBABILITY_MIN or adf_pvalue_max > VALIDATION_PROBABILITY_MAX:
        raise ValueError(
            f"`{prefix}.adf_pvalue_max` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
        )
    kpss_pvalue_min = getattr(cfg, "kpss_pvalue_min", None)
    if kpss_pvalue_min is not None:
        kpss_pvalue_min = _coerce_float(kpss_pvalue_min, f"{prefix}.kpss_pvalue_min")
        if (
            kpss_pvalue_min < VALIDATION_PROBABILITY_MIN
            or kpss_pvalue_min > VALIDATION_PROBABILITY_MAX
        ):
            raise ValueError(
                f"`{prefix}.kpss_pvalue_min` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
            )
    min_points = getattr(cfg, "min_points", None)
    normalized_min_points = _coerce_int(min_points, f"{prefix}.min_points") if min_points is not None else None
    if normalized_min_points is not None and normalized_min_points < STATIONARITY_MIN_POINTS_MIN:
        raise ValueError(f"`{prefix}.min_points` must be >= {STATIONARITY_MIN_POINTS_MIN}")
    regime_shift = _normalize_stationarity_regime_shift_config(
        getattr(cfg, "regime_shift", None),
        f"{prefix}.regime_shift",
    )
    return ValidationStationarityConfig(
        adf_pvalue_max=adf_pvalue_max,
        kpss_pvalue_min=kpss_pvalue_min,
        min_points=normalized_min_points,
        regime_shift=regime_shift,
    )


def _apply_stationarity_defaults(cfg: ValidationStationarityConfig) -> ValidationStationarityConfig:
    min_points = cfg.min_points if cfg.min_points is not None else STATIONARITY_MIN_POINTS_DEFAULT
    return ValidationStationarityConfig(
        adf_pvalue_max=cfg.adf_pvalue_max,
        kpss_pvalue_min=cfg.kpss_pvalue_min,
        min_points=min_points,
        regime_shift=(
            _apply_stationarity_regime_shift_defaults(cfg.regime_shift)
            if cfg.regime_shift is not None
            else None
        ),
    )


def _apply_stationarity_regime_shift_defaults(
    cfg: ValidationStationarityRegimeShiftConfig,
) -> ValidationStationarityRegimeShiftConfig:
    return ValidationStationarityRegimeShiftConfig(
        window=cfg.window,
        mean_shift_max=cfg.mean_shift_max,
        vol_ratio_max=cfg.vol_ratio_max,
    )


def _merge_stationarity_regime_shift_config(
    base: ValidationStationarityRegimeShiftConfig | None,
    override: ValidationStationarityRegimeShiftConfig | None,
) -> ValidationStationarityRegimeShiftConfig | None:
    if base is None and override is None:
        return None
    return ValidationStationarityRegimeShiftConfig(
        window=_merged_field(base, override, "window"),
        mean_shift_max=_merged_field(base, override, "mean_shift_max"),
        vol_ratio_max=_merged_field(base, override, "vol_ratio_max"),
    )


def _merge_stationarity_config(
    base: ValidationStationarityConfig | None,
    override: ValidationStationarityConfig | None,
) -> ValidationStationarityConfig | None:
    if base is None and override is None:
        return None
    return ValidationStationarityConfig(
        adf_pvalue_max=_merged_field(base, override, "adf_pvalue_max"),
        kpss_pvalue_min=_merged_field(base, override, "kpss_pvalue_min"),
        min_points=_merged_field(base, override, "min_points"),
        regime_shift=_merge_stationarity_regime_shift_config(
            getattr(base, "regime_shift", None),
            getattr(override, "regime_shift", None),
        ),
    )


def _normalize_lookahead_shuffle_test_config(
    cfg: ValidationLookaheadShuffleTestConfig | None,
    prefix: str,
) -> ValidationLookaheadShuffleTestConfig | None:
    if cfg is None:
        return None
    permutations = _normalize_lookahead_permutations(cfg, prefix)
    pvalue_max = _normalize_lookahead_pvalue_max(cfg, prefix)
    seed = _normalize_lookahead_seed(cfg, prefix)
    max_failed_permutations = _normalize_lookahead_max_failed_permutations(cfg, prefix, permutations)
    return ValidationLookaheadShuffleTestConfig(
        permutations=permutations,
        pvalue_max=pvalue_max,
        seed=seed,
        max_failed_permutations=max_failed_permutations,
    )


def _normalize_lookahead_permutations(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> int | None:
    permutations_raw = getattr(cfg, "permutations", None)
    permutations = _coerce_int(permutations_raw, f"{prefix}.permutations") if permutations_raw is not None else None
    if permutations is not None and permutations < LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_MIN:
        raise ValueError(
            f"`{prefix}.permutations` must be >= {LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_MIN}"
        )
    return permutations


def _normalize_lookahead_pvalue_max(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> float | None:
    pvalue_max_raw = getattr(cfg, "pvalue_max", None)
    pvalue_max = _coerce_float(pvalue_max_raw, f"{prefix}.pvalue_max") if pvalue_max_raw is not None else None
    if pvalue_max is not None and not (
        VALIDATION_PROBABILITY_MIN <= pvalue_max <= VALIDATION_PROBABILITY_MAX
    ):
        raise ValueError(
            f"`{prefix}.pvalue_max` must be between {VALIDATION_PROBABILITY_MIN} and {VALIDATION_PROBABILITY_MAX}"
        )
    return pvalue_max


def _normalize_lookahead_seed(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> int | None:
    seed_raw = getattr(cfg, "seed", None)
    seed = _coerce_int(seed_raw, f"{prefix}.seed") if seed_raw is not None else None
    if seed is not None and seed < LOOKAHEAD_SHUFFLE_TEST_SEED_MIN:
        raise ValueError(f"`{prefix}.seed` must be >= {LOOKAHEAD_SHUFFLE_TEST_SEED_MIN}")
    return seed


def _normalize_lookahead_max_failed_permutations(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
    permutations: int | None,
) -> int | None:
    max_failed_permutations_raw = getattr(cfg, "max_failed_permutations", None)
    max_failed_permutations = (
        _coerce_int(max_failed_permutations_raw, f"{prefix}.max_failed_permutations")
        if max_failed_permutations_raw is not None
        else None
    )
    if max_failed_permutations is not None:
        if max_failed_permutations < LOOKAHEAD_SHUFFLE_TEST_FAILED_PERMUTATIONS_MIN:
            raise ValueError(
                f"`{prefix}.max_failed_permutations` must be >= {LOOKAHEAD_SHUFFLE_TEST_FAILED_PERMUTATIONS_MIN}"
            )
        effective_permutations = (
            permutations
            if permutations is not None
            else LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_DEFAULT
        )
        if max_failed_permutations > effective_permutations:
            raise ValueError(
                f"`{prefix}.max_failed_permutations` must be <= `{prefix}.permutations`"
            )
    return max_failed_permutations


def _apply_lookahead_shuffle_test_defaults(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> ValidationLookaheadShuffleTestConfig:
    if cfg.permutations is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): permutations")
    if cfg.pvalue_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): pvalue_max")
    effective_permutations = cfg.permutations
    max_failed_permutations = cfg.max_failed_permutations
    if (
        max_failed_permutations is not None
        and max_failed_permutations > effective_permutations
    ):
        raise ValueError(
            f"`{prefix}.max_failed_permutations` must be <= `{prefix}.permutations`"
        )
    return ValidationLookaheadShuffleTestConfig(
        permutations=effective_permutations,
        pvalue_max=cfg.pvalue_max,
        seed=cfg.seed if cfg.seed is not None else LOOKAHEAD_SHUFFLE_TEST_SEED_DEFAULT,
        max_failed_permutations=max_failed_permutations,
    )


def _merge_lookahead_shuffle_test_config(
    base: ValidationLookaheadShuffleTestConfig | None,
    override: ValidationLookaheadShuffleTestConfig | None,
) -> ValidationLookaheadShuffleTestConfig | None:
    if base is None and override is None:
        return None
    return ValidationLookaheadShuffleTestConfig(
        permutations=_merged_field(base, override, "permutations"),
        pvalue_max=_merged_field(base, override, "pvalue_max"),
        seed=_merged_field(base, override, "seed"),
        max_failed_permutations=_merged_field(base, override, "max_failed_permutations"),
    )


def resolve_validation_overrides(cfg: Config) -> None:
    """Resolve effective collection-level validation policies.

    For each module (`data_quality`, `optimization`, `result_consistency`):
    effective policy = merge(global_policy, collection_override).
    """
    validation_cfg = cfg.validation
    if validation_cfg is None:
        global_data_quality_policy = None
        global_optimization_policy = None
        global_result_consistency_policy = None
    else:
        # Build normalized runtime globals without mutating the source config object.
        global_data_quality_policy = (
            _merge_data_quality_config(validation_cfg.data_quality, None)
            if validation_cfg.data_quality is not None
            else None
        )
        global_optimization_policy = (
            _merge_optimization_config(validation_cfg.optimization, None)
            if validation_cfg.optimization is not None
            else None
        )
        global_result_consistency_policy = (
            _merge_result_consistency_config(validation_cfg.result_consistency, None)
            if validation_cfg.result_consistency is not None
            else None
        )

    for collection in cfg.collections:
        collection_validation = collection.validation
        resolved_data_quality = _merge_data_quality_config(
            global_data_quality_policy,
            getattr(collection_validation, "data_quality", None),
        )
        resolved_optimization = _merge_optimization_config(
            global_optimization_policy,
            getattr(collection_validation, "optimization", None),
        )
        resolved_result_consistency = _merge_result_consistency_config(
            global_result_consistency_policy,
            getattr(collection_validation, "result_consistency", None),
        )
        if collection.reference_source:
            base_policy = (
                resolved_result_consistency
                if resolved_result_consistency is not None
                else ResultConsistencyConfig()
            )
            if getattr(base_policy, "data_integrity_audit", None) is None:
                base_policy = ResultConsistencyConfig(
                    min_metric=base_policy.min_metric,
                    min_trades=base_policy.min_trades,
                    outlier_dependency=base_policy.outlier_dependency,
                    execution_price_variance=base_policy.execution_price_variance,
                    lookahead_shuffle_test=base_policy.lookahead_shuffle_test,
                    transaction_cost_robustness=base_policy.transaction_cost_robustness,
                    data_integrity_audit=_default_data_integrity_audit_config(),
                )
            resolved_result_consistency = _merge_result_consistency_config(base_policy, None)
        if (
            resolved_data_quality is None
            and resolved_optimization is None
            and resolved_result_consistency is None
        ):
            continue
        collection.validation = ValidationConfig(
            data_quality=resolved_data_quality,
            optimization=resolved_optimization,
            result_consistency=resolved_result_consistency,
        )


def require_mapping(raw: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")
    return cast(dict[str, Any], raw)


def require_keys(raw: dict[str, Any], prefix: str, keys: list[str]) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        formatted = ", ".join(f"`{key}`" for key in missing)
        raise ValueError(f"Invalid `{prefix}`: missing required key(s): {formatted}")


def _coerce_int(value: Any, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid `{field_path}`: expected an integer")
    return value


def _coerce_float(value: Any, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid `{field_path}`: expected a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"`{field_path}` must be finite")
    return parsed


def parse_optional_int(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    parsed = _coerce_int(value, f"{prefix}.{key}")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"`{prefix}.{key}` must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"`{prefix}.{key}` must be <= {max_value}")
    return parsed


def parse_required_int(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = parse_optional_int(raw, prefix, key, min_value=min_value, max_value=max_value)
    if value is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): {key}")
    return value


def parse_optional_float(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    parsed = _coerce_float(value, f"{prefix}.{key}")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"`{prefix}.{key}` must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"`{prefix}.{key}` must be <= {max_value}")
    return parsed


def parse_optional_float_list(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[float] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Invalid `{prefix}.{key}`: expected a list")
    parsed: list[float] = []
    for idx, item in enumerate(value):
        parsed_item = _coerce_float(item, f"{prefix}.{key}[{idx}]")
        if min_value is not None and parsed_item < min_value:
            raise ValueError(f"`{prefix}.{key}[{idx}]` must be >= {min_value}")
        if max_value is not None and parsed_item > max_value:
            raise ValueError(f"`{prefix}.{key}[{idx}]` must be <= {max_value}")
        parsed.append(parsed_item)
    return parsed


def parse_required_float(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = parse_optional_float(raw, prefix, key, min_value=min_value, max_value=max_value)
    if value is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): {key}")
    return value


def parse_optional_str(
    raw: dict[str, Any],
    key: str,
    *,
    normalize: bool = True,
) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed.lower() if normalize else parsed


def parse_required_str(
    raw: dict[str, Any],
    prefix: str,
    key: str,
    *,
    normalize: bool = True,
) -> str:
    value = parse_optional_str(raw, key, normalize=normalize)
    if value is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): {key}")
    return value


def parse_optional_bool(
    raw: dict[str, Any],
    prefix: str,
    key: str,
) -> bool | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid `{prefix}.{key}`: expected a boolean")


def _parse_on_fail(
    raw_value: Any,
    field_path: str,
    allowed_on_fail: set[str],
) -> str | None:
    if raw_value is None:
        return None
    on_fail = str(raw_value).strip().lower()
    if on_fail not in allowed_on_fail:
        raise ValueError(
            f"Invalid `{field_path}`: expected one of {sorted(allowed_on_fail)}, got '{on_fail}'"
        )
    return on_fail


def parse_required_on_fail(
    raw_value: Any,
    field_path: str,
    allowed_on_fail: set[str],
) -> str:
    parsed = _parse_on_fail(raw_value, field_path, allowed_on_fail)
    if parsed is None:
        section = field_path.rsplit(".", 1)[0]
        key = field_path.rsplit(".", 1)[1]
        raise ValueError(f"Invalid `{section}`: missing required field(s): {key}")
    return parsed


def _parse_utc_timezone(raw_value: Any, field_path: str) -> str | None:
    if raw_value is None:
        return None
    timezone = str(raw_value).strip().upper()
    if timezone == "UTC":
        return timezone
    if not re.fullmatch(r"UTC[+-](?:0\d|1[0-4]):[0-5]\d", timezone):
        raise ValueError(
            f"Invalid `{field_path}`: expected UTC or UTC±HH:MM (e.g. UTC+00:00)"
        )
    return timezone


def _parse_validation_data_quality(
    raw: Any, prefix: str
) -> ValidationDataQualityConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)

    on_fail = parse_required_on_fail(
        parsed_raw.get("on_fail"),
        f"{prefix}.on_fail",
        {"skip_optimization", "skip_job", "skip_collection"},
    )
    min_data_points_cfg = parse_optional_int(
        parsed_raw, prefix, "min_data_points", min_value=VALIDATION_NON_NEGATIVE_INT_MIN
    )
    calendar_cfg = _parse_validation_calendar(parsed_raw.get("calendar"), f"{prefix}.calendar")
    continuity_cfg = _parse_continuity(
        parsed_raw.get("continuity"), f"{prefix}.continuity"
    )
    kurtosis_cfg = parse_optional_float(
        parsed_raw, prefix, "kurtosis", min_value=VALIDATION_NON_NEGATIVE_FLOAT_MIN
    )
    ohlc_integrity_cfg = _parse_ohlc_integrity(
        parsed_raw.get("ohlc_integrity"), f"{prefix}.ohlc_integrity"
    )
    outlier_detection_cfg = _parse_outlier_detection(
        parsed_raw.get("outlier_detection"), f"{prefix}.outlier_detection"
    )
    stationarity_cfg = _parse_stationarity(
        parsed_raw.get("stationarity"), f"{prefix}.stationarity"
    )
    is_verified = parse_optional_bool(parsed_raw, prefix, "is_verified")

    return _normalize_data_quality_config(
        ValidationDataQualityConfig(
            min_data_points=min_data_points_cfg,
            calendar=calendar_cfg,
            continuity=continuity_cfg,
            kurtosis=kurtosis_cfg,
            ohlc_integrity=ohlc_integrity_cfg,
            outlier_detection=outlier_detection_cfg,
            stationarity=stationarity_cfg,
            is_verified=is_verified,
            on_fail=on_fail,
        ),
        prefix,
    )

def _parse_continuity(
    raw: Any,
    prefix: str,
) -> ValidationContinuityConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    if "calendar" in parsed_raw:
        raise ValueError(
            f"Invalid `{prefix}.calendar`: configure calendar under `validation.data_quality.calendar`"
        )
    min_score = parse_optional_float(
        parsed_raw,
        prefix,
        "min_score",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    max_missing = parse_optional_float(
        parsed_raw,
        prefix,
        "max_missing_bar_pct",
        min_value=VALIDATION_PERCENT_MIN,
        max_value=VALIDATION_PERCENT_MAX,
    )
    return ValidationContinuityConfig(
        min_score=min_score,
        max_missing_bar_pct=max_missing,
    )


def _parse_ohlc_integrity(
    raw: Any,
    prefix: str,
) -> ValidationOHLCIntegrityConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    max_invalid_bar_pct = parse_optional_float(
        parsed_raw,
        prefix,
        "max_invalid_bar_pct",
        min_value=VALIDATION_PERCENT_MIN,
        max_value=VALIDATION_PERCENT_MAX,
    )
    allow_negative_price = parse_optional_bool(parsed_raw, prefix, "allow_negative_price")
    allow_negative_volume = parse_optional_bool(parsed_raw, prefix, "allow_negative_volume")
    return ValidationOHLCIntegrityConfig(
        max_invalid_bar_pct=max_invalid_bar_pct,
        allow_negative_price=allow_negative_price,
        allow_negative_volume=allow_negative_volume,
    )

def _parse_outlier_detection(
    raw: Any, prefix: str
) -> ValidationOutlierDetectionConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    max_outlier_pct = parse_required_float(
        parsed_raw,
        prefix,
        "max_outlier_pct",
        min_value=VALIDATION_PERCENT_MIN,
        max_value=VALIDATION_PERCENT_MAX,
    )
    method = parse_required_str(parsed_raw, prefix, "method")
    if method not in {"zscore", "modified_zscore"}:
        raise ValueError(
            f"Invalid `{prefix}.method`: expected one of ['modified_zscore', 'zscore']"
        )
    zscore_threshold = parse_required_float(parsed_raw, prefix, "zscore_threshold")
    if zscore_threshold <= OUTLIER_DETECTION_ZSCORE_THRESHOLD_MIN_EXCLUSIVE:
        raise ValueError(
            f"`{prefix}.zscore_threshold` must be > {OUTLIER_DETECTION_ZSCORE_THRESHOLD_MIN_EXCLUSIVE}"
        )
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=max_outlier_pct,
        method=method,
        zscore_threshold=zscore_threshold,
    )


def _parse_stationarity_regime_shift(
    raw: Any, prefix: str
) -> ValidationStationarityRegimeShiftConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    window = parse_required_int(
        parsed_raw,
        prefix,
        "window",
        min_value=STATIONARITY_REGIME_SHIFT_WINDOW_MIN,
    )
    mean_shift_max = parse_required_float(
        parsed_raw,
        prefix,
        "mean_shift_max",
        min_value=STATIONARITY_REGIME_SHIFT_MEAN_SHIFT_MIN,
    )
    vol_ratio_max = parse_required_float(
        parsed_raw,
        prefix,
        "vol_ratio_max",
        min_value=STATIONARITY_REGIME_SHIFT_VOL_RATIO_MIN,
    )
    return ValidationStationarityRegimeShiftConfig(
        window=window,
        mean_shift_max=mean_shift_max,
        vol_ratio_max=vol_ratio_max,
    )


def _parse_stationarity(
    raw: Any, prefix: str
) -> ValidationStationarityConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    adf_pvalue_max = parse_required_float(
        parsed_raw,
        prefix,
        "adf_pvalue_max",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    kpss_pvalue_min = parse_optional_float(
        parsed_raw,
        prefix,
        "kpss_pvalue_min",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    min_points = parse_optional_int(
        parsed_raw,
        prefix,
        "min_points",
        min_value=STATIONARITY_MIN_POINTS_MIN,
    )
    regime_shift_raw = parsed_raw.get("regime_shift")
    if regime_shift_raw is not None and not isinstance(regime_shift_raw, dict):
        raise ValueError(f"Invalid `{prefix}.regime_shift`: expected a mapping")
    regime_shift = None
    if regime_shift_raw is not None:
        regime_shift = _parse_stationarity_regime_shift(
            regime_shift_raw, f"{prefix}.regime_shift"
        )
    return ValidationStationarityConfig(
        adf_pvalue_max=float(adf_pvalue_max),
        kpss_pvalue_min=float(kpss_pvalue_min) if kpss_pvalue_min is not None else None,
        min_points=int(min_points) if min_points is not None else None,
        regime_shift=regime_shift,
    )


def _parse_lookahead_shuffle_test(
    raw: Any,
    prefix: str,
) -> ValidationLookaheadShuffleTestConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    if "threshold" in parsed_raw:
        raise ValueError(
            f"Invalid `{prefix}.threshold`: deprecated; use `{prefix}.pvalue_max`"
        )
    permutations = parse_optional_int(
        parsed_raw,
        prefix,
        "permutations",
        min_value=LOOKAHEAD_SHUFFLE_TEST_PERMUTATIONS_MIN,
    )
    pvalue_max = parse_optional_float(
        parsed_raw,
        prefix,
        "pvalue_max",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    seed = parse_optional_int(parsed_raw, prefix, "seed", min_value=LOOKAHEAD_SHUFFLE_TEST_SEED_MIN)
    max_failed_permutations = parse_optional_int(
        parsed_raw,
        prefix,
        "max_failed_permutations",
        min_value=LOOKAHEAD_SHUFFLE_TEST_FAILED_PERMUTATIONS_MIN,
    )
    return ValidationLookaheadShuffleTestConfig(
        permutations=permutations,
        pvalue_max=pvalue_max,
        seed=seed,
        max_failed_permutations=max_failed_permutations,
    )


def _parse_validation_calendar(raw: Any, prefix: str) -> ValidationCalendarConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    kind_raw = parse_optional_str(parsed_raw, "kind")
    exchange = parsed_raw.get("exchange")
    return ValidationCalendarConfig(
        kind=kind_raw,
        exchange=str(exchange).strip() if exchange is not None else None,
        timezone=parsed_raw.get("timezone"),
    )


def _parse_validation(raw: Any, prefix: str) -> ValidationConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)

    data_quality_raw = parsed_raw.get("data_quality")
    if data_quality_raw is not None and not isinstance(data_quality_raw, dict):
        raise ValueError(f"Invalid `{prefix}.data_quality`: expected a mapping")
    optimization_raw = parsed_raw.get("optimization")
    if optimization_raw is not None and not isinstance(optimization_raw, dict):
        raise ValueError(f"Invalid `{prefix}.optimization`: expected a mapping")
    result_consistency_raw = parsed_raw.get("result_consistency")
    if result_consistency_raw is not None and not isinstance(result_consistency_raw, dict):
        raise ValueError(f"Invalid `{prefix}.result_consistency`: expected a mapping")

    data_quality_cfg = (
        _parse_validation_data_quality(data_quality_raw, f"{prefix}.data_quality")
        if isinstance(data_quality_raw, dict)
        else None
    )
    optimization_cfg = (
        _parse_optimization_policy(optimization_raw, f"{prefix}.optimization")
        if isinstance(optimization_raw, dict)
        else None
    )
    result_consistency_cfg = (
        _parse_result_consistency(result_consistency_raw, f"{prefix}.result_consistency")
        if isinstance(result_consistency_raw, dict)
        else None
    )
    if data_quality_cfg is None and optimization_cfg is None and result_consistency_cfg is None:
        # Keep validation fully optional; no synthetic policy object when both modules are absent.
        return None
    return ValidationConfig(
        data_quality=data_quality_cfg,
        optimization=optimization_cfg,
        result_consistency=result_consistency_cfg,
    )


def _parse_optimization_policy(raw: Any, prefix: str) -> OptimizationPolicyConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    on_fail = parse_required_on_fail(
        parsed_raw.get("on_fail"),
        f"{prefix}.on_fail",
        {"baseline_only", "skip_job"},
    )

    min_bars = parse_required_int(
        parsed_raw,
        prefix,
        "min_bars",
        min_value=VALIDATION_NON_NEGATIVE_INT_MIN,
    )
    dof_multiplier = parse_required_int(
        parsed_raw,
        prefix,
        "dof_multiplier",
        min_value=VALIDATION_NON_NEGATIVE_INT_MIN,
    )
    runtime_error_max_per_tuple = parse_optional_int(
        parsed_raw,
        prefix,
        "runtime_error_max_per_tuple",
        min_value=OPTIMIZATION_RUNTIME_ERROR_MAX_PER_TUPLE_MIN,
    )

    return _normalize_optimization_config(
        OptimizationPolicyConfig(
            on_fail=on_fail,
            min_bars=min_bars,
            dof_multiplier=dof_multiplier,
            runtime_error_max_per_tuple=runtime_error_max_per_tuple,
        ),
        prefix,
    )


def _parse_result_consistency(raw: Any, prefix: str) -> ResultConsistencyConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)

    min_metric = parse_optional_float(parsed_raw, prefix, "min_metric")
    min_trades = parse_optional_int(
        parsed_raw,
        prefix,
        "min_trades",
        min_value=RESULT_CONSISTENCY_MIN_TRADES_MIN,
    )

    outlier_dependency_raw = parsed_raw.get("outlier_dependency")
    if outlier_dependency_raw is not None and not isinstance(outlier_dependency_raw, dict):
        raise ValueError(f"Invalid `{prefix}.outlier_dependency`: expected a mapping")
    execution_price_variance_raw = parsed_raw.get("execution_price_variance")
    if execution_price_variance_raw is not None and not isinstance(execution_price_variance_raw, dict):
        raise ValueError(f"Invalid `{prefix}.execution_price_variance`: expected a mapping")

    outlier_dependency = (
        _parse_result_consistency_outlier_dependency(
            outlier_dependency_raw,
            f"{prefix}.outlier_dependency",
        )
        if isinstance(outlier_dependency_raw, dict)
        else None
    )
    execution_price_variance = (
        _parse_result_consistency_execution_price_variance(
            execution_price_variance_raw,
            f"{prefix}.execution_price_variance",
        )
        if isinstance(execution_price_variance_raw, dict)
        else None
    )
    lookahead_shuffle_test_raw = parsed_raw.get("lookahead_shuffle_test")
    if lookahead_shuffle_test_raw is not None and not isinstance(lookahead_shuffle_test_raw, dict):
        raise ValueError(f"Invalid `{prefix}.lookahead_shuffle_test`: expected a mapping")
    lookahead_shuffle_test = (
        _parse_lookahead_shuffle_test(
            lookahead_shuffle_test_raw,
            f"{prefix}.lookahead_shuffle_test",
        )
        if isinstance(lookahead_shuffle_test_raw, dict)
        else None
    )
    data_integrity_audit_raw = parsed_raw.get("data_integrity_audit")
    if data_integrity_audit_raw is not None and not isinstance(data_integrity_audit_raw, dict):
        raise ValueError(f"Invalid `{prefix}.data_integrity_audit`: expected a mapping")
    data_integrity_audit = (
        _parse_result_consistency_data_integrity_audit(
            data_integrity_audit_raw,
            f"{prefix}.data_integrity_audit",
        )
        if isinstance(data_integrity_audit_raw, dict)
        else None
    )
    transaction_cost_robustness_raw = parsed_raw.get("transaction_cost_robustness")
    if (
        transaction_cost_robustness_raw is not None
        and not isinstance(transaction_cost_robustness_raw, dict)
    ):
        raise ValueError(
            f"Invalid `{prefix}.transaction_cost_robustness`: expected a mapping"
        )
    transaction_cost_robustness = (
        _parse_result_consistency_transaction_cost_robustness(
            transaction_cost_robustness_raw,
            f"{prefix}.transaction_cost_robustness",
        )
        if isinstance(transaction_cost_robustness_raw, dict)
        else None
    )

    return _normalize_result_consistency_config(
        ResultConsistencyConfig(
            min_metric=min_metric,
            min_trades=min_trades,
            outlier_dependency=outlier_dependency,
            execution_price_variance=execution_price_variance,
            lookahead_shuffle_test=lookahead_shuffle_test,
            data_integrity_audit=data_integrity_audit,
            transaction_cost_robustness=transaction_cost_robustness,
        ),
        prefix,
    )


def _parse_result_consistency_outlier_dependency(
    raw: Any, prefix: str
) -> ResultConsistencyOutlierDependencyConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    slices = parse_required_int(
        parsed_raw,
        prefix,
        "slices",
        min_value=RESULT_CONSISTENCY_OUTLIER_DEPENDENCY_SLICES_MIN,
    )
    profit_share_threshold = parse_required_float(
        parsed_raw,
        prefix,
        "profit_share_threshold",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    trade_share_threshold = parse_required_float(
        parsed_raw,
        prefix,
        "trade_share_threshold",
        min_value=VALIDATION_PROBABILITY_MIN,
        max_value=VALIDATION_PROBABILITY_MAX,
    )
    return ResultConsistencyOutlierDependencyConfig(
        slices=slices,
        profit_share_threshold=float(profit_share_threshold),
        trade_share_threshold=float(trade_share_threshold),
    )


def _parse_result_consistency_execution_price_variance(
    raw: Any, prefix: str
) -> ResultConsistencyExecutionPriceVarianceConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    price_tolerance_bps = parse_required_float(
        parsed_raw,
        prefix,
        "price_tolerance_bps",
        min_value=VALIDATION_NON_NEGATIVE_FLOAT_MIN,
    )
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=float(price_tolerance_bps),
    )


def _parse_result_consistency_data_integrity_audit(
    raw: Any,
    prefix: str,
) -> ResultConsistencyDataIntegrityAuditConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    return ResultConsistencyDataIntegrityAuditConfig(
        min_overlap_ratio=parse_optional_float(
            parsed_raw,
            prefix,
            "min_overlap_ratio",
            min_value=VALIDATION_PROBABILITY_MIN,
            max_value=VALIDATION_PROBABILITY_MAX,
        ),
        max_median_ohlc_diff_bps=parse_optional_float(
            parsed_raw,
            prefix,
            "max_median_ohlc_diff_bps",
            min_value=VALIDATION_NON_NEGATIVE_FLOAT_MIN,
        ),
        max_p95_ohlc_diff_bps=parse_optional_float(
            parsed_raw,
            prefix,
            "max_p95_ohlc_diff_bps",
            min_value=VALIDATION_NON_NEGATIVE_FLOAT_MIN,
        ),
    )


def _parse_result_consistency_transaction_cost_breakeven(
    raw: Any, prefix: str
) -> ResultConsistencyTransactionCostBreakevenConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    return ResultConsistencyTransactionCostBreakevenConfig(
        enabled=parse_optional_bool(parsed_raw, prefix, "enabled"),
        min_multiplier=parse_optional_float(parsed_raw, prefix, "min_multiplier"),
        max_multiplier=parse_optional_float(parsed_raw, prefix, "max_multiplier"),
        max_iterations=parse_optional_int(parsed_raw, prefix, "max_iterations"),
        tolerance=parse_optional_float(parsed_raw, prefix, "tolerance"),
    )


def _parse_result_consistency_transaction_cost_robustness(
    raw: Any,
    prefix: str,
) -> ResultConsistencyTransactionCostRobustnessConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    stress_multipliers = parse_optional_float_list(
        parsed_raw,
        prefix,
        "stress_multipliers",
    )
    return ResultConsistencyTransactionCostRobustnessConfig(
        mode=parse_optional_str(parsed_raw, "mode"),
        stress_multipliers=stress_multipliers,
        max_metric_drop_pct=parse_optional_float(
            parsed_raw,
            prefix,
            "max_metric_drop_pct",
            min_value=TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MIN,
            max_value=TRANSACTION_COST_ROBUSTNESS_MAX_METRIC_DROP_PCT_MAX,
        ),
        breakeven=_parse_result_consistency_transaction_cost_breakeven(
            parsed_raw.get("breakeven"),
            f"{prefix}.breakeven",
        ),
    )


def _parse_strategies(raw: dict[str, Any]) -> list[StrategyConfig]:
    strategies_raw = raw.get("strategies")
    # `strategies` is an optional override. Missing/empty means "discover all external strategies".
    if strategies_raw is None:
        strategies_raw = []
    elif not isinstance(strategies_raw, list):
        example = (
            "Invalid `strategies` in config; expected a list.\n"
            "Example:\n"
            "strategies:\n"
            "  - name: ExampleStrategy\n"
            "    params: {}\n"
        )
        raise ValueError(example)

    parsed: list[StrategyConfig] = []
    for idx, strategy_raw in enumerate(strategies_raw):
        strategy = require_mapping(strategy_raw, f"strategies[{idx}]")
        require_keys(strategy, f"strategies[{idx}]", ["name"])
        params_raw = strategy.get("params", {})
        if not isinstance(params_raw, dict):
            raise ValueError(f"Invalid `strategies[{idx}].params`: expected a mapping")
        parsed.append(
            StrategyConfig(
                name=str(strategy["name"]).strip(),
                module=parse_optional_str(strategy, "module", normalize=False),
                cls=(
                    parse_optional_str(strategy, "class", normalize=False)
                    or parse_optional_str(strategy, "cls", normalize=False)
                ),
                params=cast(dict[str, list[Any]], params_raw),
            )
        )
    return parsed


def _parse_timeframes(raw: dict[str, Any]) -> list[str]:
    timeframes_raw = raw["timeframes"]
    if not isinstance(timeframes_raw, list):
        raise ValueError("Invalid `timeframes`: expected a list")
    return [str(timeframe).strip() for timeframe in timeframes_raw]


def _parse_metric(raw: dict[str, Any]) -> str:
    metric = str(raw.get("metric", "sharpe")).strip().lower()
    allowed = {"sharpe", "sortino", "profit"}
    if metric not in allowed:
        raise ValueError(f"Invalid `metric`: expected one of {sorted(allowed)}, got '{metric}'")
    return metric


def _parse_collections(raw_collections: Any) -> list[CollectionConfig]:
    if not isinstance(raw_collections, list):
        raise ValueError("Invalid `collections`: expected a list at `collections`")

    collections: list[CollectionConfig] = []
    for idx, collection_raw in enumerate(raw_collections):
        if not isinstance(collection_raw, dict):
            raise ValueError(
                f"Invalid `collections[{idx}]`: expected a mapping at `collections[{idx}]`"
            )
        require_keys(collection_raw, f"collections[{idx}]", ["name", "source", "symbols"])
        collection_validation = _parse_validation(
            collection_raw.get("validation"), f"collections[{idx}].validation"
        )
        symbols_raw = collection_raw["symbols"]
        if not isinstance(symbols_raw, list):
            raise ValueError(f"Invalid `collections[{idx}].symbols`: expected a list")
        collection_fees_raw = collection_raw.get("fees")
        collection_slippage_raw = collection_raw.get("slippage")
        collections.append(
            CollectionConfig(
                name=str(collection_raw["name"]).strip(),
                source=str(collection_raw["source"]).strip(),
                symbols=[str(symbol).strip() for symbol in symbols_raw],
                reference_source=parse_optional_str(
                    collection_raw, "reference_source", normalize=False
                ),
                exchange=parse_optional_str(collection_raw, "exchange", normalize=False),
                currency=parse_optional_str(collection_raw, "currency", normalize=False),
                quote=parse_optional_str(collection_raw, "quote", normalize=False),
                fees=(
                    _coerce_float(collection_fees_raw, f"collections[{idx}].fees")
                    if collection_fees_raw is not None
                    else None
                ),
                slippage=(
                    _coerce_float(collection_slippage_raw, f"collections[{idx}].slippage")
                    if collection_slippage_raw is not None
                    else None
                ),
                validation=collection_validation,
            )
        )
    return collections


def _parse_notifications(raw: dict[str, Any]) -> NotificationsConfig | None:
    notifications_raw = raw.get("notifications")
    if notifications_raw is None:
        return None
    if not isinstance(notifications_raw, dict):
        raise ValueError("Invalid `notifications`: expected a mapping")

    slack_raw = notifications_raw.get("slack")
    if slack_raw is None:
        return None
    if not isinstance(slack_raw, dict):
        raise ValueError("Invalid `notifications.slack`: expected a mapping")
    if not slack_raw.get("webhook_url"):
        return None

    threshold_raw = slack_raw.get("threshold")
    metric = parse_optional_str(slack_raw, "metric")
    if metric is None:
        metric = str(raw.get("metric", "sharpe")).strip().lower()
    return NotificationsConfig(
        slack=SlackNotificationConfig(
            webhook_url=str(slack_raw["webhook_url"]).strip(),
            metric=metric,
            threshold=(
                _coerce_float(threshold_raw, "notifications.slack.threshold")
                if threshold_raw is not None
                else None
            ),
            channel=parse_optional_str(slack_raw, "channel", normalize=False),
            username=parse_optional_str(slack_raw, "username", normalize=False),
        )
    )


def _parse_evaluation_mode(raw: dict[str, Any]) -> str:
    evaluation_mode = str(raw.get("evaluation_mode", "backtest")).strip().lower()
    allowed_modes = {"backtest", "walk_forward"}
    if evaluation_mode not in allowed_modes:
        raise ValueError(
            f"Invalid `evaluation_mode`: expected one of {sorted(allowed_modes)}, got '{evaluation_mode}'"
        )
    return evaluation_mode


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = require_mapping(yaml.safe_load(f), "config")
    require_keys(raw, "config", ["collections", "timeframes"])

    strategies = _parse_strategies(raw)
    collections = _parse_collections(raw["collections"])
    timeframes = _parse_timeframes(raw)
    metric = _parse_metric(raw)
    notifications_cfg = _parse_notifications(raw)
    validation_cfg = _parse_validation(raw.get("validation"), "validation")
    evaluation_mode = _parse_evaluation_mode(raw)

    cfg = Config(
        collections=collections,
        timeframes=timeframes,
        metric=metric,
        strategies=strategies,
        engine=str(raw.get("engine", "pybroker")).lower(),
        param_search=str(raw.get("param_search", raw.get("param_optimizer", "grid"))).lower(),
        param_trials=_coerce_int(raw.get("param_trials", raw.get("opt_trials", 25)), "param_trials"),
        max_workers=_coerce_int(raw.get("max_workers", raw.get("asset_workers", 1)), "max_workers"),
        asset_workers=_coerce_int(raw.get("asset_workers", raw.get("max_workers", 1)), "asset_workers"),
        param_workers=_coerce_int(raw.get("param_workers", 1), "param_workers"),
        max_fetch_concurrency=_coerce_int(raw.get("max_fetch_concurrency", 2), "max_fetch_concurrency"),
        fees=_coerce_float(raw.get("fees", 0.0), "fees"),
        slippage=_coerce_float(raw.get("slippage", 0.0), "slippage"),
        risk_free_rate=_coerce_float(raw.get("risk_free_rate", 0.0), "risk_free_rate"),
        cache_dir=raw.get("cache_dir", ".cache/data"),
        evaluation_mode=evaluation_mode,
        notifications=notifications_cfg,
        validation=validation_cfg,
    )
    resolve_validation_overrides(cfg)
    return cfg
