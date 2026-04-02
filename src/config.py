from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    continuity: "ValidationContinuityConfig | None" = None
    kurtosis: float | None = None
    outlier_detection: "ValidationOutlierDetectionConfig | None" = None
    stationarity: "ValidationStationarityConfig | None" = None
    is_verified: bool | None = None
    on_fail: str | None = None


@dataclass
class ValidationContinuityConfig:
    min_score: float | None = None
    max_missing_bar_pct: float | None = None
    calendar: ValidationCalendarConfig | None = None


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
    threshold: float | None = None
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
class ResultConsistencyConfig:
    outlier_dependency: ResultConsistencyOutlierDependencyConfig | None = None
    execution_price_variance: ResultConsistencyExecutionPriceVarianceConfig | None = None
    lookahead_shuffle_test: ValidationLookaheadShuffleTestConfig | None = None


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


DEFAULT_CALENDAR_KIND = "auto"
STATIONARITY_DEFAULT_MIN_POINTS = 30
LOOKAHEAD_SHUFFLE_TEST_DEFAULT_PERMUTATIONS = 20
LOOKAHEAD_SHUFFLE_TEST_DEFAULT_THRESHOLD = 0.0
LOOKAHEAD_SHUFFLE_TEST_DEFAULT_SEED = 1337


def _merged_field(base: Any, override: Any, field: str) -> Any:
    override_value = getattr(override, field, None)
    if override_value is not None:
        return override_value
    return getattr(base, field, None)


def _apply_calendar_defaults(cfg: ValidationCalendarConfig) -> ValidationCalendarConfig:
    kind = cfg.kind if cfg.kind is not None else DEFAULT_CALENDAR_KIND
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
    min_bars = int(min_bars)
    if min_bars < 0:
        raise ValueError(f"`{prefix}.min_bars` must be >= 0")
    dof_multiplier = getattr(cfg, "dof_multiplier", None)
    if dof_multiplier is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): dof_multiplier")
    dof_multiplier = int(dof_multiplier)
    if dof_multiplier < 0:
        raise ValueError(f"`{prefix}.dof_multiplier` must be >= 0")
    runtime_error_max_per_tuple = getattr(cfg, "runtime_error_max_per_tuple", None)
    if runtime_error_max_per_tuple is not None:
        runtime_error_max_per_tuple = int(runtime_error_max_per_tuple)
        if runtime_error_max_per_tuple < 1:
            raise ValueError(f"`{prefix}.runtime_error_max_per_tuple` must be >= 1")
    return OptimizationPolicyConfig(
        on_fail=on_fail,
        min_bars=min_bars,
        dof_multiplier=dof_multiplier,
        runtime_error_max_per_tuple=runtime_error_max_per_tuple,
    )


def _apply_optimization_defaults(cfg: OptimizationPolicyConfig) -> OptimizationPolicyConfig:
    runtime_error_max_per_tuple = getattr(cfg, "runtime_error_max_per_tuple", None)
    if runtime_error_max_per_tuple is None:
        runtime_error_max_per_tuple = 1
    return OptimizationPolicyConfig(
        on_fail=cfg.on_fail,
        min_bars=cfg.min_bars,
        dof_multiplier=cfg.dof_multiplier,
        runtime_error_max_per_tuple=int(runtime_error_max_per_tuple),
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
        min_data_points = int(min_data_points)
        if min_data_points < 0:
            raise ValueError(f"`{prefix}.min_data_points` must be >= 0")
    kurtosis = getattr(cfg, "kurtosis", None)
    if kurtosis is not None:
        kurtosis = float(kurtosis)
        if kurtosis < 0:
            raise ValueError(f"`{prefix}.kurtosis` must be >= 0")
    return ValidationDataQualityConfig(
        min_data_points=min_data_points,
        continuity=_normalize_continuity_config(
            getattr(cfg, "continuity", None),
            f"{prefix}.continuity",
        ),
        kurtosis=kurtosis,
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
        continuity=_apply_continuity_defaults(cfg.continuity) if cfg.continuity is not None else None,
        kurtosis=cfg.kurtosis,
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
        min_score = float(min_score)
        if min_score < 0 or min_score > 1:
            raise ValueError(f"`{prefix}.min_score` must be between 0 and 1")
    max_missing_bar_pct = getattr(cfg, "max_missing_bar_pct", None)
    if max_missing_bar_pct is not None:
        max_missing_bar_pct = float(max_missing_bar_pct)
        if max_missing_bar_pct < 0 or max_missing_bar_pct > 100:
            raise ValueError(f"`{prefix}.max_missing_bar_pct` must be between 0 and 100")
    return ValidationContinuityConfig(
        min_score=min_score,
        max_missing_bar_pct=max_missing_bar_pct,
        calendar=_normalize_calendar_config(getattr(cfg, "calendar", None), f"{prefix}.calendar"),
    )


def _apply_continuity_defaults(cfg: ValidationContinuityConfig) -> ValidationContinuityConfig:
    return ValidationContinuityConfig(
        min_score=cfg.min_score,
        max_missing_bar_pct=cfg.max_missing_bar_pct,
        calendar=_apply_calendar_defaults(cfg.calendar) if cfg.calendar is not None else None,
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
    max_outlier_pct = float(max_outlier_pct)
    if max_outlier_pct < 0 or max_outlier_pct > 100:
        raise ValueError(f"`{prefix}.max_outlier_pct` must be <= 100")
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
    zscore_threshold = float(zscore_threshold)
    if zscore_threshold <= 0:
        raise ValueError(f"`{prefix}.zscore_threshold` must be > 0")
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
    slices = int(slices)
    if slices < 2:
        raise ValueError(f"`{prefix}.slices` must be >= 2")
    profit_share_threshold = getattr(cfg, "profit_share_threshold", None)
    if profit_share_threshold is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): profit_share_threshold")
    profit_share_threshold = float(profit_share_threshold)
    if profit_share_threshold < 0.0 or profit_share_threshold > 1.0:
        raise ValueError(f"`{prefix}.profit_share_threshold` must be <= 1.0")
    trade_share_threshold = getattr(cfg, "trade_share_threshold", None)
    if trade_share_threshold is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): trade_share_threshold")
    trade_share_threshold = float(trade_share_threshold)
    if trade_share_threshold < 0.0 or trade_share_threshold > 1.0:
        raise ValueError(f"`{prefix}.trade_share_threshold` must be <= 1.0")
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
    price_tolerance_bps = float(price_tolerance_bps)
    if price_tolerance_bps < 0:
        raise ValueError(f"`{prefix}.price_tolerance_bps` must be >= 0.0")
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=price_tolerance_bps,
    )


def _apply_result_consistency_execution_price_variance_defaults(
    cfg: ResultConsistencyExecutionPriceVarianceConfig,
) -> ResultConsistencyExecutionPriceVarianceConfig:
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=cfg.price_tolerance_bps,
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
    if outlier_dependency is None and execution_price_variance is None and lookahead_shuffle_test is None:
        raise ValueError(
            f"Invalid `{prefix}`: expected at least one configured module "
            "(`outlier_dependency`, `execution_price_variance`, or `lookahead_shuffle_test`)"
        )
    return ResultConsistencyConfig(
        outlier_dependency=outlier_dependency,
        execution_price_variance=execution_price_variance,
        lookahead_shuffle_test=lookahead_shuffle_test,
    )


def _apply_result_consistency_defaults(cfg: ResultConsistencyConfig) -> ResultConsistencyConfig:
    return ResultConsistencyConfig(
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
                "validation.result_consistency.lookahead_shuffle_test",
            )
            if cfg.lookahead_shuffle_test is not None
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
            continuity=_merge_continuity_config(
                getattr(base, "continuity", None),
                getattr(override, "continuity", None),
            ),
            kurtosis=_merged_field(base, override, "kurtosis"),
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
    assert normalized is not None
    return _apply_data_quality_defaults(normalized)


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
    assert normalized is not None
    return _apply_optimization_defaults(normalized)


def _merge_result_consistency_config(
    base: ResultConsistencyConfig | None,
    override: ResultConsistencyConfig | None,
) -> ResultConsistencyConfig | None:
    if base is None and override is None:
        return None
    merged = ResultConsistencyConfig(
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
    )
    if (
        merged.outlier_dependency is None
        and merged.execution_price_variance is None
        and merged.lookahead_shuffle_test is None
    ):
        return None
    normalized = _normalize_result_consistency_config(merged, "validation.result_consistency")
    assert normalized is not None
    return _apply_result_consistency_defaults(normalized)


def _merge_result_consistency_outlier_dependency_config(
    base: ResultConsistencyOutlierDependencyConfig | None,
    override: ResultConsistencyOutlierDependencyConfig | None,
) -> ResultConsistencyOutlierDependencyConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_result_consistency_outlier_dependency_config(
        ResultConsistencyOutlierDependencyConfig(
            slices=_merged_field(base, override, "slices"),
            profit_share_threshold=_merged_field(base, override, "profit_share_threshold"),
            trade_share_threshold=_merged_field(base, override, "trade_share_threshold"),
        ),
        "validation.result_consistency.outlier_dependency",
    )
    assert normalized is not None
    return _apply_result_consistency_outlier_dependency_defaults(normalized)


def _merge_result_consistency_execution_price_variance_config(
    base: ResultConsistencyExecutionPriceVarianceConfig | None,
    override: ResultConsistencyExecutionPriceVarianceConfig | None,
) -> ResultConsistencyExecutionPriceVarianceConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_result_consistency_execution_price_variance_config(
        ResultConsistencyExecutionPriceVarianceConfig(
            price_tolerance_bps=_merged_field(base, override, "price_tolerance_bps"),
        ),
        "validation.result_consistency.execution_price_variance",
    )
    assert normalized is not None
    return _apply_result_consistency_execution_price_variance_defaults(normalized)

def _merge_continuity_config(
    base: ValidationContinuityConfig | None,
    override: ValidationContinuityConfig | None,
) -> ValidationContinuityConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_continuity_config(
        ValidationContinuityConfig(
            min_score=_merged_field(base, override, "min_score"),
            max_missing_bar_pct=_merged_field(base, override, "max_missing_bar_pct"),
            calendar=_merge_calendar_config(
                getattr(base, "calendar", None),
                getattr(override, "calendar", None),
            ),
        ),
        "validation.data_quality.continuity",
    )
    assert normalized is not None
    return _apply_continuity_defaults(normalized)


def _merge_calendar_config(
    base: ValidationCalendarConfig | None,
    override: ValidationCalendarConfig | None,
) -> ValidationCalendarConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_calendar_config(
        ValidationCalendarConfig(
            kind=_merged_field(base, override, "kind"),
            exchange=_merged_field(base, override, "exchange"),
            timezone=_merged_field(base, override, "timezone"),
        ),
        "validation.data_quality.continuity.calendar",
    )
    assert normalized is not None
    return _apply_calendar_defaults(normalized)


def _merge_outlier_detection_config(
    base: ValidationOutlierDetectionConfig | None,
    override: ValidationOutlierDetectionConfig | None,
) -> ValidationOutlierDetectionConfig | None:
    if base is None and override is None:
        return None

    normalized = _normalize_outlier_detection_config(
        ValidationOutlierDetectionConfig(
            max_outlier_pct=_merged_field(base, override, "max_outlier_pct"),
            method=_merged_field(base, override, "method"),
            zscore_threshold=_merged_field(base, override, "zscore_threshold"),
        ),
        "validation.data_quality.outlier_detection",
    )
    assert normalized is not None
    return _apply_outlier_detection_defaults(normalized)


def _normalize_stationarity_regime_shift_config(
    cfg: ValidationStationarityRegimeShiftConfig | None,
    prefix: str,
) -> ValidationStationarityRegimeShiftConfig | None:
    if cfg is None:
        return None
    window = getattr(cfg, "window", None)
    if window is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): window")
    window = int(window)
    if window < 10:
        raise ValueError(f"`{prefix}.window` must be >= 10")
    mean_shift_max = getattr(cfg, "mean_shift_max", None)
    if mean_shift_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): mean_shift_max")
    mean_shift_max = float(mean_shift_max)
    if mean_shift_max < 0.0:
        raise ValueError(f"`{prefix}.mean_shift_max` must be >= 0.0")
    vol_ratio_max = getattr(cfg, "vol_ratio_max", None)
    if vol_ratio_max is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): vol_ratio_max")
    vol_ratio_max = float(vol_ratio_max)
    if vol_ratio_max < 1.0:
        raise ValueError(f"`{prefix}.vol_ratio_max` must be >= 1.0")
    return ValidationStationarityRegimeShiftConfig(
        window=window,
        mean_shift_max=mean_shift_max,
        vol_ratio_max=vol_ratio_max,
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
    adf_pvalue_max = float(adf_pvalue_max)
    if adf_pvalue_max < 0.0 or adf_pvalue_max > 1.0:
        raise ValueError(f"`{prefix}.adf_pvalue_max` must be between 0.0 and 1.0")
    kpss_pvalue_min = getattr(cfg, "kpss_pvalue_min", None)
    if kpss_pvalue_min is not None:
        kpss_pvalue_min = float(kpss_pvalue_min)
        if kpss_pvalue_min < 0.0 or kpss_pvalue_min > 1.0:
            raise ValueError(f"`{prefix}.kpss_pvalue_min` must be between 0.0 and 1.0")
    min_points = getattr(cfg, "min_points", None)
    normalized_min_points = int(min_points) if min_points is not None else None
    if normalized_min_points is not None and normalized_min_points < 20:
        raise ValueError(f"`{prefix}.min_points` must be >= 20")
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
    min_points = cfg.min_points if cfg.min_points is not None else STATIONARITY_DEFAULT_MIN_POINTS
    return ValidationStationarityConfig(
        adf_pvalue_max=cfg.adf_pvalue_max,
        kpss_pvalue_min=cfg.kpss_pvalue_min,
        min_points=min_points,
        regime_shift=cfg.regime_shift,
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
    normalized = _normalize_stationarity_regime_shift_config(
        ValidationStationarityRegimeShiftConfig(
            window=_merged_field(base, override, "window"),
            mean_shift_max=_merged_field(base, override, "mean_shift_max"),
            vol_ratio_max=_merged_field(base, override, "vol_ratio_max"),
        ),
        "validation.data_quality.stationarity.regime_shift",
    )
    assert normalized is not None
    return _apply_stationarity_regime_shift_defaults(normalized)


def _merge_stationarity_config(
    base: ValidationStationarityConfig | None,
    override: ValidationStationarityConfig | None,
) -> ValidationStationarityConfig | None:
    if base is None and override is None:
        return None
    normalized = _normalize_stationarity_config(
        ValidationStationarityConfig(
            adf_pvalue_max=_merged_field(base, override, "adf_pvalue_max"),
            kpss_pvalue_min=_merged_field(base, override, "kpss_pvalue_min"),
            min_points=_merged_field(base, override, "min_points"),
            regime_shift=_merge_stationarity_regime_shift_config(
                getattr(base, "regime_shift", None),
                getattr(override, "regime_shift", None),
            ),
        ),
        "validation.data_quality.stationarity",
    )
    assert normalized is not None
    return _apply_stationarity_defaults(normalized)


def _normalize_lookahead_shuffle_test_config(
    cfg: ValidationLookaheadShuffleTestConfig | None,
    prefix: str,
) -> ValidationLookaheadShuffleTestConfig | None:
    if cfg is None:
        return None
    permutations = _normalize_lookahead_permutations(cfg, prefix)
    threshold = _normalize_lookahead_threshold(cfg, prefix)
    seed = _normalize_lookahead_seed(cfg, prefix)
    max_failed_permutations = _normalize_lookahead_max_failed_permutations(cfg, prefix, permutations)
    return ValidationLookaheadShuffleTestConfig(
        permutations=permutations,
        threshold=threshold,
        seed=seed,
        max_failed_permutations=max_failed_permutations,
    )


def _normalize_lookahead_permutations(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> int | None:
    permutations_raw = getattr(cfg, "permutations", None)
    permutations = int(permutations_raw) if permutations_raw is not None else None
    if permutations is not None and permutations < 5:
        raise ValueError(f"`{prefix}.permutations` must be >= 5")
    return permutations


def _normalize_lookahead_threshold(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> float | None:
    threshold_raw = getattr(cfg, "threshold", None)
    threshold = float(threshold_raw) if threshold_raw is not None else None
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError(f"`{prefix}.threshold` must be finite")
    return threshold


def _normalize_lookahead_seed(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
) -> int | None:
    seed_raw = getattr(cfg, "seed", None)
    seed = int(seed_raw) if seed_raw is not None else None
    if seed is not None and seed < 0:
        raise ValueError(f"`{prefix}.seed` must be >= 0")
    return seed


def _normalize_lookahead_max_failed_permutations(
    cfg: ValidationLookaheadShuffleTestConfig,
    prefix: str,
    permutations: int | None,
) -> int | None:
    max_failed_permutations_raw = getattr(cfg, "max_failed_permutations", None)
    max_failed_permutations = (
        int(max_failed_permutations_raw) if max_failed_permutations_raw is not None else None
    )
    if max_failed_permutations is not None:
        if max_failed_permutations < 0:
            raise ValueError(f"`{prefix}.max_failed_permutations` must be >= 0")
        effective_permutations = (
            permutations
            if permutations is not None
            else LOOKAHEAD_SHUFFLE_TEST_DEFAULT_PERMUTATIONS
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
    effective_permutations = (
        cfg.permutations
        if cfg.permutations is not None
        else LOOKAHEAD_SHUFFLE_TEST_DEFAULT_PERMUTATIONS
    )
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
        threshold=(
            cfg.threshold if cfg.threshold is not None else LOOKAHEAD_SHUFFLE_TEST_DEFAULT_THRESHOLD
        ),
        seed=cfg.seed if cfg.seed is not None else LOOKAHEAD_SHUFFLE_TEST_DEFAULT_SEED,
        max_failed_permutations=max_failed_permutations,
    )


def _merge_lookahead_shuffle_test_config(
    base: ValidationLookaheadShuffleTestConfig | None,
    override: ValidationLookaheadShuffleTestConfig | None,
) -> ValidationLookaheadShuffleTestConfig | None:
    if base is None and override is None:
        return None
    permutations = _merged_field(base, override, "permutations")
    threshold = _merged_field(base, override, "threshold")
    seed = _merged_field(base, override, "seed")
    max_failed_permutations = _merged_field(base, override, "max_failed_permutations")
    normalized = _normalize_lookahead_shuffle_test_config(
        ValidationLookaheadShuffleTestConfig(
            permutations=int(permutations) if permutations is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            seed=int(seed) if seed is not None else None,
            max_failed_permutations=(
                int(max_failed_permutations) if max_failed_permutations is not None else None
            ),
        ),
        "validation.result_consistency.lookahead_shuffle_test",
    )
    assert normalized is not None
    return _apply_lookahead_shuffle_test_defaults(
        normalized,
        "validation.result_consistency.lookahead_shuffle_test",
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
    parsed = int(value)
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
    parsed = float(value)
    if min_value is not None and parsed < min_value:
        raise ValueError(f"`{prefix}.{key}` must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"`{prefix}.{key}` must be <= {max_value}")
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
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
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
        parsed_raw, prefix, "min_data_points", min_value=0
    )
    continuity_cfg = _parse_continuity(
        parsed_raw.get("continuity"), f"{prefix}.continuity"
    )
    kurtosis_cfg = parse_optional_float(
        parsed_raw, prefix, "kurtosis", min_value=0
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
            continuity=continuity_cfg,
            kurtosis=kurtosis_cfg,
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
    min_score = parse_optional_float(parsed_raw, prefix, "min_score", min_value=0, max_value=1)
    max_missing = parse_optional_float(
        parsed_raw, prefix, "max_missing_bar_pct", min_value=0, max_value=100
    )
    calendar_cfg = _parse_validation_calendar(parsed_raw.get("calendar"), f"{prefix}.calendar")
    return ValidationContinuityConfig(
        min_score=min_score,
        max_missing_bar_pct=max_missing,
        calendar=calendar_cfg,
    )

def _parse_outlier_detection(
    raw: Any, prefix: str
) -> ValidationOutlierDetectionConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    max_outlier_pct = parse_required_float(
        parsed_raw, prefix, "max_outlier_pct", min_value=0, max_value=100
    )
    method = parse_required_str(parsed_raw, prefix, "method")
    if method not in {"zscore", "modified_zscore"}:
        raise ValueError(
            f"Invalid `{prefix}.method`: expected one of ['modified_zscore', 'zscore']"
        )
    zscore_threshold = parse_required_float(parsed_raw, prefix, "zscore_threshold")
    if zscore_threshold <= 0:
        raise ValueError(f"`{prefix}.zscore_threshold` must be > 0")
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
    window = parse_required_int(parsed_raw, prefix, "window", min_value=10)
    mean_shift_max = parse_required_float(parsed_raw, prefix, "mean_shift_max", min_value=0.0)
    vol_ratio_max = parse_required_float(parsed_raw, prefix, "vol_ratio_max", min_value=1.0)
    return ValidationStationarityRegimeShiftConfig(
        window=window,
        mean_shift_max=float(mean_shift_max),
        vol_ratio_max=float(vol_ratio_max),
    )


def _parse_stationarity(
    raw: Any, prefix: str
) -> ValidationStationarityConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    adf_pvalue_max = parse_required_float(
        parsed_raw, prefix, "adf_pvalue_max", min_value=0.0, max_value=1.0
    )
    kpss_pvalue_min = parse_optional_float(
        parsed_raw, prefix, "kpss_pvalue_min", min_value=0.0, max_value=1.0
    )
    min_points = parse_optional_int(parsed_raw, prefix, "min_points", min_value=20)
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
    permutations = parse_optional_int(parsed_raw, prefix, "permutations", min_value=5)
    threshold = parse_optional_float(parsed_raw, prefix, "threshold")
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError(f"`{prefix}.threshold` must be finite")
    seed = parse_optional_int(parsed_raw, prefix, "seed", min_value=0)
    max_failed_permutations = parse_optional_int(
        parsed_raw, prefix, "max_failed_permutations", min_value=0
    )
    return ValidationLookaheadShuffleTestConfig(
        permutations=permutations,
        threshold=threshold,
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

    min_bars = parse_required_int(parsed_raw, prefix, "min_bars", min_value=0)
    dof_multiplier = parse_required_int(parsed_raw, prefix, "dof_multiplier", min_value=0)
    runtime_error_max_per_tuple = parse_optional_int(
        parsed_raw, prefix, "runtime_error_max_per_tuple", min_value=1
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

    return _normalize_result_consistency_config(
        ResultConsistencyConfig(
            outlier_dependency=outlier_dependency,
            execution_price_variance=execution_price_variance,
            lookahead_shuffle_test=lookahead_shuffle_test,
        ),
        prefix,
    )


def _parse_result_consistency_outlier_dependency(
    raw: Any, prefix: str
) -> ResultConsistencyOutlierDependencyConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    slices = parse_required_int(parsed_raw, prefix, "slices", min_value=2)
    profit_share_threshold = parse_required_float(
        parsed_raw,
        prefix,
        "profit_share_threshold",
        min_value=0.0,
        max_value=1.0,
    )
    trade_share_threshold = parse_required_float(
        parsed_raw,
        prefix,
        "trade_share_threshold",
        min_value=0.0,
        max_value=1.0,
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
        min_value=0.0,
    )
    return ResultConsistencyExecutionPriceVarianceConfig(
        price_tolerance_bps=float(price_tolerance_bps),
    )


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

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

    collections: list[CollectionConfig] = []
    for idx, c in enumerate(raw["collections"]):
        collection_validation = _parse_validation(c.get("validation"), f"collections[{idx}].validation")
        collections.append(
            CollectionConfig(
                name=c["name"],
                source=c["source"],
                symbols=c["symbols"],
                exchange=c.get("exchange"),
                currency=c.get("currency"),
                quote=c.get("quote"),
                fees=c.get("fees"),
                slippage=c.get("slippage"),
                validation=collection_validation,
            )
        )

    strategies = [
        StrategyConfig(
            name=s["name"],
            module=s.get("module"),
            cls=s.get("class") or s.get("cls"),
            params=s.get("params", {}),
        )
        for s in strategies_raw
    ]

    notifications_cfg = None
    notifications_raw = raw.get("notifications")
    if isinstance(notifications_raw, dict):
        slack_raw = notifications_raw.get("slack")
        slack_cfg = None
        if isinstance(slack_raw, dict) and slack_raw.get("webhook_url"):
            slack_cfg = SlackNotificationConfig(
                webhook_url=slack_raw["webhook_url"],
                metric=slack_raw.get("metric", raw.get("metric", "sharpe")),
                threshold=slack_raw.get("threshold"),
                channel=slack_raw.get("channel"),
                username=slack_raw.get("username"),
            )
        if slack_cfg is not None:
            notifications_cfg = NotificationsConfig(slack=slack_cfg)

    validation_cfg = _parse_validation(raw.get("validation"), "validation")

    evaluation_mode = str(raw.get("evaluation_mode", "backtest")).strip().lower()
    allowed_modes = {"backtest", "walk_forward"}
    if evaluation_mode not in allowed_modes:
        raise ValueError(
            f"Invalid `evaluation_mode`: expected one of {sorted(allowed_modes)}, got '{evaluation_mode}'"
        )

    cfg = Config(
        collections=collections,
        timeframes=raw["timeframes"],
        metric=raw.get("metric", "sharpe").lower(),
        strategies=strategies,
        engine=str(raw.get("engine", "pybroker")).lower(),
        param_search=str(raw.get("param_search", raw.get("param_optimizer", "grid"))).lower(),
        param_trials=int(raw.get("param_trials", raw.get("opt_trials", 25))),
        max_workers=int(raw.get("max_workers", raw.get("asset_workers", 1))),
        asset_workers=int(raw.get("asset_workers", raw.get("max_workers", 1))),
        param_workers=int(raw.get("param_workers", 1)),
        max_fetch_concurrency=int(raw.get("max_fetch_concurrency", 2)),
        fees=float(raw.get("fees", 0.0)),
        slippage=float(raw.get("slippage", 0.0)),
        risk_free_rate=float(raw.get("risk_free_rate", 0.0)),
        cache_dir=raw.get("cache_dir", ".cache/data"),
        evaluation_mode=evaluation_mode,
        notifications=notifications_cfg,
        validation=validation_cfg,
    )
    resolve_validation_overrides(cfg)
    return cfg
