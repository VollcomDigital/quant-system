from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
    kind: str = "auto"
    exchange: str | None = None
    timezone: str | None = None


@dataclass
class ValidationDataQualityConfig:
    min_data_points: int | None = None
    max_missing_bar_pct: float | None = None
    max_kurtosis: float | None = None
    max_outlier_pct: float | None = None
    outlier_method: str | None = None
    outlier_zscore_threshold: float | None = None
    min_continuity_score: float | None = None
    on_fail: str | None = None
    calendar: ValidationCalendarConfig | None = None


@dataclass
class ValidationConfig:
    data_quality: ValidationDataQualityConfig | None = None
    optimization: "OptimizationPolicyConfig | None" = None


@dataclass
class OptimizationPolicyConfig:
    on_fail: str
    min_bars: int
    dof_multiplier: int


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


DEFAULT_DATA_QUALITY_ON_FAIL = "skip_optimization"
DEFAULT_CALENDAR_KIND = "auto"
DEFAULT_OUTLIER_METHOD = "modified_zscore"
DEFAULT_ZSCORE_THRESHOLD = 3.0
DEFAULT_MODIFIED_ZSCORE_THRESHOLD = 3.5


def _default_outlier_threshold(method: str) -> float:
    return DEFAULT_ZSCORE_THRESHOLD if method == "zscore" else DEFAULT_MODIFIED_ZSCORE_THRESHOLD


def _merge_data_quality_config(
    base: ValidationDataQualityConfig | None,
    override: ValidationDataQualityConfig | None,
) -> ValidationDataQualityConfig:
    def _pick(field: str) -> Any:
        if override is not None:
            override_value = getattr(override, field)
            if override_value is not None:
                return override_value
        if base is not None:
            return getattr(base, field)
        return None

    calendar_base = base.calendar if base is not None else None
    calendar_override = override.calendar if override is not None else None
    calendar_kind = (
        str(calendar_override.kind).strip().lower()
        if calendar_override is not None and calendar_override.kind is not None
        else (
            str(calendar_base.kind).strip().lower()
            if calendar_base is not None and calendar_base.kind is not None
            else DEFAULT_CALENDAR_KIND
        )
    )
    calendar_exchange = (
        calendar_override.exchange
        if calendar_override is not None and calendar_override.exchange is not None
        else (calendar_base.exchange if calendar_base is not None else None)
    )
    calendar_timezone = (
        calendar_override.timezone
        if calendar_override is not None and calendar_override.timezone is not None
        else (calendar_base.timezone if calendar_base is not None else None)
    )

    outlier_method = _pick("outlier_method")
    method = (
        str(outlier_method).strip().lower()
        if outlier_method is not None
        else DEFAULT_OUTLIER_METHOD
    )
    outlier_threshold = _pick("outlier_zscore_threshold")
    threshold = (
        float(outlier_threshold)
        if outlier_threshold is not None
        else _default_outlier_threshold(method)
    )

    on_fail = _pick("on_fail")
    return ValidationDataQualityConfig(
        min_data_points=_pick("min_data_points"),
        max_missing_bar_pct=_pick("max_missing_bar_pct"),
        max_kurtosis=_pick("max_kurtosis"),
        max_outlier_pct=_pick("max_outlier_pct"),
        outlier_method=method,
        outlier_zscore_threshold=threshold,
        min_continuity_score=_pick("min_continuity_score"),
        on_fail=(str(on_fail).strip().lower() if on_fail is not None else DEFAULT_DATA_QUALITY_ON_FAIL),
        calendar=ValidationCalendarConfig(
            kind=calendar_kind,
            exchange=calendar_exchange,
            timezone=calendar_timezone,
        ),
    )


def normalize_validation_defaults(cfg: Config) -> Config:
    """Fill effective validation defaults in config so runtime can trust policy values."""
    if cfg.validation is None:
        cfg.validation = ValidationConfig()
    cfg.validation.data_quality = _merge_data_quality_config(cfg.validation.data_quality, None)
    global_dq = cfg.validation.data_quality

    for collection in cfg.collections:
        if collection.validation is None:
            continue
        collection_dq = collection.validation.data_quality
        if collection_dq is None:
            continue
        collection.validation.data_quality = _merge_data_quality_config(global_dq, collection_dq)
    return cfg


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


def _parse_validation_data_quality_thresholds(
    raw: Any, prefix: str
) -> ValidationDataQualityConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")

    on_fail = _parse_on_fail(
        raw.get("on_fail"),
        f"{prefix}.on_fail",
        {"skip_optimization", "skip_job", "skip_collection"},
    )
    calendar_cfg = _parse_validation_calendar(raw.get("calendar"), f"{prefix}.calendar")

    def _as_optional_int(value: Any, field: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"`{prefix}.{field}` must be >= 0")
        return parsed

    def _as_optional_float(value: Any, field: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if parsed < 0:
            raise ValueError(f"`{prefix}.{field}` must be >= 0")
        return parsed

    cfg = ValidationDataQualityConfig(
        min_data_points=_as_optional_int(raw.get("min_data_points"), "min_data_points"),
        max_missing_bar_pct=_as_optional_float(raw.get("max_missing_bar_pct"), "max_missing_bar_pct"),
        max_kurtosis=_as_optional_float(raw.get("max_kurtosis"), "max_kurtosis"),
        max_outlier_pct=_as_optional_float(raw.get("max_outlier_pct"), "max_outlier_pct"),
        outlier_method=(
            str(raw.get("outlier_method")).strip().lower()
            if raw.get("outlier_method") is not None
            else None
        ),
        outlier_zscore_threshold=_as_optional_float(
            raw.get("outlier_zscore_threshold"), "outlier_zscore_threshold"
        ),
        min_continuity_score=_as_optional_float(
            raw.get("min_continuity_score"), "min_continuity_score"
        ),
        on_fail=on_fail,
        calendar=calendar_cfg,
    )
    if cfg.min_continuity_score is not None and not 0.0 <= cfg.min_continuity_score <= 1.0:
        raise ValueError(f"`{prefix}.min_continuity_score` must be between 0 and 1")
    if cfg.max_outlier_pct is not None and not 0.0 <= cfg.max_outlier_pct <= 100.0:
        raise ValueError(f"`{prefix}.max_outlier_pct` must be between 0 and 100")
    if cfg.outlier_method is not None and cfg.outlier_method not in {
        "zscore",
        "modified_zscore",
    }:
        raise ValueError(
            f"Invalid `{prefix}.outlier_method`: expected one of ['modified_zscore', 'zscore']"
        )
    if cfg.outlier_zscore_threshold is not None and cfg.outlier_zscore_threshold <= 0:
        raise ValueError(f"`{prefix}.outlier_zscore_threshold` must be > 0")
    return cfg


def _parse_validation_calendar(raw: Any, prefix: str) -> ValidationCalendarConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")
    kind = str(raw.get("kind", "auto")).strip().lower()
    allowed_kinds = {"auto", "crypto_24_7", "weekday", "exchange"}
    if kind not in allowed_kinds:
        raise ValueError(
            f"Invalid `{prefix}.kind`: expected one of {sorted(allowed_kinds)}, got '{kind}'"
        )
    exchange = raw.get("exchange")
    return ValidationCalendarConfig(
        kind=kind,
        exchange=str(exchange).strip() if exchange is not None else None,
        timezone=_parse_utc_timezone(raw.get("timezone"), f"{prefix}.timezone"),
    )


def _parse_validation(raw: Any, prefix: str) -> ValidationConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")

    data_quality_raw = raw.get("data_quality")
    if data_quality_raw is not None and not isinstance(data_quality_raw, dict):
        raise ValueError(f"Invalid `{prefix}.data_quality`: expected a mapping")
    optimization_raw = raw.get("optimization")
    if optimization_raw is not None and not isinstance(optimization_raw, dict):
        raise ValueError(f"Invalid `{prefix}.optimization`: expected a mapping")

    data_quality_cfg = (
        _parse_validation_data_quality_thresholds(data_quality_raw, f"{prefix}.data_quality")
        if isinstance(data_quality_raw, dict)
        else None
    )
    optimization_cfg = (
        _parse_optimization_policy(optimization_raw, f"{prefix}.optimization")
        if isinstance(optimization_raw, dict)
        else None
    )

    if data_quality_cfg is None and optimization_cfg is None:
        return None
    return ValidationConfig(data_quality=data_quality_cfg, optimization=optimization_cfg)


def _parse_optimization_policy(raw: Any, prefix: str) -> OptimizationPolicyConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")

    required_fields = ("on_fail", "min_bars", "dof_multiplier")
    missing_fields = [field for field in required_fields if raw.get(field) is None]
    if missing_fields:
        raise ValueError(
            f"Invalid `{prefix}`: missing required field(s): {', '.join(missing_fields)}"
        )

    on_fail = cast(
        str,
        _parse_on_fail(
            raw.get("on_fail"),
            f"{prefix}.on_fail",
            {"baseline_only", "skip_job"},
        ),
    )

    def _as_optional_int(value: Any, field: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"`{prefix}.{field}` must be >= 0")
        return parsed

    min_bars = _as_optional_int(raw.get("min_bars"), "min_bars")
    dof_multiplier = _as_optional_int(raw.get("dof_multiplier"), "dof_multiplier")
    if min_bars is None or dof_multiplier is None:
        raise ValueError(
            f"Invalid `{prefix}`: `min_bars` and `dof_multiplier` are required"
        )

    return OptimizationPolicyConfig(
        on_fail=on_fail,
        min_bars=min_bars,
        dof_multiplier=dof_multiplier,
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
    return normalize_validation_defaults(cfg)
