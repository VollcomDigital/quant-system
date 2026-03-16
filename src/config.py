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
    continuity: "ValidationContinuityConfig | None" = None
    kurtosis: float | None = None
    outlier_detection: "ValidationOutlierDetectionConfig | None" = None
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


DEFAULT_CALENDAR_KIND = "auto"
def _merge_replace(base: Any, override: Any) -> Any:
    return override if override is not None else base


def _merged_field(base: Any, override: Any, field: str) -> Any:
    override_value = getattr(override, field, None)
    if override_value is not None:
        return override_value
    return getattr(base, field, None)


def _merge_data_quality_config(
    base: ValidationDataQualityConfig | None,
    override: ValidationDataQualityConfig | None,
) -> ValidationDataQualityConfig | None:
    if base is None and override is None:
        return None
    min_data_points = _merge_replace(
        getattr(base, "min_data_points", None),
        getattr(override, "min_data_points", None),
    )
    continuity = _merge_continuity_config(
        getattr(base, "continuity", None),
        getattr(override, "continuity", None),
    )
    kurtosis = _merge_replace(
        getattr(base, "kurtosis", None),
        getattr(override, "kurtosis", None),
    )
    outlier_detection = _merge_outlier_detection_config(
        getattr(base, "outlier_detection", None),
        getattr(override, "outlier_detection", None),
    )

    on_fail = _merged_field(base, override, "on_fail")
    if on_fail is None:
        raise ValueError("Invalid `validation.data_quality`: missing required field(s): on_fail")
    return ValidationDataQualityConfig(
        min_data_points=min_data_points,
        continuity=continuity,
        kurtosis=kurtosis,
        outlier_detection=outlier_detection,
        on_fail=str(on_fail).strip().lower(),
    )

def _merge_continuity_config(
    base: ValidationContinuityConfig | None,
    override: ValidationContinuityConfig | None,
) -> ValidationContinuityConfig | None:
    if base is None and override is None:
        return None
    min_score = _merged_field(base, override, "min_score")
    max_missing_bar_pct = _merged_field(base, override, "max_missing_bar_pct")
    calendar = _merge_calendar_config(
        _merged_field(base, None, "calendar"),
        _merged_field(None, override, "calendar"),
    )
    return ValidationContinuityConfig(
        min_score=min_score,
        max_missing_bar_pct=max_missing_bar_pct,
        calendar=calendar,
    )


def _merge_calendar_config(
    base: ValidationCalendarConfig | None,
    override: ValidationCalendarConfig | None,
) -> ValidationCalendarConfig:
    kind = str(_merged_field(base, override, "kind") or DEFAULT_CALENDAR_KIND).strip().lower()
    exchange = _merged_field(base, override, "exchange")
    timezone = _merged_field(base, override, "timezone")
    return ValidationCalendarConfig(kind=kind, exchange=exchange, timezone=timezone)


def _merge_outlier_detection_config(
    base: ValidationOutlierDetectionConfig | None,
    override: ValidationOutlierDetectionConfig | None,
) -> ValidationOutlierDetectionConfig | None:
    if base is None and override is None:
        return None

    max_outlier_pct = _merged_field(base, override, "max_outlier_pct")
    if max_outlier_pct is None:
        raise ValueError(
            "Invalid `validation.data_quality.outlier_detection`: missing required field(s): max_outlier_pct"
        )
    method = _merged_field(base, override, "method")
    if method is None:
        raise ValueError(
            "Invalid `validation.data_quality.outlier_detection`: missing required field(s): method"
        )
    zscore_threshold = _merged_field(base, override, "zscore_threshold")
    if zscore_threshold is None:
        raise ValueError(
            "Invalid `validation.data_quality.outlier_detection`: missing required field(s): zscore_threshold"
        )
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=float(max_outlier_pct),
        method=str(method),
        zscore_threshold=float(zscore_threshold),
    )


def resolve_validation_overrides(cfg: Config) -> Config:
    """Resolve effective `validation.data_quality` per collection.

    Rule: effective policy = merge(global_data_quality_policy, collection_data_quality_override).
    """
    validation_cfg = cfg.validation
    if validation_cfg is None:
        return cfg
    global_data_quality_policy = validation_cfg.data_quality
    if global_data_quality_policy is not None:
        validation_cfg.data_quality = _merge_data_quality_config(global_data_quality_policy, None)

    for collection in cfg.collections:
        collection_validation_cfg = collection.validation
        collection_data_quality_override = (
            getattr(collection_validation_cfg, "data_quality", None)
            if collection_validation_cfg is not None
            else None
        )
        if collection_data_quality_override is None:
            if global_data_quality_policy is None:
                continue
            # No collection override: inherit the normalized global data-quality policy.
            if collection.validation is None:
                collection.validation = ValidationConfig(
                    data_quality=_merge_data_quality_config(global_data_quality_policy, None),
                    optimization=None,
                )
            else:
                collection.validation.data_quality = _merge_data_quality_config(
                    global_data_quality_policy, None
                )
            continue
        if global_data_quality_policy is None:
            # Collection-only data-quality policy: normalize and keep as effective policy.
            collection.validation.data_quality = _merge_data_quality_config(
                collection_data_quality_override, None
            )
            continue
        # Both global and collection are set: collection values override global field-by-field.
        collection.validation.data_quality = _merge_data_quality_config(
            global_data_quality_policy, collection_data_quality_override
        )
    return cfg


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

    cfg = ValidationDataQualityConfig(
        min_data_points=min_data_points_cfg,
        continuity=continuity_cfg,
        kurtosis=kurtosis_cfg,
        outlier_detection=outlier_detection_cfg,
        on_fail=on_fail,
    )
    return cfg

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
    method = parse_optional_str(parsed_raw, "method")
    if method is None:
        raise ValueError(f"Invalid `{prefix}`: missing required field(s): method")
    if method not in {"zscore", "modified_zscore"}:
        raise ValueError(
            f"Invalid `{prefix}.method`: expected one of ['modified_zscore', 'zscore']"
        )
    zscore_threshold = parse_required_float(parsed_raw, prefix, "zscore_threshold", min_value=0.0)
    if zscore_threshold <= 0:
        raise ValueError(f"`{prefix}.zscore_threshold` must be > 0")
    return ValidationOutlierDetectionConfig(
        max_outlier_pct=max_outlier_pct,
        method=method,
        zscore_threshold=zscore_threshold,
    )


def _parse_validation_calendar(raw: Any, prefix: str) -> ValidationCalendarConfig | None:
    if raw is None:
        return None
    parsed_raw = require_mapping(raw, prefix)
    kind = str(parsed_raw.get("kind", "auto")).strip().lower()
    allowed_kinds = {"auto", "crypto_24_7", "weekday", "exchange"}
    if kind not in allowed_kinds:
        raise ValueError(
            f"Invalid `{prefix}.kind`: expected one of {sorted(allowed_kinds)}, got '{kind}'"
        )
    exchange = parsed_raw.get("exchange")
    return ValidationCalendarConfig(
        kind=kind,
        exchange=str(exchange).strip() if exchange is not None else None,
        timezone=_parse_utc_timezone(parsed_raw.get("timezone"), f"{prefix}.timezone"),
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

    if data_quality_cfg is None and optimization_cfg is None:
        # Keep validation fully optional; no synthetic policy object when both modules are absent.
        return None
    return ValidationConfig(data_quality=data_quality_cfg, optimization=optimization_cfg)


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
    return resolve_validation_overrides(cfg)
