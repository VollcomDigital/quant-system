from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class ValidationDataQualityConfig:
    min_data_points: int | None = None
    max_missing_bar_pct: float | None = None
    max_kurtosis: float | None = None
    min_continuity_score: float | None = None
    on_fail: str | None = None


@dataclass
class ValidationConfig:
    data_quality: ValidationDataQualityConfig | None = None


@dataclass
class OptimizationPolicyConfig:
    on_fail: str = "baseline_only"
    min_bars: int | None = None
    dof_multiplier: int | None = None


@dataclass
class Config:
    collections: list[CollectionConfig]
    timeframes: list[str]
    metric: str  # sharpe | sortino | profit
    strategies: list[StrategyConfig]
    engine: str = "pybroker"  # pybroker engine
    param_search: str = "grid"  # grid | optuna
    param_trials: int = 25
    dof_multiplier: int = 100
    min_bars: int = 2000
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
    optimization_policy: OptimizationPolicyConfig | None = None


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
        min_continuity_score=_as_optional_float(
            raw.get("min_continuity_score"), "min_continuity_score"
        ),
        on_fail=on_fail,
    )
    if cfg.min_continuity_score is not None and not 0.0 <= cfg.min_continuity_score <= 1.0:
        raise ValueError(f"`{prefix}.min_continuity_score` must be between 0 and 1")
    return cfg


def _parse_validation(raw: Any, prefix: str) -> ValidationConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")

    data_quality_raw = raw.get("data_quality")
    if data_quality_raw is not None and not isinstance(data_quality_raw, dict):
        raise ValueError(f"Invalid `{prefix}.data_quality`: expected a mapping")

    data_quality_cfg = None
    if isinstance(data_quality_raw, dict):
        parsed_data_quality = _parse_validation_data_quality_thresholds(
            data_quality_raw, f"{prefix}.data_quality"
        )
        assert parsed_data_quality is not None  # data_quality_raw is dict
        data_quality_cfg = parsed_data_quality

    if data_quality_cfg is None:
        return None
    return ValidationConfig(data_quality=data_quality_cfg)


def _parse_optimization_policy(raw: Any, prefix: str) -> OptimizationPolicyConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid `{prefix}`: expected a mapping")

    on_fail = _parse_on_fail(
        raw.get("on_fail"),
        f"{prefix}.on_fail",
        {"baseline_only", "skip_job"},
    ) or "baseline_only"

    def _as_optional_int(value: Any, field: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"`{prefix}.{field}` must be >= 0")
        return parsed

    return OptimizationPolicyConfig(
        on_fail=on_fail,
        min_bars=_as_optional_int(raw.get("min_bars"), "min_bars"),
        dof_multiplier=_as_optional_int(raw.get("dof_multiplier"), "dof_multiplier"),
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
    optimization_policy_cfg = _parse_optimization_policy(
        raw.get("optimization_policy"), "optimization_policy"
    )

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
        dof_multiplier=int(raw.get("dof_multiplier", 100)),
        min_bars=int(raw.get("min_bars", 2000)),
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
        optimization_policy=optimization_policy_cfg,
    )
    return cfg
