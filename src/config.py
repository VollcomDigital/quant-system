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
class ReliabilityThresholdsConfig:
    min_data_points: int | None = None
    max_missing_bar_pct: float | None = None
    max_kurtosis: float | None = None
    min_continuity_score: float | None = None
    on_fail: str = "skip_optimization"


@dataclass
class Config:
    collections: list[CollectionConfig]
    timeframes: list[str]
    metric: str  # sharpe | sortino | profit
    strategies: list[StrategyConfig]
    engine: str = "pybroker"  # pybroker engine
    param_search: str = "grid"  # grid | optuna
    param_trials: int = 25
    param_dof_multiplier: int = 100
    param_min_bars: int = 2000
    max_workers: int = 1
    asset_workers: int = 1
    param_workers: int = 1
    max_fetch_concurrency: int = 2
    fees: float = 0.0
    slippage: float = 0.0
    risk_free_rate: float = 0.0
    cache_dir: str = ".cache/data"
    notifications: NotificationsConfig | None = None
    reliability_thresholds: ReliabilityThresholdsConfig | None = None


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

    collections = [
        CollectionConfig(
            name=c["name"],
            source=c["source"],
            symbols=c["symbols"],
            exchange=c.get("exchange"),
            currency=c.get("currency"),
            quote=c.get("quote"),
            fees=c.get("fees"),
            slippage=c.get("slippage"),
        )
        for c in raw["collections"]
    ]

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

    reliability_cfg = None
    reliability_raw = raw.get("reliability_thresholds")
    if reliability_raw is not None:
        # Reliability thresholds are enforcement policy (separate from collection metadata).
        if not isinstance(reliability_raw, dict):
            raise ValueError("Invalid `reliability_thresholds`: expected a mapping")
        on_fail = str(reliability_raw.get("on_fail", "skip_optimization")).strip().lower()
        allowed_on_fail = {"skip_optimization", "skip_evaluation"}
        if on_fail not in allowed_on_fail:
            raise ValueError(
                "Invalid `reliability_thresholds.on_fail`: "
                f"expected one of {sorted(allowed_on_fail)}, got '{on_fail}'"
            )

        def _as_optional_int(value: Any, field: str) -> int | None:
            if value is None:
                return None
            parsed = int(value)
            if parsed < 0:
                raise ValueError(f"`reliability_thresholds.{field}` must be >= 0")
            return parsed

        def _as_optional_float(value: Any, field: str) -> float | None:
            if value is None:
                return None
            parsed = float(value)
            if parsed < 0:
                raise ValueError(f"`reliability_thresholds.{field}` must be >= 0")
            return parsed

        reliability_cfg = ReliabilityThresholdsConfig(
            min_data_points=_as_optional_int(
                reliability_raw.get("min_data_points"),
                "min_data_points",
            ),
            max_missing_bar_pct=_as_optional_float(
                reliability_raw.get("max_missing_bar_pct"), "max_missing_bar_pct"
            ),
            max_kurtosis=_as_optional_float(reliability_raw.get("max_kurtosis"), "max_kurtosis"),
            min_continuity_score=_as_optional_float(
                reliability_raw.get("min_continuity_score"), "min_continuity_score"
            ),
            on_fail=on_fail,
        )
        if (
            reliability_cfg.min_continuity_score is not None
            and not 0.0 <= reliability_cfg.min_continuity_score <= 1.0
        ):
            raise ValueError(
                "`reliability_thresholds.min_continuity_score` must be between 0 and 1"
            )

    cfg = Config(
        collections=collections,
        timeframes=raw["timeframes"],
        metric=raw.get("metric", "sharpe").lower(),
        strategies=strategies,
        engine=str(raw.get("engine", "pybroker")).lower(),
        param_search=str(raw.get("param_search", raw.get("param_optimizer", "grid"))).lower(),
        param_trials=int(raw.get("param_trials", raw.get("opt_trials", 25))),
        param_dof_multiplier=int(raw.get("param_dof_multiplier", 100)),
        param_min_bars=int(raw.get("param_min_bars", 2000)),
        max_workers=int(raw.get("max_workers", raw.get("asset_workers", 1))),
        asset_workers=int(raw.get("asset_workers", raw.get("max_workers", 1))),
        param_workers=int(raw.get("param_workers", 1)),
        max_fetch_concurrency=int(raw.get("max_fetch_concurrency", 2)),
        fees=float(raw.get("fees", 0.0)),
        slippage=float(raw.get("slippage", 0.0)),
        risk_free_rate=float(raw.get("risk_free_rate", 0.0)),
        cache_dir=raw.get("cache_dir", ".cache/data"),
        notifications=notifications_cfg,
        reliability_thresholds=reliability_cfg,
    )
    return cfg
