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
    notifications: NotificationsConfig | None = None


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

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
        for s in raw["strategies"]
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
        notifications=notifications_cfg,
    )
    return cfg
