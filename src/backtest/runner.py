from __future__ import annotations

import itertools
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Silence numba deprecation noise from vectorbt on Py3.10 stack
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPendingDeprecationWarning,
    )

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*generated_jit is deprecated.*", category=Warning)
except Exception:
    pass
import vectorbt as vbt

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
from ..utils.telemetry import get_logger, time_block
from .metrics import sharpe_ratio, sortino_ratio, total_return
from .results_cache import ResultsCache


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


class BacktestRunner:
    def __init__(self, cfg: Config, strategies_root: Path, run_id: str | None = None):
        self.cfg = cfg
        self.strategies_root = strategies_root
        self.external_index = discover_external_strategies(strategies_root)
        self.results_cache = ResultsCache(Path(self.cfg.cache_dir).parent / "results")
        self.run_id = run_id
        self.logger = get_logger()

    def _make_source(self, col: CollectionConfig) -> DataSource:
        cache_dir = Path(self.cfg.cache_dir)
        src = col.source.lower()
        if src == "yfinance":
            return YFinanceSource(cache_dir)
        if src in ("ccxt", "binance", "bybit"):
            if not col.exchange:
                # Allow shorthand where source is the exchange name
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
        # Defaults: IBKR-like for traditional markets, Bybit-like for crypto
        if col.fees is not None or col.slippage is not None:
            return (
                col.fees if col.fees is not None else self.cfg.fees,
                col.slippage if col.slippage is not None else self.cfg.slippage,
            )
        src = col.source.lower()
        if src in ("binance", "bybit", "ccxt"):
            return (0.0006, 0.0005)  # approx taker + small slippage
        # yfinance/polygon/tiingo/alpaca stocks/etfs
        return (0.0005, 0.0005)

    def _grid(self, grid: dict[str, list[Any]]):
        if not grid:
            yield {}
            return
        keys = list(grid.keys())
        for values in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, values, strict=False))

    def _evaluate_metric(self, metric: str, returns: pd.Series, equity: pd.Series) -> float:
        metric = metric.lower()
        if metric == "sharpe":
            return sharpe_ratio(returns, risk_free_rate=self.cfg.risk_free_rate)
        if metric == "sortino":
            return sortino_ratio(returns, risk_free_rate=self.cfg.risk_free_rate)
        if metric == "profit":
            return total_return(equity)
        raise ValueError(f"Unknown metric: {metric}")

    def run_all(self, only_cached: bool = False) -> list[BestResult]:
        best_results: list[BestResult] = []
        self.metrics = {
            "result_cache_hits": 0,
            "result_cache_misses": 0,
            "param_evals": 0,
            "symbols_tested": 0,
            "strategies_used": set(),
        }
        # Collect per-symbol failures (e.g., data fetch issues)
        self.failures: list[dict[str, Any]] = []

        overrides = {s.name: s.params for s in self.cfg.strategies} if self.cfg.strategies else {}

        # Global fetch concurrency control
        fetch_sema = threading.Semaphore(max(1, getattr(self.cfg, "max_fetch_concurrency", 2)))
        metrics_lock = threading.Lock()
        results_lock = threading.Lock()

        jobs: list[tuple[CollectionConfig, str, str, str]] = []
        for col in self.cfg.collections:
            for symbol in col.symbols:
                for timeframe in self.cfg.timeframes:
                    for name in self.external_index.keys():
                        jobs.append((col, symbol, timeframe, name))

        def run_job(job: tuple[CollectionConfig, str, str, str]):
            col, symbol, timeframe, strat_name = job
            StrategyClass = self.external_index[strat_name]
            strat: BaseStrategy = StrategyClass()
            base_params = overrides.get(strat_name, {}) if overrides else {}
            grid_override = base_params.get("grid") if isinstance(base_params, dict) else None
            if isinstance(grid_override, dict):
                grid = grid_override
                static_params = {k: v for k, v in base_params.items() if k != "grid"}
            else:
                grid = strat.param_grid() | base_params
                static_params = {}
            source = self._make_source(col)
            # Fetch with global semaphore + timing
            with fetch_sema:
                with time_block(
                    self.logger,
                    "data_fetch",
                    collection=col.name,
                    symbol=symbol,
                    timeframe=timeframe,
                    source=col.source,
                ):
                    try:
                        df = source.fetch(symbol, timeframe, only_cached=only_cached)
                    except Exception as e:
                        # Record failure and skip this (collection, symbol, timeframe)
                        with metrics_lock:
                            self.failures.append(
                                {
                                    "collection": col.name,
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "source": col.source,
                                    "error": str(e),
                                }
                            )
                        return
            price = df["Close"].astype(float)
            data_fingerprint = f"{len(df)}:{df.index[-1].isoformat()}:{float(price.iloc[-1])}"
            fees_use, slippage_use = self._fees_slippage_for(col)

            best_val = -np.inf
            best: tuple[dict[str, Any], dict[str, Any]] | None = None
            with metrics_lock:
                self.metrics["symbols_tested"] += 1
                self.metrics["strategies_used"].add(strat.name)

            def eval_params(params: dict[str, Any]):
                call_params = {**static_params, **params}
                entries, exits = strat.generate_signals(df, call_params)
                entries = entries.reindex(df.index).fillna(False)
                exits = exits.reindex(df.index).fillna(False)
                try:
                    pf = vbt.Portfolio.from_signals(
                        price,
                        entries,
                        exits,
                        fees=fees_use,
                        slippage=slippage_use,
                        init_cash=10000.0,
                    )
                except Exception:
                    return params, None, None
                returns = pf.returns()
                equity = (1 + returns).cumprod()
                return params, returns, equity

            with time_block(
                self.logger,
                "grid_search",
                collection=col.name,
                symbol=symbol,
                timeframe=timeframe,
                strategy=strat.name,
            ):
                futures = []
                with ThreadPoolExecutor(
                    max_workers=max(1, getattr(self.cfg, "param_workers", 1))
                ) as ex:
                    for params in self._grid(grid):
                        cached = self.results_cache.get(
                            collection=col.name,
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy=strat.name,
                            params=params,
                            metric_name=self.cfg.metric,
                            data_fingerprint=data_fingerprint,
                            fees=fees_use,
                            slippage=slippage_use,
                        )
                        if cached is not None:
                            with metrics_lock:
                                self.metrics["result_cache_hits"] += 1
                            val = float(cached["metric_value"])
                            # Record this cached evaluation under current run_id
                            try:
                                self.results_cache.set(
                                    collection=col.name,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    strategy=strat.name,
                                    params=params,
                                    metric_name=self.cfg.metric,
                                    metric_value=val,
                                    stats=cached["stats"],
                                    data_fingerprint=data_fingerprint,
                                    fees=fees_use,
                                    slippage=slippage_use,
                                    run_id=self.run_id,
                                )
                            except Exception:
                                pass
                            if val > best_val:
                                best_val = val
                                best = (params, cached["stats"])
                            continue
                        with metrics_lock:
                            self.metrics["result_cache_misses"] += 1
                        futures.append(ex.submit(eval_params, params))

                    for fut in as_completed(futures):
                        params, returns, equity = fut.result()
                        if returns is None or equity is None:
                            continue
                        with metrics_lock:
                            self.metrics["param_evals"] += 1
                        val = self._evaluate_metric(self.cfg.metric, returns, equity)
                        if np.isnan(val):
                            continue
                        if val > best_val:
                            best_val = val
                            roll_max = equity.cummax()
                            dd = (equity / roll_max) - 1.0
                            max_dd = float(dd.min())
                            trades = int((returns != 0).sum())
                            # Duration and CAGR for Calmar
                            try:
                                days = max(1, (equity.index[-1] - equity.index[0]).days)
                                years = max(1e-9, days / 365.25)
                                ending = float(equity.iloc[-1])
                                cagr = (
                                    (ending ** (1.0 / years)) - 1.0 if ending > 0 else float("nan")
                                )
                            except Exception:
                                cagr = float("nan")
                            calmar = float("nan")
                            if max_dd < 0:
                                try:
                                    calmar = float(cagr / abs(max_dd))
                                except Exception:
                                    calmar = float("nan")
                            stats = {
                                "sharpe": float(sharpe_ratio(returns)),
                                "sortino": float(sortino_ratio(returns)),
                                "profit": float(total_return(equity)),
                                "trades": trades,
                                "max_drawdown": max_dd,
                                "cagr": float(cagr),
                                "calmar": float(calmar),
                            }
                            self.results_cache.set(
                                collection=col.name,
                                symbol=symbol,
                                timeframe=timeframe,
                                strategy=strat.name,
                                params=params,
                                metric_name=self.cfg.metric,
                                metric_value=float(best_val),
                                stats=stats,
                                data_fingerprint=data_fingerprint,
                                fees=fees_use,
                                slippage=slippage_use,
                                run_id=self.run_id,
                            )
                            best = (params, stats)

            if best is not None:
                with results_lock:
                    params_best, stats_best = best
                    best_results.append(
                        BestResult(
                            collection=col.name,
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy=strat.name,
                            params=params_best,
                            metric_name=self.cfg.metric,
                            metric_value=float(best_val),
                            stats=stats_best,
                        )
                    )

        # Execute jobs with global executor
        with ThreadPoolExecutor(max_workers=max(1, getattr(self.cfg, "asset_workers", 1))) as ex:
            list(as_completed([ex.submit(run_job, job) for job in jobs]))

        if isinstance(self.metrics.get("strategies_used"), set):
            self.metrics["strategies_count"] = len(self.metrics["strategies_used"])  # type: ignore
            self.metrics.pop("strategies_used", None)
        return best_results
